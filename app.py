"""
Traffic Signal Control using Deep Reinforcement Learning
Double DQN Methodology  |  Real-Time Demonstration Dashboard

Runs a live traffic simulation across 7 Tamil Nadu intersections,
comparing a traditional fixed-timer controller against an AI
controller powered by Double DQN that learns in real-time.
"""

from flask import Flask, render_template, jsonify, Response, request
from flask_cors import CORS
import threading, time, json, random, os, pickle, signal, atexit, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

SAVE_DIR   = Path('checkpoints')
SAVE_DIR.mkdir(exist_ok=True)
SAVE_EVERY = 5          # auto-save every N episodes

# Single CSV file that logs all live simulation data
SIM_LOG_PATH = Path('data/simulation_log.csv')
SIM_LOG_COLUMNS = [
    'step', 'timestamp', 'episode', 'epsilon',
    'city', 'controller',
    'north_queue', 'south_queue', 'east_queue', 'west_queue',
    'signal_phase', 'timer', 'arrived', 'departed',
    'avg_wait', 'throughput', 'reward',
    'bikes', 'cars', 'buses', 'trucks',
]

# ═══════════════════════════════════════════════════════════
#  Tamil Nadu Intersection Config
# ═══════════════════════════════════════════════════════════

CITIES = {
    'Chennai':      {'lat': 13.0827, 'lng': 80.2707, 'intensity': 1.00, 'pop': '11.5M', 'type': 'Metro'},
    'Coimbatore':   {'lat': 11.0168, 'lng': 76.9558, 'intensity': 0.70, 'pop': '2.1M',  'type': 'City'},
    'Madurai':      {'lat':  9.9252, 'lng': 78.1198, 'intensity': 0.75, 'pop': '1.5M',  'type': 'City'},
    'Trichy':       {'lat': 10.7905, 'lng': 78.7047, 'intensity': 0.65, 'pop': '1.0M',  'type': 'City'},
    'Salem':        {'lat': 11.6643, 'lng': 78.1460, 'intensity': 0.55, 'pop': '0.9M',  'type': 'Town'},
    'Tambaram':     {'lat': 12.9249, 'lng': 80.1000, 'intensity': 0.80, 'pop': '0.8M',  'type': 'Suburb'},
    'Chengalpattu': {'lat': 12.6819, 'lng': 79.9888, 'intensity': 0.60, 'pop': '0.6M',  'type': 'Town'},
}

DIRECTIONS = ['North', 'South', 'East', 'West']

def tod_multiplier(step, cycle=96):
    h = (step % cycle) / cycle * 24
    if   7.5 <= h < 10:           return 1.6
    elif 12  <= h < 14:           return 1.2
    elif 16.5 <= h < 19:          return 1.8
    elif 22  <= h or h < 5:       return 0.3
    else:                         return 0.8

# ═══════════════════════════════════════════════════════════
#  Load dataset for realistic traffic intensity patterns
# ═══════════════════════════════════════════════════════════
import pandas as pd

DATASET_PATH = Path('data/traffic_dataset.csv')
DATASET_LOADED = False
CITY_HOUR_INTENSITY = {}   # {city: {hour_bin: mean_total_vehicles}}

if DATASET_PATH.exists():
    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = [c.strip() for c in df.columns]
        if 'city' in df.columns:
            df['city'] = df['city'].str.strip()
        # Extract hour from timestamp
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp'].str.strip()).dt.hour
        # Compute total vehicles per row
        veh_cols = [c for c in df.columns if 'vehicle_count' in c]
        if veh_cols:
            df['total_vehicles'] = df[veh_cols].sum(axis=1)
        # Build hour→intensity lookup from dataset
        for city in CITIES:
            cdf = df[df['city'] == city] if 'city' in df.columns else df
            if 'hour' in df.columns and 'total_vehicles' in df.columns:
                hourly = cdf.groupby('hour')['total_vehicles'].mean().to_dict()
            else:
                hourly = {}
            CITY_HOUR_INTENSITY[city] = hourly
        DATASET_LOADED = True
        print(f'  [✓] Dataset loaded: {len(df)} rows from {DATASET_PATH}')
    except Exception as e:
        print(f'  [warn] Could not load dataset: {e}')
else:
    print(f'  [i] No dataset at {DATASET_PATH}, using synthetic arrivals')


# ═══════════════════════════════════════════════════════════
#  Double DQN
# ═══════════════════════════════════════════════════════════

class DQNNetwork(nn.Module):
    def __init__(self, sdim=9, adim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, 128), nn.ReLU(),
            nn.Linear(128, 128),  nn.ReLU(),
            nn.Linear(128, adim),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, min(len(self.buf), n))
        return [np.array(x, dtype=np.float32) if i != 1 else np.array(x, dtype=np.int64)
                for i, x in enumerate(zip(*batch))]
    def __len__(self):
        return len(self.buf)

class DoubleDQNAgent:
    def __init__(self):
        self.policy = DQNNetwork()
        self.target = DQNNetwork()
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=0.003)
        self.mem = ReplayBuffer(10000)
        self.bs       = 32
        self.gamma    = 0.95
        self.eps      = 1.0
        self.eps_min  = 0.01
        self.eps_dec  = 0.97
        self.tgt_upd  = 5
        self.episode  = 0
        self.rw_hist  = []
        self.ls_hist  = []
        self.ep_hist  = []
        self.q_hist   = []
        self._qacc    = []

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 1)
        with torch.no_grad():
            q = self.policy(torch.FloatTensor(state).unsqueeze(0))
            self._qacc.append(q.max().item())
            return q.argmax(1).item()

    def learn(self):
        if len(self.mem) < self.bs:
            return 0.0
        s, a, r, ns, d = self.mem.sample(self.bs)
        s=torch.FloatTensor(s); a=torch.LongTensor(a).unsqueeze(1)
        r=torch.FloatTensor(r); ns=torch.FloatTensor(ns); d=torch.FloatTensor(d)
        cq = self.policy(s).gather(1, a).squeeze()
        with torch.no_grad():
            ba = self.policy(ns).argmax(1, keepdim=True)
            nq = self.target(ns).gather(1, ba).squeeze()
            tq = r + self.gamma * nq * (1 - d)
        loss = nn.SmoothL1Loss()(cq, tq)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def end_ep(self, rw, ls):
        self.episode += 1
        self.rw_hist.append(round(rw, 4))
        self.ls_hist.append(round(ls, 6))
        self.ep_hist.append(round(self.eps, 4))
        self.q_hist.append(round(float(np.mean(self._qacc)) if self._qacc else 0, 4))
        self._qacc = []
        self.eps = max(self.eps_min, self.eps * self.eps_dec)
        if self.episode % self.tgt_upd == 0:
            self.target.load_state_dict(self.policy.state_dict())

    # ── Persistence ──────────────────────────────────────
    def save(self, path):
        torch.save({
            'policy':    self.policy.state_dict(),
            'target':    self.target.state_dict(),
            'optimizer': self.opt.state_dict(),
            'episode':   self.episode,
            'epsilon':   self.eps,
            'rw_hist':   self.rw_hist,
            'ls_hist':   self.ls_hist,
            'ep_hist':   self.ep_hist,
            'q_hist':    self.q_hist,
        }, path)

    def load(self, path):
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.target.load_state_dict(ckpt['target'])
        self.opt.load_state_dict(ckpt['optimizer'])
        self.episode = ckpt['episode']
        self.eps     = ckpt['epsilon']
        self.rw_hist = ckpt['rw_hist']
        self.ls_hist = ckpt['ls_hist']
        self.ep_hist = ckpt['ep_hist']
        self.q_hist  = ckpt['q_hist']
        return True


# ═══════════════════════════════════════════════════════════
#  Intersection Model
# ═══════════════════════════════════════════════════════════

class Intersection:
    def __init__(self, city, cfg):
        self.city = city
        self.lat = cfg['lat']; self.lng = cfg['lng']
        self.intensity = cfg['intensity']
        self.queues  = {d: random.randint(0,4) for d in DIRECTIONS}
        self.phase   = 0
        self.timer   = 0
        self.arrived = 0
        self.departed= 0
        self.wait_acc= 0.0
        self.steps   = 0
        self.vehs    = {d: {'bikes':0,'cars':0,'buses':0,'trucks':0} for d in DIRECTIONS}

    def arrivals(self, rng, sim_hour=12):
        for d in DIRECTIONS:
            # Use dataset intensity if available for this city & hour
            ds_factor = 1.0
            if DATASET_LOADED and self.city in CITY_HOUR_INTENSITY:
                h_bin = int(sim_hour) % 24
                h_data = CITY_HOUR_INTENSITY[self.city]
                if h_bin in h_data:
                    ds_factor = h_data[h_bin] / 40.0   # normalise to ~1.0
                    ds_factor = max(0.2, min(ds_factor, 3.0))
            n = rng.poisson(self.intensity * 0.5 * ds_factor)
            self.queues[d] += n
            self.arrived += n
            for _ in range(n):
                r = rng.random()
                if   r < 0.55: self.vehs[d]['bikes']  += 1
                elif r < 0.85: self.vehs[d]['cars']   += 1
                elif r < 0.95: self.vehs[d]['buses']  += 1
                else:          self.vehs[d]['trucks'] += 1

    def state_vec(self):
        s = [min(self.queues[d]/20,1) for d in DIRECTIONS]
        ph=[0]*4; ph[self.phase]=1; s+=ph
        s.append(min(self.timer/30,1))
        return np.array(s, dtype=np.float32)

    def act(self, action):
        sw = False
        if action == 1 and self.timer >= 5:
            self.phase = (self.phase+1)%4; self.timer=0; sw=True
        g = DIRECTIONS[self.phase]
        dep = min(self.queues[g], random.randint(1,3))
        self.queues[g] = max(0, self.queues[g]-dep)
        self.departed += dep
        for _ in range(dep):
            for t in ('bikes','cars','buses','trucks'):
                if self.vehs[g][t]>0: self.vehs[g][t]-=1; break
        for d in DIRECTIONS:
            if d != g: self.wait_acc += self.queues[d]
        self.timer += 1; self.steps += 1
        tq = sum(self.queues.values())
        return -tq/20 - self.wait_acc/max(self.steps,1)/10 + dep/3 + (-0.15 if sw else 0)

    def signals(self):
        g=DIRECTIONS[self.phase]
        return {d:('green' if d==g else 'red') for d in DIRECTIONS}

    def metrics(self):
        s=max(self.steps,1)
        return {'total_queue': sum(self.queues.values()),
                'avg_wait': round(self.wait_acc/s,2),
                'throughput': round(self.departed/s*60,1),
                'arrived': self.arrived, 'departed': self.departed}

    def veh_summary(self):
        t={'bikes':0,'cars':0,'buses':0,'trucks':0}
        for d in DIRECTIONS:
            for k in t: t[k]+=self.vehs[d][k]
        return t


# ═══════════════════════════════════════════════════════════
#  Simulation Engine
# ═══════════════════════════════════════════════════════════

EP_LEN = 20
AGENT_PATH = SAVE_DIR / 'agent.pt'
ENGINE_PATH = SAVE_DIR / 'engine.pkl'

class Engine:
    def __init__(self):
        self.trad  = {c: Intersection(c, cfg) for c,cfg in CITIES.items()}
        self.ai    = {c: Intersection(c, cfg) for c,cfg in CITIES.items()}
        self.agent = DoubleDQNAgent()
        self.step_n = 0; self.ep_s = 0; self.ep_r = 0.0; self.ep_l = []
        self.running = False; self.lock = threading.Lock()
        self.t0 = time.time()
        self._last_save_ep = 0
        self._sim_start_time = datetime.now()
        # CSV log buffer — flushed periodically and on shutdown
        self._log_rows = []
        self._csv_file = None
        self._csv_writer = None
        self._init_csv_log()
        self._try_resume()
        # ── Enhancement state ──────────────────────────────
        # Per-city scenario map: { 'Chennai': {'type':'emergency','direction':'North'}, ... }
        self.city_scenarios = {}
        self.coord_group = set()      #  cities in coordinated group (legacy compat)
        self.history = deque(maxlen=10)
        # ── Multi-cluster coordination state ──────────────
        self._next_cluster_id = 1
        # clusters: { cluster_id: { id, cities:[], direction:'NS'|'EW', sync_delay:2.0, color_idx } }
        self.coord_clusters = {}
        # Timing offsets per city within cluster (visual only)
        # { city: { cluster_id, offset_ms, green_wave_phase } }
        self._coord_offsets = {}
        self._cluster_colors = ['#10b981','#06b6d4','#8b5cf6','#f59e0b','#f43f5e','#ec4899']

    def _init_csv_log(self):
        """Open the single simulation log CSV in append mode."""
        file_exists = SIM_LOG_PATH.exists() and SIM_LOG_PATH.stat().st_size > 0
        self._csv_file = open(SIM_LOG_PATH, 'a', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(SIM_LOG_COLUMNS)
            self._csv_file.flush()
            print(f'  [i] Created new simulation log: {SIM_LOG_PATH}')
        else:
            # Count existing rows to report
            with open(SIM_LOG_PATH, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            print(f'  [✓] Appending to simulation log: {SIM_LOG_PATH} ({row_count} existing rows)')

    # ── Resume from checkpoint ────────────────────────────
    def _try_resume(self):
        loaded_agent = self.agent.load(str(AGENT_PATH))
        loaded_engine = False
        if ENGINE_PATH.exists():
            try:
                with open(ENGINE_PATH, 'rb') as f:
                    state = pickle.load(f)
                self.step_n = state['step_n']
                for c in CITIES:
                    if c in state['intersections_ai']:
                        self._restore_intersection(self.ai[c], state['intersections_ai'][c])
                    if c in state['intersections_trad']:
                        self._restore_intersection(self.trad[c], state['intersections_trad'][c])
                loaded_engine = True
            except Exception as e:
                print(f'  [warn] Could not load engine state: {e}')
        if loaded_agent:
            self._last_save_ep = self.agent.episode
            status = f'Ep {self.agent.episode}, ε={self.agent.eps:.3f}, step={self.step_n}'
            print(f'  [✓] Resumed from checkpoint: {status}')
        else:
            print('  [i] Starting fresh (no checkpoint found)')

    @staticmethod
    def _snapshot_intersection(ix):
        return {
            'queues': dict(ix.queues), 'phase': ix.phase, 'timer': ix.timer,
            'arrived': ix.arrived, 'departed': ix.departed,
            'wait_acc': ix.wait_acc, 'steps': ix.steps,
            'vehs': {d: dict(v) for d, v in ix.vehs.items()},
        }

    @staticmethod
    def _restore_intersection(ix, data):
        ix.queues   = data['queues']
        ix.phase    = data['phase']
        ix.timer    = data['timer']
        ix.arrived  = data['arrived']
        ix.departed = data['departed']
        ix.wait_acc = data['wait_acc']
        ix.steps    = data['steps']
        ix.vehs     = data['vehs']

    # ── Multi-Cluster Coordination Methods ────────────────
    def create_cluster(self, cities, direction='NS', sync_delay=2.0):
        """Create a new coordination cluster from a list of cities."""
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        color_idx = (cid - 1) % len(self._cluster_colors)
        cluster = {
            'id': cid,
            'cities': list(cities),
            'direction': direction,  # 'NS' or 'EW'
            'sync_delay': float(sync_delay),
            'color_idx': color_idx,
            'color': self._cluster_colors[color_idx],
            'created_step': self.step_n,
        }
        self.coord_clusters[cid] = cluster
        self._recalc_offsets(cid)
        # Update legacy coord_group for backward compat
        self._sync_legacy_coord_group()
        return cluster

    def add_to_cluster(self, cluster_id, city):
        """Add a city to an existing cluster."""
        if cluster_id not in self.coord_clusters:
            return None
        cl = self.coord_clusters[cluster_id]
        if city not in cl['cities'] and city in CITIES:
            cl['cities'].append(city)
            self._recalc_offsets(cluster_id)
            self._sync_legacy_coord_group()
        return cl

    def remove_from_cluster(self, cluster_id, city):
        """Remove a city from a cluster. Deletes cluster if empty."""
        if cluster_id not in self.coord_clusters:
            return None
        cl = self.coord_clusters[cluster_id]
        if city in cl['cities']:
            cl['cities'].remove(city)
        if len(cl['cities']) == 0:
            del self.coord_clusters[cluster_id]
            self._sync_legacy_coord_group()
            return None
        self._recalc_offsets(cluster_id)
        self._sync_legacy_coord_group()
        return cl

    def delete_cluster(self, cluster_id):
        """Delete an entire cluster."""
        if cluster_id in self.coord_clusters:
            del self.coord_clusters[cluster_id]
            self._sync_legacy_coord_group()

    def clear_all_clusters(self):
        """Remove all coordination clusters."""
        self.coord_clusters.clear()
        self._coord_offsets.clear()
        self.coord_group.clear()

    def update_cluster_settings(self, cluster_id, direction=None, sync_delay=None):
        """Update direction or sync delay for a cluster."""
        if cluster_id not in self.coord_clusters:
            return None
        cl = self.coord_clusters[cluster_id]
        if direction is not None:
            cl['direction'] = direction
        if sync_delay is not None:
            cl['sync_delay'] = float(sync_delay)
        self._recalc_offsets(cluster_id)
        return cl

    def _recalc_offsets(self, cluster_id):
        """Recalculate green-wave phase offsets for a cluster based on
        geographic ordering along the chosen direction."""
        cl = self.coord_clusters.get(cluster_id)
        if not cl:
            return
        cities = cl['cities']
        direction = cl['direction']
        delay = cl['sync_delay']
        # Sort cities by geographic position along the flow axis
        if direction == 'NS':
            # North→South: sort by descending latitude
            ordered = sorted(cities, key=lambda c: -CITIES[c]['lat'])
        else:
            # East→West: sort by ascending longitude (west to east flow)
            ordered = sorted(cities, key=lambda c: CITIES[c]['lng'])
        cl['ordered_cities'] = ordered
        for i, city in enumerate(ordered):
            self._coord_offsets[city] = {
                'cluster_id': cluster_id,
                'offset_sec': i * delay,
                'order': i,
                'total': len(ordered),
            }

    def _sync_legacy_coord_group(self):
        """Keep the legacy coord_group set in sync with clusters."""
        self.coord_group = set()
        for cl in self.coord_clusters.values():
            self.coord_group.update(cl['cities'])

    def get_coordination_metrics(self):
        """Compute coordination performance metrics per cluster.
        These are visual/statistical metrics — they do NOT affect RL."""
        results = {}
        for cid, cl in self.coord_clusters.items():
            cities = cl['cities']
            if len(cities) < 2:
                continue
            direction = cl['direction']
            target_dirs = ['North', 'South'] if direction == 'NS' else ['East', 'West']
            # Average cluster wait time (AI controller)
            waits = []
            throughputs_ai = []
            throughputs_trad = []
            sync_scores = []
            for city in cities:
                a_ix = self.ai[city]
                t_ix = self.trad[city]
                s = max(a_ix.steps, 1)
                waits.append(a_ix.wait_acc / s)
                throughputs_ai.append(a_ix.departed / s * 60)
                throughputs_trad.append(t_ix.departed / max(t_ix.steps, 1) * 60)
                # Check if green phase is aligned with flow direction
                current_phase = DIRECTIONS[a_ix.phase]
                if current_phase in target_dirs:
                    sync_scores.append(1.0)
                else:
                    sync_scores.append(0.0)
            avg_wait = round(float(np.mean(waits)), 2) if waits else 0
            avg_tp_ai = float(np.mean(throughputs_ai)) if throughputs_ai else 0
            avg_tp_trad = float(np.mean(throughputs_trad)) if throughputs_trad else 0
            flow_efficiency = round(
                (avg_tp_ai - avg_tp_trad) / max(avg_tp_trad, 0.1) * 100, 1
            )
            sync_pct = round(float(np.mean(sync_scores)) * 100, 1) if sync_scores else 0
            results[cid] = {
                'cluster_id': cid,
                'node_count': len(cities),
                'direction': cl['direction'],
                'direction_label': 'North-South' if direction == 'NS' else 'East-West',
                'sync_delay': cl['sync_delay'],
                'color': cl['color'],
                'avg_wait': avg_wait,
                'flow_efficiency': flow_efficiency,
                'sync_pct': sync_pct,
                'ordered_cities': cl.get('ordered_cities', cities),
            }
        return results

    def get_green_wave_state(self):
        """Get the current green-wave visual timing state for all clustered cities.
        Returns timing offset data for the frontend to animate."""
        wave_state = {}
        now_sec = time.time()
        for cid, cl in self.coord_clusters.items():
            ordered = cl.get('ordered_cities', cl['cities'])
            delay = cl['sync_delay']
            direction = cl['direction']
            target_dirs = ['North', 'South'] if direction == 'NS' else ['East', 'West']
            for i, city in enumerate(ordered):
                offset = i * delay
                # Cycle position within a green wave (visual only)
                cycle_len = len(ordered) * delay + 5.0  # full cycle length
                phase_pos = ((now_sec - offset) % cycle_len) / cycle_len
                # Is this city in its "green wave window"?
                window = delay / cycle_len if cycle_len > 0 else 0.5
                in_wave = phase_pos < window * 2
                wave_state[city] = {
                    'cluster_id': cid,
                    'order': i,
                    'offset_sec': offset,
                    'phase_pos': round(phase_pos, 3),
                    'in_wave': in_wave,
                    'color': cl['color'],
                    'direction': direction,
                    'target_dirs': target_dirs,
                    'next_city': ordered[i + 1] if i + 1 < len(ordered) else None,
                    'prev_city': ordered[i - 1] if i > 0 else None,
                }
        return wave_state

    def _save_checkpoint(self):
        """Save DQN weights + full engine state to disk."""
        try:
            self.agent.save(str(AGENT_PATH))
            state = {
                'step_n': self.step_n,
                'intersections_ai':   {c: self._snapshot_intersection(self.ai[c])   for c in CITIES},
                'intersections_trad': {c: self._snapshot_intersection(self.trad[c]) for c in CITIES},
            }
            with open(ENGINE_PATH, 'wb') as f:
                pickle.dump(state, f)
            self._last_save_ep = self.agent.episode
        except Exception as e:
            print(f'  [warn] Checkpoint save failed: {e}')

    def _log_step(self, city, controller, ix, reward=0.0):
        """Append one row to the CSV log buffer."""
        ts = self._sim_start_time + timedelta(seconds=self.step_n * 0.5)
        m = ix.metrics()
        vs = ix.veh_summary()
        self._csv_writer.writerow([
            self.step_n, ts.strftime('%Y-%m-%d %H:%M:%S'),
            self.agent.episode, round(self.agent.eps, 4),
            city, controller,
            ix.queues['North'], ix.queues['South'],
            ix.queues['East'], ix.queues['West'],
            DIRECTIONS[ix.phase], ix.timer,
            ix.arrived, ix.departed,
            m['avg_wait'], m['throughput'], round(reward, 4),
            vs['bikes'], vs['cars'], vs['buses'], vs['trucks'],
        ])

    def _flush_csv(self):
        """Flush the CSV file to disk."""
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()

    def shutdown(self):
        """Graceful shutdown — save everything."""
        self.running = False
        with self.lock:
            self._save_checkpoint()
            self._flush_csv()
            if self._csv_file and not self._csv_file.closed:
                self._csv_file.close()
            print(f'  [✓] Saved on shutdown: Ep {self.agent.episode}, Step {self.step_n}')
            print(f'  [✓] Simulation log: {SIM_LOG_PATH}')

    def step(self):
        with self.lock:
            self.step_n += 1; self.ep_s += 1
            tod = tod_multiplier(self.step_n)
            sim_hour = (self.step_n % 96) / 96 * 24
            for city in CITIES:
                seed = (self.step_n*7 + hash(city)) % (2**31)
                t = self.trad[city]; a = self.ai[city]
                oi = t.intensity
                # ── Per-city scenario effects ──
                sc = self.city_scenarios.get(city)
                sc_mult = 1.0
                if sc:
                    st = sc.get('type', 'normal')
                    sd = sc.get('direction', 'North')
                    if st == 'festival':   sc_mult = 2.5
                    elif st == 'emergency': sc_mult = 1.5
                    elif st == 'accident': sc_mult = 1.3
                t.intensity = oi*tod*sc_mult; a.intensity = oi*tod*sc_mult
                t.arrivals(np.random.RandomState(seed), sim_hour)
                a.arrivals(np.random.RandomState(seed), sim_hour)
                t.intensity = oi; a.intensity = oi

                if sc:
                    st = sc.get('type', 'normal')
                    sd = sc.get('direction', 'North')
                    if st == 'emergency':
                        extra = random.randint(2, 5)
                        t.queues[sd] += extra; t.arrived += extra
                        a.queues[sd] += extra; a.arrived += extra

                t.act(1 if t.timer>=30 else 0)
                self._log_step(city, 'traditional', t, reward=0)

                sv = a.state_vec()
                ac = self.agent.act(sv)
                rw = a.act(ac)

                if sc:
                    st = sc.get('type', 'normal')
                    sd = sc.get('direction', 'North')
                    if st == 'accident':
                        block = random.randint(1, 3)
                        t.queues[sd] += block; a.queues[sd] += block

                self._log_step(city, 'ai_dqn', a, reward=rw)
                ns = a.state_vec()
                dn = float(self.ep_s >= EP_LEN)
                self.agent.mem.push(sv, ac, rw, ns, dn)
                self.ep_r += rw
                if len(self.agent.mem) >= self.agent.bs:
                    l = self.agent.learn()
                    if l > 0: self.ep_l.append(l)

            if self.ep_s >= EP_LEN:
                # ── Historical performance tracking ──
                _tw = [ix.wait_acc/max(ix.steps,1) for ix in self.trad.values()]
                _aw = [ix.wait_acc/max(ix.steps,1) for ix in self.ai.values()]
                _tq = sum(sum(ix.queues.values()) for ix in self.trad.values())
                _aq = sum(sum(ix.queues.values()) for ix in self.ai.values())
                _tt = [ix.departed/max(ix.steps,1)*60 for ix in self.trad.values()]
                _at = [ix.departed/max(ix.steps,1)*60 for ix in self.ai.values()]
                _tw_avg = round(float(np.mean(_tw)),2)
                _aw_avg = round(float(np.mean(_aw)),2)
                _imp = round((_tw_avg-_aw_avg)/max(_tw_avg,0.1)*100,1)
                self.history.append({
                    'episode': self.agent.episode+1,
                    'ai_avg_wait': _aw_avg, 'trad_avg_wait': _tw_avg,
                    'ai_queue': _aq, 'trad_queue': _tq,
                    'ai_throughput': round(float(np.mean(_at)),1),
                    'trad_throughput': round(float(np.mean(_tt)),1),
                    'improvement_pct': _imp,
                    'scenario': ','.join(sorted(set(s['type'] for s in self.city_scenarios.values()))) or 'normal',
                    'timestamp': datetime.now().isoformat(),
                })
                al = float(np.mean(self.ep_l)) if self.ep_l else 0
                self.agent.end_ep(self.ep_r, al)
                self.ep_s=0; self.ep_r=0; self.ep_l=[]
                # Auto-save checkpoint
                if self.agent.episode - self._last_save_ep >= SAVE_EVERY:
                    self._save_checkpoint()
                    self._flush_csv()

    def snapshot(self):
        with self.lock:
            cities = {}; tv_ai=0; tv_tr=0
            for c, cfg in CITIES.items():
                t=self.trad[c]; a=self.ai[c]
                tq_t=sum(t.queues.values()); tq_a=sum(a.queues.values())
                tv_tr+=tq_t; tv_ai+=tq_a
                cities[c] = {
                    'lat':cfg['lat'],'lng':cfg['lng'],'pop':cfg['pop'],'type':cfg['type'],
                    'traditional':{'queues':dict(t.queues),'signals':t.signals(),
                        'metrics':t.metrics(),'phase':DIRECTIONS[t.phase],'timer':t.timer,
                        'vehicles':t.veh_summary()},
                    'ai':{'queues':dict(a.queues),'signals':a.signals(),
                        'metrics':a.metrics(),'phase':DIRECTIONS[a.phase],'timer':a.timer,
                        'vehicles':a.veh_summary()},
                }

            def agg(side):
                ints = self.trad if side=='trad' else self.ai
                tq=sum(sum(i.queues.values()) for i in ints.values())
                ws=[i.wait_acc/max(i.steps,1) for i in ints.values()]
                th=[i.departed/max(i.steps,1)*60 for i in ints.values()]
                return {'total_queue':tq,'avg_wait':round(float(np.mean(ws)),2),
                        'throughput':round(float(np.mean(th)),1),
                        'total_arrived':sum(i.arrived for i in ints.values()),
                        'total_departed':sum(i.departed for i in ints.values())}
            tm=agg('trad'); am=agg('ai')
            imp={
                'wait':round((tm['avg_wait']-am['avg_wait'])/max(tm['avg_wait'],0.1)*100,1),
                'queue':round((tm['total_queue']-am['total_queue'])/max(tm['total_queue'],1)*100,1),
                'throughput':round((am['throughput']-tm['throughput'])/max(tm['throughput'],0.1)*100,1),
            }
            h=(self.step_n%96)/96*24; hh=int(h); mm=int((h-hh)*60)
            # ── Heatmap congestion data ──
            heatmap_data = []
            for c in CITIES:
                a_ix = self.ai[c]
                tq = sum(a_ix.queues.values())
                congestion = min(tq / 30.0, 1.0)
                heatmap_data.append({
                    'lat': CITIES[c]['lat'], 'lng': CITIES[c]['lng'],
                    'intensity': round(congestion, 3), 'queue': tq, 'city': c,
                })
            return {
                'cities':cities,'trad':tm,'ai':am,'improvement':imp,
                'training':{'episode':self.agent.episode,'epsilon':round(self.agent.eps,4),
                    'rewards':self.agent.rw_hist[-120:],'losses':self.agent.ls_hist[-120:],
                    'epsilons':self.agent.ep_hist[-120:],'avg_q':self.agent.q_hist[-120:],
                    'memory':len(self.agent.mem),'step':self.step_n,'ep_step':self.ep_s},
                'clock':{'hour':hh,'minute':mm,'display':f'{hh:02d}:{mm:02d}',
                    'elapsed':round(time.time()-self.t0,1)},
                'summary':{'vehicles_ai':tv_ai,'vehicles_trad':tv_tr,
                    'cities_count':len(CITIES),'arrived_ai':am['total_arrived'],
                    'departed_ai':am['total_departed']},
                'checkpoint':{
                    'dataset_loaded': DATASET_LOADED,
                    'last_save_ep': self._last_save_ep,
                    'save_dir': str(SAVE_DIR),
                },
                'scenario': {c: dict(s) for c, s in self.city_scenarios.items()},
                'coord_group': sorted(self.coord_group),
                'coord_clusters': {str(cid): {
                    'id': cl['id'],
                    'cities': cl['cities'],
                    'direction': cl['direction'],
                    'sync_delay': cl['sync_delay'],
                    'color': cl['color'],
                    'color_idx': cl['color_idx'],
                    'ordered_cities': cl.get('ordered_cities', cl['cities']),
                } for cid, cl in self.coord_clusters.items()},
                'coord_metrics': self.get_coordination_metrics(),
                'green_wave': self.get_green_wave_state(),
                'heatmap': heatmap_data,
                'history_count': len(self.history),
            }


engine = Engine()

# ── Graceful shutdown on Ctrl+C / process kill ─────────────
def _on_exit():
    engine.shutdown()

atexit.register(_on_exit)

def sim_loop():
    engine.running = True
    while engine.running:
        engine.step(); time.sleep(0.5)


# ═══════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def api_state():
    return jsonify(engine.snapshot())

@app.route('/api/stream')
def api_stream():
    def gen():
        while True:
            try: yield f"data: {json.dumps(engine.snapshot())}\n\n"
            except: yield "data: {}\n\n"
            time.sleep(1)
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/api/cities')
def api_cities():
    return jsonify(CITIES)

@app.route('/api/scenario', methods=['POST'])
def api_set_scenario():
    """Set or clear scenario for a specific city.
    Body: {city, type, direction}  — set scenario
    Body: {city, type:'normal'}    — clear scenario for that city
    Body: {clear_all: true}        — clear all city scenarios
    """
    data = request.get_json()
    with engine.lock:
        if data.get('clear_all'):
            engine.city_scenarios.clear()
            return jsonify({'status': 'ok', 'scenarios': {}})
        city = data.get('city')
        stype = data.get('type', 'normal')
        direction = data.get('direction', 'North')
        if not city or city not in CITIES:
            return jsonify({'status': 'error', 'msg': 'invalid city'}), 400
        if stype == 'normal':
            engine.city_scenarios.pop(city, None)
        else:
            engine.city_scenarios[city] = {'type': stype, 'direction': direction}
        return jsonify({'status': 'ok', 'scenarios': {c: dict(s) for c, s in engine.city_scenarios.items()}})

@app.route('/api/history')
def api_history():
    with engine.lock:
        records = list(engine.history)
    if not records:
        return jsonify({'records': [], 'summary': {}})
    improvements = [r['improvement_pct'] for r in records]
    return jsonify({
        'records': records,
        'summary': {
            'avg_improvement': round(float(np.mean(improvements)), 1),
            'best': round(max(improvements), 1),
            'worst': round(min(improvements), 1),
            'mean': round(float(np.mean(improvements)), 1),
            'std': round(float(np.std(improvements)), 1),
            'count': len(records),
        }
    })

@app.route('/api/coordinated', methods=['POST'])
def api_set_coordinated():
    """Toggle a city in the coordination group (legacy compat).
    Body: {city}          — toggle single city in group
    Body: {clear: true}   — clear entire coordination group
    """
    data = request.get_json()
    with engine.lock:
        if data.get('clear'):
            engine.clear_all_clusters()
            return jsonify({'status': 'ok', 'group': [], 'clusters': {}})
        city = data.get('city')
        if city and city in CITIES:
            if city in engine.coord_group:
                engine.coord_group.discard(city)
            else:
                engine.coord_group.add(city)
    return jsonify({'status': 'ok', 'group': sorted(engine.coord_group)})


@app.route('/api/coordination/cluster', methods=['POST'])
def api_coordination_cluster():
    """Create, update, or manage coordination clusters.
    Body: {action:'create', cities:[], direction:'NS', sync_delay:2.0}
    Body: {action:'add', cluster_id:1, city:'Chennai'}
    Body: {action:'remove', cluster_id:1, city:'Chennai'}
    Body: {action:'delete', cluster_id:1}
    Body: {action:'update', cluster_id:1, direction:'EW', sync_delay:3.0}
    Body: {action:'clear_all'}
    """
    data = request.get_json()
    action = data.get('action', '')
    with engine.lock:
        if action == 'create':
            cities = data.get('cities', [])
            cities = [c for c in cities if c in CITIES]
            if len(cities) < 2:
                return jsonify({'status': 'error', 'msg': 'Need at least 2 cities'}), 400
            direction = data.get('direction', 'NS')
            sync_delay = data.get('sync_delay', 2.0)
            cl = engine.create_cluster(cities, direction, sync_delay)
            return jsonify({'status': 'ok', 'cluster': cl})

        elif action == 'add':
            cid = data.get('cluster_id')
            city = data.get('city')
            if not cid or not city:
                return jsonify({'status': 'error', 'msg': 'cluster_id and city required'}), 400
            cl = engine.add_to_cluster(int(cid), city)
            if cl:
                return jsonify({'status': 'ok', 'cluster': cl})
            return jsonify({'status': 'error', 'msg': 'cluster not found'}), 404

        elif action == 'remove':
            cid = data.get('cluster_id')
            city = data.get('city')
            if not cid or not city:
                return jsonify({'status': 'error', 'msg': 'cluster_id and city required'}), 400
            cl = engine.remove_from_cluster(int(cid), city)
            return jsonify({'status': 'ok', 'cluster': cl})

        elif action == 'delete':
            cid = data.get('cluster_id')
            if not cid:
                return jsonify({'status': 'error', 'msg': 'cluster_id required'}), 400
            engine.delete_cluster(int(cid))
            return jsonify({'status': 'ok'})

        elif action == 'update':
            cid = data.get('cluster_id')
            if not cid:
                return jsonify({'status': 'error', 'msg': 'cluster_id required'}), 400
            cl = engine.update_cluster_settings(
                int(cid),
                direction=data.get('direction'),
                sync_delay=data.get('sync_delay'),
            )
            if cl:
                return jsonify({'status': 'ok', 'cluster': {
                    'id': cl['id'], 'cities': cl['cities'],
                    'direction': cl['direction'], 'sync_delay': cl['sync_delay'],
                    'color': cl['color'],
                    'ordered_cities': cl.get('ordered_cities', cl['cities']),
                }})
            return jsonify({'status': 'error', 'msg': 'cluster not found'}), 404

        elif action == 'clear_all':
            engine.clear_all_clusters()
            return jsonify({'status': 'ok'})

    return jsonify({'status': 'error', 'msg': 'unknown action'}), 400


@app.route('/api/coordination/metrics')
def api_coordination_metrics():
    """Get coordination performance metrics for all clusters."""
    with engine.lock:
        metrics = engine.get_coordination_metrics()
        wave = engine.get_green_wave_state()
    return jsonify({'metrics': metrics, 'green_wave': wave})


if __name__ == '__main__':
    threading.Thread(target=sim_loop, daemon=True).start()
    print('\n  ╔═══════════════════════════════════════════════╗')
    print('  ║  Traffic Signal Control  ·  Double DQN        ║')
    print('  ║  Dashboard → http://127.0.0.1:5000            ║')
    print('  ╚═══════════════════════════════════════════════╝\n')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
