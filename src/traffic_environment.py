"""
Professional Traffic Environment System
Handles vehicle simulation, state management, and controller logic
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import math

class TrafficEnvironment:
    """
    Simulates traffic environment with vehicles in 4 directions (N, E, S, W)
    Tracks queue lengths, waiting times, throughput, and delays
    """
    
    def __init__(self):
        """Initialize traffic environment with 4 directions"""
        self.directions = ['north', 'east', 'south', 'west']
        self.queue_length = {d: 0 for d in self.directions}
        self.waiting_time = {d: 0.0 for d in self.directions}
        self.total_vehicles_served = {d: 0 for d in self.directions}
        self.timestamp = datetime.now()
        self.simulation_time = 0  # in seconds
        
        # Performance tracking
        self.all_wait_times = {d: [] for d in self.directions}
        self.throughput_history = {d: [] for d in self.directions}
        
    def add_vehicles(self, direction: str, count: int) -> None:
        """Add vehicles to a specific direction queue"""
        if direction.lower() not in self.directions:
            raise ValueError(f"Invalid direction: {direction}")
        
        self.queue_length[direction.lower()] += count
    
    def get_state(self) -> Dict[str, int]:
        """Get current environment state (queue lengths for all directions)"""
        return {
            'north': self.queue_length['north'],
            'east': self.queue_length['east'],
            'south': self.queue_length['south'],
            'west': self.queue_length['west'],
            'total_vehicles': sum(self.queue_length.values()),
            'simulation_time': self.simulation_time
        }
    
    def process_signal(self, green_direction: str, time_duration: float) -> Dict:
        """
        Process signal change and update environment
        
        Args:
            green_direction: Direction with green signal
            time_duration: Duration green light is on (in seconds)
            
        Returns:
            Dictionary with metrics for this cycle
        """
        green_dir = green_direction.lower()
        
        # Throughput: vehicles that can pass in this time
        # Assume 1 vehicle per 2 seconds average
        vehicles_served = max(0, int(time_duration / 2))
        
        # Process served vehicles
        for direction in self.directions:
            if direction == green_dir:
                # Vehicles exit
                vehicles_passed = min(vehicles_served, self.queue_length[direction])
                self.queue_length[direction] -= vehicles_passed
                self.total_vehicles_served[direction] += vehicles_passed
                self.throughput_history[direction].append(vehicles_passed)
            else:
                # Vehicles waiting
                wait_increase = time_duration
                vehicles_waiting = self.queue_length[direction]
                self.waiting_time[direction] += wait_increase * vehicles_waiting
                
                if vehicles_waiting > 0:
                    self.all_wait_times[direction].extend([wait_increase] * vehicles_waiting)
        
        # Green direction vehicles wait too if queue not served completely
        remaining = self.queue_length[green_dir]
        if remaining > 0:
            unserved_wait = time_duration * remaining
            self.waiting_time[green_dir] += unserved_wait
            self.all_wait_times[green_dir].extend([time_duration] * remaining)
        
        self.simulation_time += time_duration
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict:
        """Calculate and return comprehensive metrics"""
        total_vehicles = sum(self.total_vehicles_served.values())
        
        if total_vehicles == 0:
            return {
                'avg_wait_time': 0,
                'total_queue_length': sum(self.queue_length.values()),
                'throughput': 0,
                'total_delay': 0
            }
        
        # Average wait time across all served vehicles
        all_waits = []
        for waits in self.all_wait_times.values():
            all_waits.extend(waits)
        avg_wait = np.mean(all_waits) if all_waits else 0
        
        # Total delay = sum of all waiting times
        total_delay = sum(self.waiting_time.values())
        
        # Throughput = vehicles per minute
        throughput = (total_vehicles / max(1, self.simulation_time)) * 60
        
        # Current queue length
        current_queue = sum(self.queue_length.values())
        
        return {
            'avg_wait_time': float(avg_wait),
            'total_queue_length': current_queue,
            'throughput': float(throughput),
            'total_delay': float(total_delay),
            'vehicles_served': total_vehicles,
            'simulation_time': self.simulation_time
        }
    
    def reset(self) -> None:
        """Reset environment for new simulation"""
        self.queue_length = {d: 0 for d in self.directions}
        self.waiting_time = {d: 0.0 for d in self.directions}
        self.total_vehicles_served = {d: 0 for d in self.directions}
        self.all_wait_times = {d: [] for d in self.directions}
        self.throughput_history = {d: [] for d in self.directions}
        self.simulation_time = 0
        self.timestamp = datetime.now()


class TraditionalController:
    """
    Fixed-timer traffic control
    Uses 30-second cycle: 7.5s per direction (North -> East -> South -> West)
    """
    
    def __init__(self, cycle_time: float = 30.0):
        self.cycle_time = cycle_time
        self.phase_time = cycle_time / 4  # 7.5 seconds per phase
        self.current_time = 0
        self.directions = ['north', 'east', 'south', 'west']
    
    def get_signal_state(self, elapsed_time: float) -> Dict[str, str]:
        """Get signal state for all directions at given elapsed time"""
        phase = int((elapsed_time % self.cycle_time) / self.phase_time)
        
        signals = {}
        for i, direction in enumerate(self.directions):
            signals[direction] = 'green' if i == phase else 'red'
        
        return signals
    
    def get_green_direction(self, elapsed_time: float) -> str:
        """Get which direction has green light"""
        phase = int((elapsed_time % self.cycle_time) / self.phase_time)
        return self.directions[phase]
    
    def get_next_phase_time(self, elapsed_time: float) -> float:
        """Get time until next phase change"""
        phase_position = (elapsed_time % self.cycle_time) / self.phase_time
        next_phase = (int(phase_position) + 1) * self.phase_time
        time_to_next = next_phase - (elapsed_time % self.cycle_time)
        return max(0.1, time_to_next)


class AITrafficController:
    """
    Adaptive AI traffic control using learned policy
    Prioritizes directions with higher vehicle counts
    Extends green light for congested directions
    """
    
    def __init__(self):
        self.current_green_direction = 'north'
        self.direction_time = {}
        self.directions = ['north', 'east', 'south', 'west']
        self.min_green_time = 5.0
        self.max_green_time = 15.0
    
    def decide_action(self, state: Dict[str, int]) -> Tuple[str, float]:
        """
        Decide which direction should be green and for how long
        
        Args:
            state: Dictionary with queue lengths for each direction
            
        Returns:
            Tuple of (green_direction, duration)
        """
        # Extract queue lengths
        queue_lengths = {
            'north': state.get('north', 0),
            'east': state.get('east', 0),
            'south': state.get('south', 0),
            'west': state.get('west', 0)
        }
        
        # Find direction with most vehicles
        max_direction = max(queue_lengths, key=queue_lengths.get)
        max_queue_length = queue_lengths[max_direction]
        
        # Adaptive timing based on congestion
        # More vehicles = longer green time (up to max_green_time)
        base_duration = self.min_green_time
        if max_queue_length > 0:
            # Scale duration based on queue length
            # Formula: min_green + (queue_length / 100) * (max_green - min_green)
            duration = base_duration + min(
                (max_queue_length / 50) * (self.max_green_time - base_duration),
                self.max_green_time - base_duration
            )
        else:
            duration = base_duration
        
        return max_direction, duration
    
    def get_signal_state(self, green_direction: str) -> Dict[str, str]:
        """Get signal state with specified direction as green"""
        signals = {}
        for direction in self.directions:
            signals[direction] = 'green' if direction == green_direction else 'red'
        return signals


class PerformanceAnalyzer:
    """Analyzes and compares performance between controllers"""
    
    @staticmethod
    def compare_controllers(
        traditional_metrics: Dict,
        ai_metrics: Dict
    ) -> Dict:
        """
        Compare metrics between traditional and AI controllers
        """
        improvements = {}
        
        # Wait time improvement (%)
        trad_wait = traditional_metrics.get('avg_wait_time', 1)
        ai_wait = ai_metrics.get('avg_wait_time', 1)
        if trad_wait > 0:
            improvements['wait_time'] = ((trad_wait - ai_wait) / trad_wait) * 100
        else:
            improvements['wait_time'] = 0
        
        # Queue length improvement (%)
        trad_queue = traditional_metrics.get('total_queue_length', 1)
        ai_queue = ai_metrics.get('total_queue_length', 1)
        if trad_queue > 0:
            improvements['queue_length'] = ((trad_queue - ai_queue) / trad_queue) * 100
        else:
            improvements['queue_length'] = 0
        
        # Throughput improvement (%)
        trad_throughput = traditional_metrics.get('throughput', 1)
        ai_throughput = ai_metrics.get('throughput', 1)
        if trad_throughput > 0:
            improvements['throughput'] = ((ai_throughput - trad_throughput) / trad_throughput) * 100
        else:
            improvements['throughput'] = 0
        
        # Delay improvement (%)
        trad_delay = traditional_metrics.get('total_delay', 1)
        ai_delay = ai_metrics.get('total_delay', 1)
        if trad_delay > 0:
            improvements['delay'] = ((trad_delay - ai_delay) / trad_delay) * 100
        else:
            improvements['delay'] = 0
        
        return {
            'wait_time_reduction': float(improvements['wait_time']),
            'queue_reduction': float(improvements['queue_length']),
            'throughput_increase': float(improvements['throughput']),
            'delay_reduction': float(improvements['delay']),
            'overall_score': float(np.mean([
                improvements['wait_time'],
                improvements['queue_length'],
                improvements['throughput'],
                improvements['delay']
            ]))
        }
