"""
Traffic Simulation API
Handles vehicle management, controller simulation, and metrics calculation
"""

from src.traffic_environment import (
    TrafficEnvironment,
    TraditionalController,
    AITrafficController,
    PerformanceAnalyzer
)
from typing import Dict, List, Tuple


class TrafficSimulationEngine:
    """
    Main simulation engine coordinating both controllers
    and tracking real-time metrics
    """
    
    def __init__(self):
        self.env_traditional = TrafficEnvironment()
        self.env_ai = TrafficEnvironment()
        
        self.controller_traditional = TraditionalController(cycle_time=30.0)
        self.controller_ai = AITrafficController()
        
        self.analyzer = PerformanceAnalyzer()
        
        self.simulation_time = 0
        self.vehicles = []
        self.last_vehicle_id = 0
    
    def add_vehicle(self, direction: str, vehicle_type: str, intersection: str) -> Dict:
        """
        Add a vehicle to both environments
        """
        self.last_vehicle_id += 1
        
        vehicle = {
            'id': self.last_vehicle_id,
            'direction': direction.lower(),
            'type': vehicle_type,
            'intersection': intersection,
            'timestamp': self.simulation_time
        }
        
        self.vehicles.append(vehicle)
        
        # Add to both environments (they operate independently)
        self.env_traditional.add_vehicles(direction, 1)
        self.env_ai.add_vehicles(direction, 1)
        
        return vehicle
    
    def remove_vehicle(self, vehicle_id: int) -> bool:
        """Remove a vehicle from the simulation"""
        vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
        if vehicle:
            self.vehicles.remove(vehicle)
            return True
        return False
    
    def step_simulation(self, time_step: float = 1.0) -> Dict:
        """
        Step the simulation forward by time_step seconds
        Returns current state and metrics
        """
        self.simulation_time += time_step
        
        # Traditional controller (fixed timing)
        trad_green = self.controller_traditional.get_green_direction(self.simulation_time)
        trad_signals = self.controller_traditional.get_signal_state(self.simulation_time)
        self.env_traditional.process_signal(trad_green, time_step)
        
        # AI controller (adaptive)
        state = self.env_ai.get_state()
        ai_green, ai_duration = self.controller_ai.decide_action(state)
        ai_signals = self.controller_ai.get_signal_state(ai_green)
        self.env_ai.process_signal(ai_green, min(time_step, ai_duration))
        
        return {
            'simulation_time': self.simulation_time,
            'traditional': {
                'signals': trad_signals,
                'metrics': self.env_traditional.get_metrics()
            },
            'ai': {
                'signals': ai_signals,
                'metrics': self.env_ai.get_metrics()
            },
            'vehicles': {
                'count': len(self.vehicles),
                'by_direction': {
                    'north': sum(1 for v in self.vehicles if v['direction'] == 'north'),
                    'east': sum(1 for v in self.vehicles if v['direction'] == 'east'),
                    'south': sum(1 for v in self.vehicles if v['direction'] == 'south'),
                    'west': sum(1 for v in self.vehicles if v['direction'] == 'west')
                }
            },
            'comparison': self.analyzer.compare_controllers(
                self.env_traditional.get_metrics(),
                self.env_ai.get_metrics()
            )
        }
    
    def get_current_state(self) -> Dict:
        """Get current state without stepping simulation"""
        return {
            'simulation_time': self.simulation_time,
            'traditional': {
                'signals': self.controller_traditional.get_signal_state(self.simulation_time),
                'metrics': self.env_traditional.get_metrics()
            },
            'ai': {
                'signals': self.controller_ai.get_signal_state(
                    self.controller_ai.get_signal_state(
                        self.env_ai.get_state()
                    ) if True else 'north'
                ),
                'metrics': self.env_ai.get_metrics()
            },
            'vehicles': {
                'count': len(self.vehicles),
                'by_direction': {
                    'north': sum(1 for v in self.vehicles if v['direction'] == 'north'),
                    'east': sum(1 for v in self.vehicles if v['direction'] == 'east'),
                    'south': sum(1 for v in self.vehicles if v['direction'] == 'south'),
                    'west': sum(1 for v in self.vehicles if v['direction'] == 'west')
                }
            },
            'comparison': self.analyzer.compare_controllers(
                self.env_traditional.get_metrics(),
                self.env_ai.get_metrics()
            )
        }
    
    def reset(self) -> None:
        """Reset entire simulation"""
        self.env_traditional.reset()
        self.env_ai.reset()
        self.simulation_time = 0
        self.vehicles = []
        self.last_vehicle_id = 0
    
    def get_comparison(self) -> Dict:
        """Get detailed comparison between controllers"""
        return self.analyzer.compare_controllers(
            self.env_traditional.get_metrics(),
            self.env_ai.get_metrics()
        )
