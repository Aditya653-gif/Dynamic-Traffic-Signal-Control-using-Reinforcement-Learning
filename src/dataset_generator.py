"""
Synthetic Traffic Dataset Generator for Tamil Nadu Urban Intersections
Generates realistic traffic data for AI-Based Dynamic Traffic Signal Control
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TrafficDatasetGenerator:
    """Generate synthetic traffic dataset for Tamil Nadu intersections"""
    
    def __init__(self, random_seed=42):
        """Initialize generator with traffic parameters"""
        np.random.seed(random_seed)
        
        # City traffic intensity multipliers (relative to base)
        self.city_intensity = {
            'Chennai': 1.0,          # Base intensity
            'Coimbatore': 0.75,
            'Madurai': 0.60,
            'Trichy': 0.55,
            'Salem': 0.50,
            'Tambaram': 0.65,
            'Chengalpattu': 0.58
        }
        
        # Vehicle type distribution (%)
        self.vehicle_distribution = {
            'Bikes': 0.55,
            'Cars': 0.30,
            'Buses': 0.10,
            'Trucks': 0.05
        }
        
        # Time period configurations (traffic multiplier, phase duration)
        self.time_periods = {
            'Morning Peak (7-9 AM)': {'multiplier': 0.95, 'phase_duration': 50},
            'Afternoon (9-12 PM)': {'multiplier': 0.60, 'phase_duration': 40},
            'Lunch Peak (12-2 PM)': {'multiplier': 0.80, 'phase_duration': 50},
            'Afternoon (2-5 PM)': {'multiplier': 0.65, 'phase_duration': 40},
            'Evening Peak (5-8 PM)': {'multiplier': 0.90, 'phase_duration': 50},
            'Night (8 PM-7 AM)': {'multiplier': 0.25, 'phase_duration': 30}
        }
        
        # Reward function parameters
        self.alpha = 0.6   # Wait time penalty
        self.beta = 0.3    # Queue length penalty
        self.gamma = 0.1   # Phase switch penalty
        
        # Signal phases: N-S, E-W
        self.signal_phases = ['North-South', 'East-West']
        
    def get_time_period(self, hour):
        """Determine traffic period based on hour"""
        if 7 <= hour < 9:
            return 'Morning Peak (7-9 AM)'
        elif 9 <= hour < 12:
            return 'Afternoon (9-12 PM)'
        elif 12 <= hour < 14:
            return 'Lunch Peak (12-2 PM)'
        elif 14 <= hour < 17:
            return 'Afternoon (2-5 PM)'
        elif 17 <= hour < 20:
            return 'Evening Peak (5-8 PM)'
        else:
            return 'Night (8 PM-7 AM)'
    
    def generate_vehicle_counts(self, city, hour):
        """Generate vehicle counts for all directions"""
        period = self.get_time_period(hour)
        base_multiplier = self.time_periods[period]['multiplier']
        city_multiplier = self.city_intensity[city]
        
        # Base vehicle volume per direction (peak hour baseline)
        base_volume = 100
        
        # Apply multipliers and add randomness
        direction_multipliers = {
            'north': np.random.uniform(0.9, 1.1),
            'south': np.random.uniform(0.9, 1.1),
            'east': np.random.uniform(0.9, 1.1),
            'west': np.random.uniform(0.9, 1.1)
        }
        
        total_multiplier = base_multiplier * city_multiplier
        
        vehicles = {
            'north': int(base_volume * total_multiplier * direction_multipliers['north']),
            'south': int(base_volume * total_multiplier * direction_multipliers['south']),
            'east': int(base_volume * total_multiplier * direction_multipliers['east']),
            'west': int(base_volume * total_multiplier * direction_multipliers['west'])
        }
        
        # Ensure minimum 1 vehicle for data validity
        for direction in vehicles:
            vehicles[direction] = max(1, vehicles[direction])
        
        return vehicles
    
    def generate_vehicle_type_distribution(self):
        """Generate vehicle type distribution string"""
        dist_str = "; ".join([f"{vtype}: {int(pct*100)}%" 
                             for vtype, pct in self.vehicle_distribution.items()])
        return dist_str
    
    def calculate_queue_length(self, vehicle_count, phase_duration):
        """Estimate queue length based on vehicles and signal phase"""
        # Assume ~0.7 vehicles per second can pass through intersection
        vehicles_per_second = 0.7
        max_pass_capacity = vehicles_per_second * phase_duration
        
        # Queue saturated if vehicles > capacity
        if vehicle_count > max_pass_capacity:
            queue = vehicle_count - max_pass_capacity
        else:
            queue = np.random.uniform(0, vehicle_count * 0.3)  # 0-30% queue in normal conditions
        
        return int(max(1, queue))
    
    def calculate_avg_wait_time(self, vehicle_count, signal_phase_index, total_signal_cycle=90):
        """Calculate average wait time at intersection"""
        # Red light duration for this direction
        red_light_duration = total_signal_cycle / 2
        
        # Average wait time = red light / 2 + queue delay
        base_wait = red_light_duration / 2
        
        # Add queue-based delay (longer queues = more wait)
        queue_delay = min(vehicle_count * 0.5, 30)  # Cap at 30 seconds
        
        avg_wait = base_wait + queue_delay + np.random.uniform(-5, 5)
        
        return max(1, avg_wait)
    
    def calculate_reward(self, total_wait_time, total_queue_length, phase_switches):
        """Calculate reward based on improved reward function"""
        # Penalty-based reward function
        reward = -(self.alpha * total_wait_time + 
                  self.beta * total_queue_length + 
                  self.gamma * phase_switches)
        
        return round(reward, 2)
    
    def generate_dataset(self, num_days=30, num_samples_per_intersection_per_day=16):
        """
        Generate complete synthetic traffic dataset
        16 samples per intersection per day (roughly every 1.5 hours)
        """
        records = []
        intersection_id = 1
        
        for city in self.city_intensity.keys():
            for day in range(num_days):
                base_timestamp = datetime(2024, 1, 1) + timedelta(days=day)
                
                for sample_idx in range(num_samples_per_intersection_per_day):
                    # Spread samples across 24 hours
                    hour = (sample_idx * 24 // num_samples_per_intersection_per_day)
                    minute = (sample_idx * 1440 // num_samples_per_intersection_per_day) % 60
                    timestamp = base_timestamp.replace(hour=hour, minute=minute)
                    
                    # Generate traffic data
                    vehicles = self.generate_vehicle_counts(city, hour)
                    period = self.get_time_period(hour)
                    phase_duration = self.time_periods[period]['phase_duration']
                    
                    # Calculate metrics for all directions combined
                    total_vehicles = sum(vehicles.values())
                    total_queue = sum([self.calculate_queue_length(vehicles[d], phase_duration) 
                                      for d in ['north', 'south', 'east', 'west']])
                    total_wait = sum([self.calculate_avg_wait_time(vehicles[d], i, 90) 
                                     for i, d in enumerate(['north', 'south', 'east', 'west'])])
                    
                    # Random phase switches (0-3 per cycle)
                    phase_switches = np.random.randint(0, 4)
                    
                    # Calculate reward
                    reward = self.calculate_reward(total_wait, total_queue, phase_switches)
                    
                    # Determine signal phase
                    signal_phase = np.random.choice(self.signal_phases)
                    
                    record = {
                        'intersection_id': intersection_id,
                        'city': city,
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'north_vehicle_count': vehicles['north'],
                        'south_vehicle_count': vehicles['south'],
                        'east_vehicle_count': vehicles['east'],
                        'west_vehicle_count': vehicles['west'],
                        'vehicle_type_distribution': self.generate_vehicle_type_distribution(),
                        'avg_wait_time': round(total_wait, 2),
                        'queue_length': total_queue,
                        'signal_phase': signal_phase,
                        'reward': reward
                    }
                    
                    records.append(record)
                
                intersection_id += 1
        
        return pd.DataFrame(records)
    
    def save_dataset(self, df, output_path=None):
        """Save dataset to CSV file"""
        # Use default path if not specified
        if output_path is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_path = os.path.join(project_dir, 'data', 'traffic_dataset.csv')
        
        # Create path if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Total records: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Cities: {df['city'].nunique()}")
        print(f"  Intersections: {df['intersection_id'].nunique()}")
        
        return output_path
    
    def get_dataset_summary(self, df):
        """Print dataset summary statistics"""
        print("\n" + "="*70)
        print("TRAFFIC DATASET SUMMARY")
        print("="*70)
        
        print("\n📍 City-wise Distribution:")
        city_counts = df['city'].value_counts()
        for city, count in city_counts.items():
            intensity = self.city_intensity[city]
            print(f"  {city:20} - {count:5} records (Intensity: {intensity:.1%})")
        
        print("\n⏰ Time Period Distribution:")
        df['time_period'] = df.apply(
            lambda row: self.get_time_period(
                int(row['timestamp'].split()[1].split(':')[0])
            ), axis=1
        )
        period_counts = df['time_period'].value_counts()
        for period, count in period_counts.items():
            print(f"  {period:30} - {count:5} records")
        
        print("\n🚗 Traffic Metrics (Aggregate):")
        print(f"  Average vehicles per direction: {df[['north_vehicle_count', 'south_vehicle_count', 'east_vehicle_count', 'west_vehicle_count']].mean().mean():.2f}")
        print(f"  Max queue length: {df['queue_length'].max()}")
        print(f"  Avg queue length: {df['queue_length'].mean():.2f}")
        print(f"  Max wait time: {df['avg_wait_time'].max():.2f}s")
        print(f"  Avg wait time: {df['avg_wait_time'].mean():.2f}s")
        
        print("\n💰 Reward Statistics:")
        print(f"  Min reward: {df['reward'].min():.2f}")
        print(f"  Max reward: {df['reward'].max():.2f}")
        print(f"  Avg reward: {df['reward'].mean():.2f}")
        print(f"  Std deviation: {df['reward'].std():.2f}")
        
        print("\n🚦 Signal Phase Distribution:")
        phase_counts = df['signal_phase'].value_counts()
        for phase, count in phase_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {phase:20} - {count:5} records ({pct:.1f}%)")
        
        print("\n🚲 Vehicle Type Distribution (Fixed):")
        for vtype, pct in self.vehicle_distribution.items():
            print(f"  {vtype:15} - {pct*100:5.0f}%")
        
        print("\n" + "="*70)


def main():
    """Main execution function"""
    print("\n🎯 AI-Based Dynamic Traffic Signal Control - Phase 1")
    print("📊 Synthetic Traffic Dataset Generation\n")
    
    # Initialize generator
    generator = TrafficDatasetGenerator(random_seed=42)
    
    # Generate dataset
    print("⏳ Generating synthetic traffic dataset...")
    print("   Cities: 7 (Chennai, Coimbatore, Madurai, Trichy, Salem, Tambaram, Chengalpattu)")
    print("   Days: 30")
    print("   Samples per intersection per day: 16")
    print("   Total intersections: 7 × 30 = 210\n")
    
    df = generator.generate_dataset(num_days=30, num_samples_per_intersection_per_day=16)
    
    # Save dataset
    output_file = generator.save_dataset(df)
    
    # Print summary
    generator.get_dataset_summary(df)
    
    # Display sample records
    print("\n📋 Sample Records (First 5):")
    print(df.head().to_string(index=False))
    
    print("\n✅ Phase 1 Complete! Dataset ready for RL training.")
    
    return df


if __name__ == "__main__":
    df = main()
