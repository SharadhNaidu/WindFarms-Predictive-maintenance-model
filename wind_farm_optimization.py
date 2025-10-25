"""
Wind Farm Energy Efficiency Optimization Module
Improves overall energy efficiency and output through advanced analytics and optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class WindFarmOptimizer:
    """
    Optimize wind farm performance through wake effect mitigation, 
    turbine coordination, and power output maximization.
    """
    
    def __init__(self, turbine_positions=None):
        """
        Initialize wind farm optimizer
        
        Parameters:
        - turbine_positions: List of (x, y) coordinates for each turbine
        """
        self.turbine_positions = turbine_positions
        self.wake_model_trained = False
        
    def calculate_wake_loss(self, wind_direction, wind_speed, turbine_idx):
        """
        Calculate power loss due to wake effects from upstream turbines
        
        Parameters:
        - wind_direction: Wind direction in degrees
        - wind_speed: Wind speed in m/s
        - turbine_idx: Index of turbine to calculate wake loss for
        
        Returns:
        - wake_loss: Percentage of power loss (0-100)
        """
        if self.turbine_positions is None:
            return 0.0
        
        wake_loss = 0.0
        current_pos = self.turbine_positions[turbine_idx]
        
        # Simplified Jensen wake model
        for i, pos in enumerate(self.turbine_positions):
            if i == turbine_idx:
                continue
                
            # Calculate if current turbine is in wake of turbine i
            dx = current_pos[0] - pos[0]
            dy = current_pos[1] - pos[1]
            
            # Direction vector
            wind_rad = np.radians(wind_direction)
            wind_dx = np.cos(wind_rad)
            wind_dy = np.sin(wind_rad)
            
            # Distance in wind direction
            distance = dx * wind_dx + dy * wind_dy
            
            if distance > 0:  # Downstream
                # Lateral distance
                lateral = abs(dx * wind_dy - dy * wind_dx)
                
                # Rotor diameter (assumed 80m)
                D = 80
                
                # Wake expansion
                k = 0.075  # Wake decay constant
                wake_radius = D/2 + k * distance
                
                if lateral < wake_radius:
                    # Velocity deficit
                    CT = 0.8  # Thrust coefficient
                    deficit = (1 - np.sqrt(1 - CT)) / (1 + 2*k*distance/D)**2
                    wake_loss += deficit * 100 * (1 - lateral/wake_radius)
        
        return min(wake_loss, 50)  # Cap at 50% loss
    
    def optimize_yaw_angles(self, wind_direction, wind_speed, n_turbines=3):
        """
        Optimize turbine yaw angles to maximize farm power output
        considering wake effects
        
        Parameters:
        - wind_direction: Current wind direction in degrees
        - wind_speed: Current wind speed in m/s
        - n_turbines: Number of turbines in farm
        
        Returns:
        - optimal_yaw_offsets: Array of optimal yaw angle offsets
        - power_gain: Expected power gain percentage
        """
        
        def farm_power_objective(yaw_offsets):
            """Calculate negative total farm power (for minimization)"""
            total_power = 0
            
            for i in range(n_turbines):
                # Base power (simplified power curve)
                if wind_speed < 3:
                    base_power = 0
                elif wind_speed < 12:
                    base_power = 500 * (wind_speed - 3) ** 2
                else:
                    base_power = min(3000, 500 + (wind_speed - 12) * 200)
                
                # Yaw loss
                yaw_loss = 1 - (np.cos(np.radians(yaw_offsets[i])) ** 3)
                
                # Wake loss (simplified)
                wake_loss = self.calculate_wake_loss(
                    wind_direction + yaw_offsets[i], 
                    wind_speed, 
                    i
                ) / 100
                
                turbine_power = base_power * (1 - yaw_loss) * (1 - wake_loss)
                total_power += turbine_power
            
            return -total_power  # Negative for minimization
        
        # Initial guess: no yaw offset
        x0 = np.zeros(n_turbines)
        
        # Bounds: yaw offset between -25 and +25 degrees
        bounds = [(-25, 25) for _ in range(n_turbines)]
        
        # Optimize
        result = minimize(
            farm_power_objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        optimal_yaw_offsets = result.x
        
        # Calculate power gain
        baseline_power = -farm_power_objective(x0)
        optimized_power = -result.fun
        power_gain = ((optimized_power - baseline_power) / baseline_power) * 100
        
        return optimal_yaw_offsets, power_gain
    
    def calculate_farm_efficiency(self, turbines_data):
        """
        Calculate overall wind farm efficiency
        
        Parameters:
        - turbines_data: DataFrame with columns ['turbine_id', 'power_output', 
                        'wind_speed', 'availability']
        
        Returns:
        - efficiency_metrics: Dictionary of efficiency metrics
        """
        if turbines_data.empty:
            return {
                'overall_efficiency': 0,
                'average_capacity_factor': 0,
                'availability': 0,
                'total_losses': 0
            }
        
        # Theoretical maximum power
        rated_power = 3000  # kW
        n_turbines = len(turbines_data)
        
        # Actual power
        actual_power = turbines_data['power_output'].sum()
        
        # Theoretical power based on wind conditions
        theoretical_power = 0
        for _, row in turbines_data.iterrows():
            ws = row['wind_speed']
            if ws < 3:
                theoretical_power += 0
            elif ws < 12:
                theoretical_power += 500 * (ws - 3) ** 2
            else:
                theoretical_power += min(rated_power, 500 + (ws - 12) * 200)
        
        # Efficiency
        efficiency = (actual_power / theoretical_power * 100) if theoretical_power > 0 else 0
        
        # Capacity factor
        capacity_factor = (actual_power / (rated_power * n_turbines)) * 100
        
        # Availability
        availability = turbines_data['availability'].mean() * 100
        
        # Losses
        losses = 100 - efficiency
        
        return {
            'overall_efficiency': round(efficiency, 2),
            'average_capacity_factor': round(capacity_factor, 2),
            'availability': round(availability, 2),
            'total_losses': round(losses, 2),
            'actual_power_kw': round(actual_power, 2),
            'theoretical_power_kw': round(theoretical_power, 2)
        }
    
    def identify_underperforming_turbines(self, turbines_data, threshold=80):
        """
        Identify turbines performing below threshold
        
        Parameters:
        - turbines_data: DataFrame with turbine performance data
        - threshold: Performance threshold percentage
        
        Returns:
        - underperforming: List of turbine IDs and their efficiency scores
        """
        underperforming = []
        
        for _, row in turbines_data.iterrows():
            ws = row['wind_speed']
            actual_power = row['power_output']
            
            # Calculate expected power
            if ws < 3:
                expected_power = 0
            elif ws < 12:
                expected_power = 500 * (ws - 3) ** 2
            else:
                expected_power = min(3000, 500 + (ws - 12) * 200)
            
            if expected_power > 0:
                efficiency = (actual_power / expected_power) * 100
                
                if efficiency < threshold:
                    underperforming.append({
                        'turbine_id': row.get('turbine_id', 'Unknown'),
                        'efficiency': round(efficiency, 2),
                        'power_loss_kw': round(expected_power - actual_power, 2)
                    })
        
        return underperforming


class PowerOutputForecaster:
    """
    Forecast power output to enable proactive maintenance scheduling
    and grid integration optimization
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.is_trained = False
    
    def train(self, X, y):
        """
        Train forecasting model
        
        Parameters:
        - X: Features (wind_speed, wind_direction, temperature, time_features)
        - y: Target (power output)
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """
        Predict power output
        
        Parameters:
        - X: Feature array
        
        Returns:
        - predictions: Power output predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def forecast_next_hours(self, current_data, hours=24):
        """
        Forecast power output for next N hours
        
        Parameters:
        - current_data: Current conditions dictionary
        - hours: Number of hours to forecast
        
        Returns:
        - forecast: DataFrame with hourly predictions
        """
        if not self.is_trained:
            # Return simplified forecast if model not trained
            forecasts = []
            base_power = current_data.get('power', 1000)
            
            for h in range(hours):
                # Simple sinusoidal variation
                variation = 1 + 0.3 * np.sin(h * np.pi / 12)
                forecasts.append({
                    'hour': h + 1,
                    'predicted_power_kw': round(base_power * variation, 2)
                })
            
            return pd.DataFrame(forecasts)
        
        # If model is trained, use actual predictions
        forecasts = []
        for h in range(hours):
            # Create feature vector (simplified)
            X = np.array([[
                current_data.get('wind_speed', 8),
                current_data.get('wind_direction', 180),
                current_data.get('temperature', 15),
                h  # Hour offset
            ]])
            
            pred = self.predict(X)[0]
            forecasts.append({
                'hour': h + 1,
                'predicted_power_kw': round(pred, 2)
            })
        
        return pd.DataFrame(forecasts)


class PerformanceBenchmarking:
    """
    Benchmark turbine performance against ideal models and fleet averages
    """
    
    @staticmethod
    def calculate_ideal_power(wind_speed, rated_power=3000):
        """
        Calculate ideal power output based on standard power curve
        
        Parameters:
        - wind_speed: Wind speed in m/s
        - rated_power: Turbine rated power in kW
        
        Returns:
        - ideal_power: Ideal power output in kW
        """
        if wind_speed < 3:
            return 0
        elif wind_speed < 12:
            return 500 * (wind_speed - 3) ** 2
        elif wind_speed < 25:
            return rated_power
        else:
            return 0  # Cut-off
    
    @staticmethod
    def calculate_performance_ratio(actual_power, ideal_power):
        """
        Calculate performance ratio
        
        Parameters:
        - actual_power: Actual power output
        - ideal_power: Ideal power output
        
        Returns:
        - performance_ratio: Ratio as percentage
        """
        if ideal_power == 0:
            return 100 if actual_power == 0 else 0
        
        return min(100, (actual_power / ideal_power) * 100)
    
    @staticmethod
    def benchmark_against_fleet(turbine_data, fleet_data):
        """
        Compare turbine performance against fleet average
        
        Parameters:
        - turbine_data: Dict with turbine metrics
        - fleet_data: DataFrame with fleet turbine data
        
        Returns:
        - benchmark_results: Dict with comparison metrics
        """
        fleet_avg_power = fleet_data['power_output'].mean()
        fleet_avg_efficiency = fleet_data['efficiency'].mean() if 'efficiency' in fleet_data else 85
        
        turbine_power = turbine_data.get('power', 0)
        turbine_efficiency = turbine_data.get('efficiency', 0)
        
        return {
            'power_vs_fleet': round((turbine_power / fleet_avg_power * 100) if fleet_avg_power > 0 else 0, 2),
            'efficiency_vs_fleet': round(turbine_efficiency - fleet_avg_efficiency, 2),
            'ranking': 'Above Average' if turbine_efficiency > fleet_avg_efficiency else 'Below Average'
        }


def calculate_energy_gains(baseline_data, optimized_data):
    """
    Calculate energy efficiency gains from optimization
    
    Parameters:
    - baseline_data: Dict with baseline performance metrics
    - optimized_data: Dict with optimized performance metrics
    
    Returns:
    - gains: Dict with improvement metrics
    """
    baseline_energy = baseline_data.get('total_energy_kwh', 0)
    optimized_energy = optimized_data.get('total_energy_kwh', 0)
    
    energy_gain = optimized_energy - baseline_energy
    energy_gain_pct = (energy_gain / baseline_energy * 100) if baseline_energy > 0 else 0
    
    # Estimate CO2 reduction (0.5 kg CO2 per kWh average)
    co2_reduction_kg = energy_gain * 0.5
    
    return {
        'energy_gain_kwh': round(energy_gain, 2),
        'energy_gain_percentage': round(energy_gain_pct, 2),
        'co2_reduction_kg': round(co2_reduction_kg, 2),
        'co2_reduction_tons': round(co2_reduction_kg / 1000, 2)
    }


if __name__ == "__main__":
    print("=" * 80)
    print("WIND FARM OPTIMIZATION MODULE")
    print("=" * 80)
    
    # Demo: Wind farm optimization
    print("\n1. Wind Farm Efficiency Optimization")
    print("-" * 80)
    
    optimizer = WindFarmOptimizer()
    
    # Simulate turbine data
    turbine_data = pd.DataFrame({
        'turbine_id': ['T1', 'T2', 'T3'],
        'power_output': [2500, 2300, 2100],
        'wind_speed': [10, 10, 10],
        'availability': [1.0, 0.98, 0.95]
    })
    
    efficiency_metrics = optimizer.calculate_farm_efficiency(turbine_data)
    print("\nFarm Efficiency Metrics:")
    for key, value in efficiency_metrics.items():
        print(f"  {key}: {value}")
    
    # Identify underperforming turbines
    print("\n2. Underperforming Turbines")
    print("-" * 80)
    underperforming = optimizer.identify_underperforming_turbines(turbine_data, threshold=85)
    if underperforming:
        for turbine in underperforming:
            print(f"  Turbine {turbine['turbine_id']}: {turbine['efficiency']}% "
                  f"(Loss: {turbine['power_loss_kw']} kW)")
    else:
        print("  All turbines performing above threshold")
    
    # Power forecasting
    print("\n3. Power Output Forecasting")
    print("-" * 80)
    forecaster = PowerOutputForecaster()
    current_conditions = {'power': 2000, 'wind_speed': 8}
    forecast = forecaster.forecast_next_hours(current_conditions, hours=6)
    print("\nNext 6 Hours Forecast:")
    print(forecast.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION MODULE READY")
    print("=" * 80)
