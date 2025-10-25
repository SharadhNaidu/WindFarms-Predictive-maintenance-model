"""
Operational Cost Optimization and Smart Maintenance Scheduling
Reduces downtime and operational costs through data-driven decision making
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DowntimeCostCalculator:
    """
    Calculate operational costs including maintenance, downtime, and energy losses
    """
    
    def __init__(self, electricity_price_per_kwh=0.08, maintenance_hourly_rate=150):
        """
        Initialize cost calculator
        
        Parameters:
        - electricity_price_per_kwh: Average electricity price ($/kWh)
        - maintenance_hourly_rate: Cost per hour for maintenance crew ($/hour)
        """
        self.electricity_price = electricity_price_per_kwh
        self.maintenance_rate = maintenance_hourly_rate
        
    def calculate_downtime_cost(self, downtime_hours, avg_power_kw):
        """
        Calculate revenue loss from downtime
        
        Parameters:
        - downtime_hours: Hours of downtime
        - avg_power_kw: Average power output during lost period
        
        Returns:
        - cost: Lost revenue in dollars
        """
        energy_loss_kwh = downtime_hours * avg_power_kw
        revenue_loss = energy_loss_kwh * self.electricity_price
        
        return {
            'energy_loss_kwh': round(energy_loss_kwh, 2),
            'revenue_loss_usd': round(revenue_loss, 2),
            'downtime_hours': downtime_hours
        }
    
    def calculate_maintenance_cost(self, labor_hours, parts_cost=0, travel_cost=0):
        """
        Calculate total maintenance cost
        
        Parameters:
        - labor_hours: Hours of labor required
        - parts_cost: Cost of replacement parts ($)
        - travel_cost: Travel and logistics costs ($)
        
        Returns:
        - cost_breakdown: Dictionary with cost components
        """
        labor_cost = labor_hours * self.maintenance_rate
        total_cost = labor_cost + parts_cost + travel_cost
        
        return {
            'labor_cost_usd': round(labor_cost, 2),
            'parts_cost_usd': round(parts_cost, 2),
            'travel_cost_usd': round(travel_cost, 2),
            'total_maintenance_cost_usd': round(total_cost, 2)
        }
    
    def calculate_total_operational_cost(self, failure_type, downtime_hours, avg_power_kw):
        """
        Calculate total operational cost for a failure event
        
        Parameters:
        - failure_type: Type of failure ('gearbox', 'generator', 'blade', etc.)
        - downtime_hours: Expected downtime
        - avg_power_kw: Average power output
        
        Returns:
        - total_cost: Complete cost breakdown
        """
        # Typical cost estimates by component
        component_costs = {
            'gearbox': {'labor_hours': 40, 'parts': 50000, 'travel': 2000},
            'generator': {'labor_hours': 32, 'parts': 35000, 'travel': 1500},
            'blade': {'labor_hours': 60, 'parts': 100000, 'travel': 5000},
            'yaw': {'labor_hours': 16, 'parts': 15000, 'travel': 1000},
            'pitch': {'labor_hours': 20, 'parts': 20000, 'travel': 1200},
            'sensor': {'labor_hours': 4, 'parts': 500, 'travel': 500},
            'minor': {'labor_hours': 8, 'parts': 2000, 'travel': 500}
        }
        
        # Get cost parameters
        params = component_costs.get(failure_type, component_costs['minor'])
        
        # Calculate costs
        downtime_cost = self.calculate_downtime_cost(downtime_hours, avg_power_kw)
        maintenance_cost = self.calculate_maintenance_cost(
            params['labor_hours'],
            params['parts'],
            params['travel']
        )
        
        total = downtime_cost['revenue_loss_usd'] + maintenance_cost['total_maintenance_cost_usd']
        
        return {
            'failure_type': failure_type,
            'downtime_cost': downtime_cost,
            'maintenance_cost': maintenance_cost,
            'total_cost_usd': round(total, 2)
        }
    
    def calculate_preventive_savings(self, reactive_cost, preventive_cost):
        """
        Calculate savings from preventive vs reactive maintenance
        
        Parameters:
        - reactive_cost: Cost of reactive maintenance
        - preventive_cost: Cost of preventive maintenance
        
        Returns:
        - savings: Cost savings and ROI
        """
        savings = reactive_cost - preventive_cost
        roi = (savings / preventive_cost * 100) if preventive_cost > 0 else 0
        
        return {
            'reactive_cost_usd': round(reactive_cost, 2),
            'preventive_cost_usd': round(preventive_cost, 2),
            'savings_usd': round(savings, 2),
            'roi_percentage': round(roi, 2)
        }
    
    def estimate_annual_costs(self, failure_rates, avg_power_kw):
        """
        Estimate annual operational costs based on failure rates
        
        Parameters:
        - failure_rates: Dict with failure rates per year for each component
        - avg_power_kw: Average power output
        
        Returns:
        - annual_costs: Projected annual cost breakdown
        """
        total_annual_cost = 0
        cost_breakdown = {}
        
        for component, failures_per_year in failure_rates.items():
            # Typical downtime per failure
            downtime_map = {
                'gearbox': 120, 'generator': 96, 'blade': 168,
                'yaw': 48, 'pitch': 60, 'sensor': 8, 'minor': 24
            }
            
            downtime = downtime_map.get(component, 24)
            
            # Calculate cost per failure
            cost_per_failure = self.calculate_total_operational_cost(
                component, downtime, avg_power_kw
            )['total_cost_usd']
            
            # Annual cost
            annual_component_cost = cost_per_failure * failures_per_year
            cost_breakdown[component] = round(annual_component_cost, 2)
            total_annual_cost += annual_component_cost
        
        return {
            'breakdown': cost_breakdown,
            'total_annual_cost_usd': round(total_annual_cost, 2)
        }


class SmartMaintenanceScheduler:
    """
    Intelligent maintenance scheduling to minimize costs and maximize availability
    """
    
    def __init__(self):
        self.schedule = []
        
    def calculate_optimal_maintenance_window(self, 
                                            failure_probability,
                                            current_health_score,
                                            weather_forecast,
                                            energy_prices):
        """
        Determine optimal maintenance timing
        
        Parameters:
        - failure_probability: Probability of failure in next 30 days (0-1)
        - current_health_score: Current component health (0-100)
        - weather_forecast: List of weather conditions for next 14 days
        - energy_prices: List of energy prices for next 14 days
        
        Returns:
        - optimal_day: Best day for maintenance (0-13)
        - urgency: Urgency level ('critical', 'high', 'medium', 'low')
        """
        # Determine urgency
        if failure_probability > 0.7 or current_health_score < 30:
            urgency = 'critical'
            max_delay = 3  # Days
        elif failure_probability > 0.4 or current_health_score < 60:
            urgency = 'high'
            max_delay = 7
        elif failure_probability > 0.2 or current_health_score < 80:
            urgency = 'medium'
            max_delay = 14
        else:
            urgency = 'low'
            max_delay = 30
        
        # Find optimal day within allowed window
        optimal_day = 0
        min_cost = float('inf')
        
        for day in range(min(max_delay, len(weather_forecast), len(energy_prices))):
            # Weather suitability (0=bad, 1=good)
            weather_score = weather_forecast[day]
            
            # Energy price (lower is better for scheduling downtime)
            price = energy_prices[day]
            
            # Combined cost score (lower is better)
            cost_score = price * (2 - weather_score)  # Bad weather increases effective cost
            
            if cost_score < min_cost:
                min_cost = cost_score
                optimal_day = day
        
        return {
            'optimal_day': optimal_day,
            'urgency': urgency,
            'max_delay_days': max_delay,
            'recommended_date': (datetime.now() + timedelta(days=optimal_day)).strftime('%Y-%m-%d')
        }
    
    def prioritize_maintenance_tasks(self, components_health):
        """
        Prioritize multiple maintenance tasks
        
        Parameters:
        - components_health: Dict with component names and health scores
        
        Returns:
        - priority_list: Sorted list of maintenance priorities
        """
        priorities = []
        
        for component, health in components_health.items():
            # Calculate priority score
            if health < 30:
                priority = 1  # Critical
                impact = 'high'
            elif health < 60:
                priority = 2  # High
                impact = 'medium'
            elif health < 80:
                priority = 3  # Medium
                impact = 'low'
            else:
                priority = 4  # Low
                impact = 'minimal'
            
            priorities.append({
                'component': component,
                'health_score': health,
                'priority': priority,
                'impact': impact
            })
        
        # Sort by priority (lower number = higher priority)
        priorities.sort(key=lambda x: (x['priority'], x['health_score']))
        
        return priorities
    
    def generate_maintenance_schedule(self, components_health, forecast_days=30):
        """
        Generate comprehensive maintenance schedule
        
        Parameters:
        - components_health: Dict with component health and failure probabilities
        - forecast_days: Days to schedule ahead
        
        Returns:
        - schedule: List of scheduled maintenance activities
        """
        schedule = []
        
        # Mock weather and price forecasts (in real system, fetch from APIs)
        weather_forecast = [0.8 + 0.2 * np.random.random() for _ in range(forecast_days)]
        energy_prices = [0.08 + 0.04 * np.random.random() for _ in range(forecast_days)]
        
        for component, data in components_health.items():
            health = data.get('health_score', 100)
            failure_prob = data.get('failure_probability', 0.0)
            
            if health < 90 or failure_prob > 0.1:  # Needs attention
                optimal_window = self.calculate_optimal_maintenance_window(
                    failure_prob,
                    health,
                    weather_forecast[:14],
                    energy_prices[:14]
                )
                
                schedule.append({
                    'component': component,
                    'health_score': health,
                    'failure_probability': round(failure_prob * 100, 2),
                    'urgency': optimal_window['urgency'],
                    'recommended_date': optimal_window['recommended_date'],
                    'days_from_now': optimal_window['optimal_day']
                })
        
        # Sort by urgency and days
        urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        schedule.sort(key=lambda x: (urgency_order[x['urgency']], x['days_from_now']))
        
        return schedule


class CostBenefitAnalyzer:
    """
    Analyze cost-benefit of predictive maintenance vs reactive maintenance
    """
    
    @staticmethod
    def calculate_predictive_maintenance_roi(baseline_costs, predictive_costs, 
                                            avoided_failures):
        """
        Calculate ROI of predictive maintenance system
        
        Parameters:
        - baseline_costs: Annual costs with reactive maintenance
        - predictive_costs: Annual costs with predictive system (including system cost)
        - avoided_failures: Number of major failures avoided
        
        Returns:
        - roi_analysis: Complete ROI breakdown
        """
        savings = baseline_costs - predictive_costs
        roi = (savings / predictive_costs * 100) if predictive_costs > 0 else 0
        cost_per_avoided_failure = savings / avoided_failures if avoided_failures > 0 else 0
        
        return {
            'annual_baseline_cost_usd': round(baseline_costs, 2),
            'annual_predictive_cost_usd': round(predictive_costs, 2),
            'annual_savings_usd': round(savings, 2),
            'roi_percentage': round(roi, 2),
            'avoided_failures': avoided_failures,
            'cost_per_avoided_failure_usd': round(cost_per_avoided_failure, 2),
            'payback_period_months': round((predictive_costs / (savings / 12)) if savings > 0 else 0, 1)
        }
    
    @staticmethod
    def estimate_system_value(annual_energy_output_mwh, uptime_improvement_pct):
        """
        Estimate value of improved system reliability
        
        Parameters:
        - annual_energy_output_mwh: Annual energy output in MWh
        - uptime_improvement_pct: Improvement in uptime percentage
        
        Returns:
        - value_estimate: Estimated value
        """
        electricity_price = 0.08  # $/kWh
        
        # Additional energy captured
        additional_energy_mwh = annual_energy_output_mwh * (uptime_improvement_pct / 100)
        additional_revenue = additional_energy_mwh * 1000 * electricity_price
        
        # Avoided emergency repairs (estimate)
        avoided_emergency_cost = uptime_improvement_pct * 5000  # Rough estimate
        
        total_value = additional_revenue + avoided_emergency_cost
        
        return {
            'additional_energy_mwh': round(additional_energy_mwh, 2),
            'additional_revenue_usd': round(additional_revenue, 2),
            'avoided_emergency_costs_usd': round(avoided_emergency_cost, 2),
            'total_annual_value_usd': round(total_value, 2)
        }


if __name__ == "__main__":
    print("=" * 80)
    print("OPERATIONAL COST OPTIMIZATION MODULE")
    print("=" * 80)
    
    # Demo: Downtime cost calculation
    print("\n1. Downtime Cost Analysis")
    print("-" * 80)
    
    calculator = DowntimeCostCalculator()
    
    # Example: Gearbox failure
    gearbox_cost = calculator.calculate_total_operational_cost(
        failure_type='gearbox',
        downtime_hours=120,
        avg_power_kw=2000
    )
    
    print("\nGearbox Failure Cost:")
    print(f"  Total Cost: ${gearbox_cost['total_cost_usd']:,}")
    print(f"  Downtime Loss: ${gearbox_cost['downtime_cost']['revenue_loss_usd']:,}")
    print(f"  Maintenance: ${gearbox_cost['maintenance_cost']['total_maintenance_cost_usd']:,}")
    
    # Demo: Preventive vs Reactive
    print("\n2. Preventive Maintenance Savings")
    print("-" * 80)
    
    savings = calculator.calculate_preventive_savings(
        reactive_cost=gearbox_cost['total_cost_usd'],
        preventive_cost=15000
    )
    
    print(f"  Reactive Cost: ${savings['reactive_cost_usd']:,}")
    print(f"  Preventive Cost: ${savings['preventive_cost_usd']:,}")
    print(f"  Savings: ${savings['savings_usd']:,}")
    print(f"  ROI: {savings['roi_percentage']}%")
    
    # Demo: Smart scheduling
    print("\n3. Smart Maintenance Scheduling")
    print("-" * 80)
    
    scheduler = SmartMaintenanceScheduler()
    
    components = {
        'gearbox': {'health_score': 45, 'failure_probability': 0.35},
        'generator': {'health_score': 72, 'failure_probability': 0.15},
        'blade': {'health_score': 88, 'failure_probability': 0.05}
    }
    
    schedule = scheduler.generate_maintenance_schedule(components)
    
    print("\nRecommended Maintenance Schedule:")
    for task in schedule:
        print(f"  {task['component'].upper()}: {task['urgency'].upper()} "
              f"- Schedule for {task['recommended_date']} "
              f"(Health: {task['health_score']}%)")
    
    # Demo: ROI Analysis
    print("\n4. Predictive Maintenance ROI")
    print("-" * 80)
    
    roi = CostBenefitAnalyzer.calculate_predictive_maintenance_roi(
        baseline_costs=250000,
        predictive_costs=180000,
        avoided_failures=3
    )
    
    print(f"  Annual Savings: ${roi['annual_savings_usd']:,}")
    print(f"  ROI: {roi['roi_percentage']}%")
    print(f"  Payback Period: {roi['payback_period_months']} months")
    print(f"  Cost per Avoided Failure: ${roi['cost_per_avoided_failure_usd']:,}")
    
    print("\n" + "=" * 80)
    print("COST OPTIMIZATION MODULE READY")
    print("=" * 80)
