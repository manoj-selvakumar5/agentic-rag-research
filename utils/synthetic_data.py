import numpy as np
import json
from datetime import datetime, timedelta
import random

class EnergyDataGenerator:
    def __init__(self, start_date="2024/01/01", num_days=365, num_customers=10):
        self.start_date = datetime.strptime(start_date, "%Y/%m/%d")
        self.num_days = num_days
        self.num_customers = num_customers
        
    def _base_consumption(self):
        return 100.0
        
    def _seasonal_pattern(self, day_of_year):
        # Yearly seasonality
        return 20 * np.sin(2 * np.pi * day_of_year / 365)
    
    def _weekly_pattern(self, day_of_week):
        # Weekend vs weekday pattern
        patterns = {0: -10, 1: 0, 2: 0, 3: 0, 4: 0, 5: 10, 6: 10}
        return patterns[day_of_week]
        
    def _daily_noise(self):
        return np.random.normal(0, 5)
        
    def generate_data(self):
        data = []
        
        for customer_id in range(1, self.num_customers + 1):
            for day in range(self.num_days):
                current_date = self.start_date + timedelta(days=day)
                day_of_year = current_date.timetuple().tm_yday
                day_of_week = current_date.weekday()
                
                power_reading = (
                    self._base_consumption() +
                    self._seasonal_pattern(day_of_year) +
                    self._weekly_pattern(day_of_week) +
                    self._daily_noise()
                )
                
                data.append({
                    "customer_id": str(customer_id),
                    "day": current_date.strftime("%Y/%m/%d"),
                    "sumPowerReading": f"{max(0, power_reading):.1f}",
                    "kind": "measured"
                })
                
        return data
    
    def save_to_file(self, filename):
        data = self.generate_data()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Usage example
if __name__ == "__main__":
    generator = EnergyDataGenerator(start_date="2024/01/01", num_days=365, num_customers=5)
    generator.save_to_file("synthetic_energy_data.json")


# generator = EnergyDataGenerator()
# generator.save_to_file("energy_data.json")