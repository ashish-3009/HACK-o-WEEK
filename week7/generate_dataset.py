import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='H')

records = []

for dt in dates:
    hour = dt.hour
    day_of_week = dt.dayofweek
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0

    if month in [12, 1, 2]:
        seasonal_factor = 1.3
    elif month in [6, 7, 8]:
        seasonal_factor = 1.5
    elif month in [3, 4, 5]:
        seasonal_factor = 1.0
    else:
        seasonal_factor = 1.1

    if is_weekend:
        if 0 <= hour < 6:
            base_kwh = np.random.uniform(8, 15)
            hvac = np.random.uniform(3, 7)
            lighting = np.random.uniform(1, 3)
            equipment = np.random.uniform(2, 5)
            occupancy = np.random.randint(0, 5)
        elif 6 <= hour < 9:
            base_kwh = np.random.uniform(15, 25)
            hvac = np.random.uniform(5, 10)
            lighting = np.random.uniform(2, 5)
            equipment = np.random.uniform(4, 8)
            occupancy = np.random.randint(2, 15)
        elif 9 <= hour < 17:
            base_kwh = np.random.uniform(20, 35)
            hvac = np.random.uniform(8, 15)
            lighting = np.random.uniform(3, 7)
            equipment = np.random.uniform(5, 10)
            occupancy = np.random.randint(5, 30)
        elif 17 <= hour < 21:
            base_kwh = np.random.uniform(15, 25)
            hvac = np.random.uniform(5, 10)
            lighting = np.random.uniform(2, 5)
            equipment = np.random.uniform(3, 7)
            occupancy = np.random.randint(2, 10)
        else:
            base_kwh = np.random.uniform(8, 15)
            hvac = np.random.uniform(3, 7)
            lighting = np.random.uniform(1, 3)
            equipment = np.random.uniform(2, 5)
            occupancy = np.random.randint(0, 5)
    else:
        if 0 <= hour < 6:
            base_kwh = np.random.uniform(12, 22)
            hvac = np.random.uniform(5, 10)
            lighting = np.random.uniform(2, 5)
            equipment = np.random.uniform(3, 7)
            occupancy = np.random.randint(0, 8)
        elif 6 <= hour < 9:
            base_kwh = np.random.uniform(35, 55)
            hvac = np.random.uniform(12, 22)
            lighting = np.random.uniform(5, 10)
            equipment = np.random.uniform(10, 18)
            occupancy = np.random.randint(20, 60)
        elif 9 <= hour < 12:
            base_kwh = np.random.uniform(60, 90)
            hvac = np.random.uniform(20, 35)
            lighting = np.random.uniform(8, 15)
            equipment = np.random.uniform(15, 25)
            occupancy = np.random.randint(80, 150)
        elif 12 <= hour < 14:
            base_kwh = np.random.uniform(55, 80)
            hvac = np.random.uniform(18, 30)
            lighting = np.random.uniform(7, 13)
            equipment = np.random.uniform(12, 22)
            occupancy = np.random.randint(60, 120)
        elif 14 <= hour < 17:
            base_kwh = np.random.uniform(60, 85)
            hvac = np.random.uniform(20, 33)
            lighting = np.random.uniform(8, 14)
            equipment = np.random.uniform(14, 24)
            occupancy = np.random.randint(75, 140)
        elif 17 <= hour < 20:
            base_kwh = np.random.uniform(30, 50)
            hvac = np.random.uniform(10, 20)
            lighting = np.random.uniform(5, 10)
            equipment = np.random.uniform(8, 15)
            occupancy = np.random.randint(15, 50)
        elif 20 <= hour < 22:
            base_kwh = np.random.uniform(18, 30)
            hvac = np.random.uniform(6, 12)
            lighting = np.random.uniform(3, 7)
            equipment = np.random.uniform(4, 9)
            occupancy = np.random.randint(5, 20)
        else:
            base_kwh = np.random.uniform(12, 20)
            hvac = np.random.uniform(4, 9)
            lighting = np.random.uniform(2, 4)
            equipment = np.random.uniform(3, 6)
            occupancy = np.random.randint(0, 8)

    base_kwh *= seasonal_factor
    hvac *= seasonal_factor
    lighting *= seasonal_factor
    equipment *= seasonal_factor

    if month in [6, 7, 8]:
        temp = np.random.uniform(28, 42)
    elif month in [12, 1, 2]:
        temp = np.random.uniform(5, 18)
    elif month in [3, 4, 5]:
        temp = np.random.uniform(18, 30)
    else:
        temp = np.random.uniform(20, 32)

    humidity = np.random.uniform(30, 85)

    total_kwh = base_kwh + hvac + lighting + equipment + np.random.normal(0, 2)
    total_kwh = max(total_kwh, 5)

    records.append({
        'timestamp': dt,
        'date': dt.date(),
        'hour': hour,
        'day_of_week': day_of_week,
        'day_name': dt.strftime('%A'),
        'month': month,
        'is_weekend': is_weekend,
        'total_kwh': round(total_kwh, 2),
        'hvac_kwh': round(hvac, 2),
        'lighting_kwh': round(lighting, 2),
        'equipment_kwh': round(equipment, 2),
        'occupancy': occupancy,
        'temperature_c': round(temp, 1),
        'humidity_pct': round(humidity, 1)
    })

df = pd.DataFrame(records)
df.to_csv('dataset/admin_building_usage.csv', index=False)

print(f"Dataset created: {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {list(df.columns)}")
print(f"\nWeekday avg kWh: {df[df['is_weekend']==0]['total_kwh'].mean():.2f}")
print(f"Weekend avg kWh: {df[df['is_weekend']==1]['total_kwh'].mean():.2f}")
print(f"Weekend dip: {((df[df['is_weekend']==0]['total_kwh'].mean() - df[df['is_weekend']==1]['total_kwh'].mean()) / df[df['is_weekend']==0]['total_kwh'].mean() * 100):.1f}%")
