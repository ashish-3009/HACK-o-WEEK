# Admin Building Weekend Dip - Energy Analytics

K-Means Clustering + Gradient Boosting Regression pipeline for analyzing admin building energy usage patterns, forecasting consumption, and quantifying weekend savings potential.

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy (100 - MAPE) | **91.18%** |
| Avg R² Score | **0.9489** |
| Avg MAE | 6.26 kWh |
| Avg RMSE | 8.24 kWh |
| Silhouette Score | 0.4295 |
| Clusters Found | 3 |

## Architecture

```
├── generate_dataset.py      # Synthetic 3-year hourly dataset generator
├── model.py                 # ML pipeline (K-Means + GradientBoosting)
├── app.py                   # Flask web server + REST API
├── templates/
│   └── dashboard.html       # Interactive analytics dashboard
├── dataset/
│   └── admin_building_usage.csv
├── model_artifacts/         # Saved models & metrics
└── requirements.txt
```

## Pipeline

1. **Data Generation** - 26,280 hourly records (2023-2025) with HVAC, lighting, equipment, occupancy, weather
2. **Feature Engineering** - Daily aggregations, peak/off-peak ratios, component shares
3. **K-Means Clustering** - Auto-selects optimal K via silhouette score (found K=3)
4. **Per-Cluster Regression** - GradientBoostingRegressor per cluster for targeted forecasting
5. **Savings Analysis** - Weekend dip quantification, component-level savings breakdown

## Setup & Run

```bash
pip install -r requirements.txt
python generate_dataset.py
python app.py
```

Dashboard: http://localhost:5000

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/dashboard` | All dashboard data |
| `/api/metrics` | Model performance metrics |
| `/api/clusters` | Cluster profiles |
| `/api/savings` | Savings breakdown |
| `/api/forecasts` | 7-day hourly forecasts |
| `/api/hourly` | Hourly usage profiles |
| `/api/monthly` | Monthly trends |
