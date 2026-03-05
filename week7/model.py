import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, calinski_harabasz_score
)
import joblib
import os
import json


class AdminBuildingModel:
    def __init__(self, data_path='dataset/admin_building_usage.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.kmeans = None
        self.regressors = {}
        self.cluster_profiles = {}
        self.metrics = {}
        self.savings_data = {}
        self.df = pd.DataFrame()
        self.daily_data = pd.DataFrame()
        self.hourly_clustered = pd.DataFrame()
        self.n_clusters = 4

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        return self.df

    def prepare_clustering_features(self):
        daily = self.df.groupby('date').agg({
            'total_kwh': ['mean', 'std', 'min', 'max', 'sum'],
            'hvac_kwh': 'mean',
            'lighting_kwh': 'mean',
            'equipment_kwh': 'mean',
            'occupancy': 'mean',
            'temperature_c': 'mean',
            'is_weekend': 'first',
            'day_of_week': 'first',
            'month': 'first'
        }).reset_index()

        daily.columns = [
            'date', 'kwh_mean', 'kwh_std', 'kwh_min', 'kwh_max', 'kwh_sum',
            'hvac_mean', 'lighting_mean', 'equipment_mean',
            'occupancy_mean', 'temp_mean', 'is_weekend', 'day_of_week', 'month'
        ]

        peak_hours = self.df[self.df['hour'].between(9, 17)].groupby('date')['total_kwh'].mean().reset_index()
        peak_hours.columns = ['date', 'peak_kwh']

        offpeak_hours = self.df[~self.df['hour'].between(9, 17)].groupby('date')['total_kwh'].mean().reset_index()
        offpeak_hours.columns = ['date', 'offpeak_kwh']

        daily['date'] = daily['date'].astype(str)
        peak_hours['date'] = peak_hours['date'].astype(str)
        offpeak_hours['date'] = offpeak_hours['date'].astype(str)

        daily = daily.merge(peak_hours, on='date', how='left')
        daily = daily.merge(offpeak_hours, on='date', how='left')
        daily = daily.fillna(0)
        daily['peak_offpeak_ratio'] = daily['peak_kwh'] / (daily['offpeak_kwh'] + 0.01)
        daily['hvac_share'] = daily['hvac_mean'] / (daily['kwh_mean'] + 0.01)
        daily['lighting_share'] = daily['lighting_mean'] / (daily['kwh_mean'] + 0.01)
        daily['equipment_share'] = daily['equipment_mean'] / (daily['kwh_mean'] + 0.01)
        daily = daily.fillna(0)

        self.daily_data = daily
        return daily

    def run_clustering(self):
        feature_cols = [
            'kwh_mean', 'kwh_std', 'kwh_min', 'kwh_max',
            'hvac_mean', 'lighting_mean', 'equipment_mean',
            'occupancy_mean', 'peak_offpeak_ratio',
            'hvac_share', 'lighting_share', 'equipment_share',
            'is_weekend', 'temp_mean'
        ]

        X = self.daily_data[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        best_score = -1
        best_k = 4
        for k in range(3, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

        self.n_clusters = best_k
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
        self.daily_data['cluster'] = self.kmeans.fit_predict(X_scaled)

        self.metrics['clustering'] = {
            'n_clusters': best_k,
            'silhouette_score': round(silhouette_score(X_scaled, self.daily_data['cluster']), 4),
            'calinski_harabasz': round(calinski_harabasz_score(X_scaled, self.daily_data['cluster']), 2),
            'inertia': round(self.kmeans.inertia_, 2)
        }

        for c in range(best_k):
            cluster_data = self.daily_data[self.daily_data['cluster'] == c]
            weekend_pct = cluster_data['is_weekend'].mean() * 100
            label = self._assign_cluster_label(cluster_data, weekend_pct)

            self.cluster_profiles[c] = {
                'label': label,
                'count': int(len(cluster_data)),
                'avg_kwh': round(cluster_data['kwh_mean'].mean(), 2),
                'avg_occupancy': round(cluster_data['occupancy_mean'].mean(), 1),
                'weekend_pct': round(weekend_pct, 1),
                'avg_temp': round(cluster_data['temp_mean'].mean(), 1),
                'hvac_share': round(cluster_data['hvac_share'].mean() * 100, 1),
                'lighting_share': round(cluster_data['lighting_share'].mean() * 100, 1),
                'equipment_share': round(cluster_data['equipment_share'].mean() * 100, 1),
                'peak_ratio': round(cluster_data['peak_offpeak_ratio'].mean(), 2)
            }

        return self.daily_data

    def _assign_cluster_label(self, cluster_data, weekend_pct):
        avg_kwh = cluster_data['kwh_mean'].mean()
        avg_occ = cluster_data['occupancy_mean'].mean()

        if weekend_pct > 60:
            return "Weekend Low-Usage"
        elif avg_kwh > self.daily_data['kwh_mean'].quantile(0.75):
            if avg_occ > self.daily_data['occupancy_mean'].quantile(0.7):
                return "Peak Weekday High-Load"
            else:
                return "High HVAC Demand"
        elif avg_kwh < self.daily_data['kwh_mean'].quantile(0.25):
            return "Minimal Usage Period"
        else:
            return "Standard Weekday"

    def train_regressors(self):
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature_c', 'humidity_pct', 'occupancy'
        ]
        target = 'total_kwh'

        hourly_with_cluster = self.df.copy()
        hourly_with_cluster['date'] = hourly_with_cluster['timestamp'].dt.date.astype(str)
        self.daily_data['date'] = self.daily_data['date'].astype(str)

        cluster_map = self.daily_data.set_index('date')['cluster'].to_dict()
        hourly_with_cluster['cluster'] = hourly_with_cluster['date'].map(cluster_map)
        hourly_with_cluster = hourly_with_cluster.dropna(subset=['cluster'])
        hourly_with_cluster['cluster'] = hourly_with_cluster['cluster'].astype(int)

        self.hourly_clustered = hourly_with_cluster

        overall_metrics = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}

        for c in range(self.n_clusters):
            cluster_df = hourly_with_cluster[hourly_with_cluster['cluster'] == c]

            if len(cluster_df) < 50:
                continue

            X = cluster_df[feature_cols].values
            y = cluster_df[target].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 0.01))) * 100

            self.regressors[c] = model
            self.metrics[f'cluster_{c}_regression'] = {
                'mae': round(float(mae), 4),
                'rmse': round(float(rmse), 4),
                'r2': round(float(r2), 4),
                'mape': round(float(mape), 2),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

            overall_metrics['mae'].append(mae)
            overall_metrics['rmse'].append(rmse)
            overall_metrics['r2'].append(r2)
            overall_metrics['mape'].append(mape)

            feature_imp = dict(zip(feature_cols, model.feature_importances_.tolist()))
            self.metrics[f'cluster_{c}_feature_importance'] = {
                k: round(v, 4) for k, v in sorted(feature_imp.items(), key=lambda x: -x[1])
            }

        self.metrics['overall_regression'] = {
            'avg_mae': round(float(np.mean(overall_metrics['mae'])), 4),
            'avg_rmse': round(float(np.mean(overall_metrics['rmse'])), 4),
            'avg_r2': round(float(np.mean(overall_metrics['r2'])), 4),
            'avg_mape': round(float(np.mean(overall_metrics['mape'])), 2),
            'accuracy_pct': round(100.0 - float(np.mean(overall_metrics['mape'])), 2)
        }

        return self.metrics

    def calculate_savings(self):
        weekday_data = self.daily_data[self.daily_data['is_weekend'] == 0]
        weekend_data = self.daily_data[self.daily_data['is_weekend'] == 1]

        weekday_avg = weekday_data['kwh_sum'].mean()
        weekend_avg = weekend_data['kwh_sum'].mean()
        weekend_optimal = weekday_avg * 0.25

        current_weekend_annual = weekend_avg * 104
        optimal_weekend_annual = weekend_optimal * 104
        potential_savings_kwh = current_weekend_annual - optimal_weekend_annual

        rate_per_kwh = 0.12
        potential_savings_usd = potential_savings_kwh * rate_per_kwh

        total_annual = self.daily_data['kwh_sum'].sum() / 3
        savings_pct = (potential_savings_kwh / total_annual) * 100

        hvac_weekend = weekend_data['hvac_mean'].mean()
        lighting_weekend = weekend_data['lighting_mean'].mean()
        equipment_weekend = weekend_data['equipment_mean'].mean()

        hvac_savings = hvac_weekend * 0.6 * 104 * 24
        lighting_savings = lighting_weekend * 0.7 * 104 * 24
        equipment_savings = equipment_weekend * 0.4 * 104 * 24
        other_savings = potential_savings_kwh - hvac_savings - lighting_savings - equipment_savings
        if other_savings < 0:
            other_savings = potential_savings_kwh * 0.05
            total_cat = hvac_savings + lighting_savings + equipment_savings + other_savings
            hvac_savings = hvac_savings / total_cat * potential_savings_kwh
            lighting_savings = lighting_savings / total_cat * potential_savings_kwh
            equipment_savings = equipment_savings / total_cat * potential_savings_kwh
            other_savings = other_savings / total_cat * potential_savings_kwh

        self.savings_data = {
            'weekday_avg_kwh': round(weekday_avg, 2),
            'weekend_avg_kwh': round(weekend_avg, 2),
            'weekend_optimal_kwh': round(weekend_optimal, 2),
            'dip_percentage': round((1 - weekend_avg / weekday_avg) * 100, 1),
            'potential_savings_kwh': round(potential_savings_kwh, 0),
            'potential_savings_usd': round(potential_savings_usd, 0),
            'savings_pct_of_total': round(savings_pct, 1),
            'breakdown': {
                'hvac': round(hvac_savings * rate_per_kwh, 0),
                'lighting': round(lighting_savings * rate_per_kwh, 0),
                'equipment': round(equipment_savings * rate_per_kwh, 0),
                'other': round(other_savings * rate_per_kwh, 0)
            },
            'breakdown_kwh': {
                'hvac': round(hvac_savings, 0),
                'lighting': round(lighting_savings, 0),
                'equipment': round(equipment_savings, 0),
                'other': round(other_savings, 0)
            }
        }

        return self.savings_data

    def generate_forecasts(self):
        forecasts = {}
        for day_name, dow in [('Monday', 0), ('Tuesday', 1), ('Wednesday', 2),
                               ('Thursday', 3), ('Friday', 4), ('Saturday', 5), ('Sunday', 6)]:
            hourly_forecast = []
            is_weekend = 1 if dow >= 5 else 0

            day_data = self.df[self.df['day_of_week'] == dow]
            avg_temp = day_data['temperature_c'].mean()
            avg_humidity = day_data['humidity_pct'].mean()
            avg_occupancy_by_hour = day_data.groupby('hour')['occupancy'].mean()

            for hour in range(24):
                occ = avg_occupancy_by_hour.get(hour, 10)
                features = np.array([[hour, dow, 6, is_weekend, avg_temp, avg_humidity, occ]])

                date_key = f"2025-06-{15 + dow}"
                cluster = self.daily_data[
                    (self.daily_data['is_weekend'] == is_weekend)
                ]['cluster'].mode()
                c = cluster.iloc[0] if len(cluster) > 0 else 0

                if c in self.regressors:
                    pred = self.regressors[c].predict(features)[0]
                else:
                    pred = self.df[
                        (self.df['hour'] == hour) & (self.df['is_weekend'] == is_weekend)
                    ]['total_kwh'].mean()

                hourly_forecast.append(round(max(pred, 0), 2))

            forecasts[day_name] = hourly_forecast

        return forecasts

    def get_hourly_profiles(self):
        weekday = self.df[self.df['is_weekend'] == 0].groupby('hour').agg({
            'total_kwh': 'mean',
            'hvac_kwh': 'mean',
            'lighting_kwh': 'mean',
            'equipment_kwh': 'mean',
            'occupancy': 'mean'
        }).round(2).to_dict()

        weekend = self.df[self.df['is_weekend'] == 1].groupby('hour').agg({
            'total_kwh': 'mean',
            'hvac_kwh': 'mean',
            'lighting_kwh': 'mean',
            'equipment_kwh': 'mean',
            'occupancy': 'mean'
        }).round(2).to_dict()

        return {'weekday': weekday, 'weekend': weekend}

    def get_monthly_trends(self):
        monthly = self.df.groupby(['month', 'is_weekend']).agg({
            'total_kwh': 'mean'
        }).reset_index()

        result = {}
        for _, row in monthly.iterrows():
            m = int(row['month'])
            key = 'weekend' if row['is_weekend'] == 1 else 'weekday'
            if m not in result:
                result[m] = {}
            result[m][key] = round(row['total_kwh'], 2)

        return result

    def run_pipeline(self):
        print("Loading data...")
        self.load_data()
        print("Preparing clustering features...")
        self.prepare_clustering_features()
        print("Running K-Means clustering...")
        self.run_clustering()
        print("Training cluster-specific regressors...")
        self.train_regressors()
        print("Calculating savings potential...")
        self.calculate_savings()
        print("Pipeline complete!")
        return self.metrics

    def save_model(self, path='model_artifacts'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.kmeans, os.path.join(path, 'kmeans.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        for c, reg in self.regressors.items():
            joblib.dump(reg, os.path.join(path, f'regressor_cluster_{c}.pkl'))
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        with open(os.path.join(path, 'cluster_profiles.json'), 'w') as f:
            json.dump(self.cluster_profiles, f, indent=2)
        with open(os.path.join(path, 'savings.json'), 'w') as f:
            json.dump(self.savings_data, f, indent=2)
        print(f"Model artifacts saved to {path}/")


if __name__ == '__main__':
    model = AdminBuildingModel()
    metrics = model.run_pipeline()

    print("\n===== CLUSTERING METRICS =====")
    print(json.dumps(model.metrics['clustering'], indent=2))

    print("\n===== CLUSTER PROFILES =====")
    for c, profile in model.cluster_profiles.items():
        print(f"\nCluster {c} - {profile['label']}:")
        print(f"  Days: {profile['count']}, Avg kWh: {profile['avg_kwh']}, Weekend%: {profile['weekend_pct']}")

    print("\n===== REGRESSION METRICS =====")
    print(json.dumps(model.metrics['overall_regression'], indent=2))

    print("\n===== SAVINGS POTENTIAL =====")
    print(json.dumps(model.savings_data, indent=2))

    model.save_model()
