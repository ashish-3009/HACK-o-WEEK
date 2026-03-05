from flask import Flask, render_template, jsonify
from model import AdminBuildingModel
import json
import os

app = Flask(__name__)

model = None
dashboard_data = {}


def initialize():
    global model, dashboard_data

    if not os.path.exists('dataset/admin_building_usage.csv'):
        import subprocess
        subprocess.run(['python', 'generate_dataset.py'], check=True)

    model = AdminBuildingModel()
    model.run_pipeline()
    model.save_model()

    forecasts = model.generate_forecasts()
    hourly_profiles = model.get_hourly_profiles()
    monthly_trends = model.get_monthly_trends()

    cluster_profiles_list = []
    for c in sorted(model.cluster_profiles.keys()):
        p = model.cluster_profiles[c]
        p['id'] = c
        cluster_profiles_list.append(p)

    dashboard_data = {
        'metrics': model.metrics,
        'cluster_profiles': cluster_profiles_list,
        'savings': model.savings_data,
        'forecasts': forecasts,
        'hourly_profiles': hourly_profiles,
        'monthly_trends': monthly_trends
    }


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/dashboard')
def api_dashboard():
    return jsonify(dashboard_data)


@app.route('/api/metrics')
def api_metrics():
    return jsonify(dashboard_data.get('metrics', {}))


@app.route('/api/clusters')
def api_clusters():
    return jsonify(dashboard_data.get('cluster_profiles', []))


@app.route('/api/savings')
def api_savings():
    return jsonify(dashboard_data.get('savings', {}))


@app.route('/api/forecasts')
def api_forecasts():
    return jsonify(dashboard_data.get('forecasts', {}))


@app.route('/api/hourly')
def api_hourly():
    return jsonify(dashboard_data.get('hourly_profiles', {}))


@app.route('/api/monthly')
def api_monthly():
    return jsonify(dashboard_data.get('monthly_trends', {}))


if __name__ == '__main__':
    initialize()
    app.run(debug=False, port=5000)
