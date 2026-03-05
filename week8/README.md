# 🅿️ Parking Lot Lighting Forecast

> Sensor-based vehicle count data fed into polynomial regression for light usage prediction, with real-time bar charts and anomaly alerts.

![Dashboard](https://img.shields.io/badge/Type-Dashboard-blue)
![Tech](https://img.shields.io/badge/Tech-HTML%20%7C%20CSS%20%7C%20JS-yellow)
![ML](https://img.shields.io/badge/ML-Polynomial%20Regression-green)

---

## 📋 Overview

This project simulates a **smart parking lot lighting management system** that uses IoT sensor data (vehicle counts per zone) and **polynomial regression** to predict optimal lighting levels. The dashboard provides:

- **Real-time bar charts** showing live vehicle counts across 5 parking zones
- **Polynomial regression model** (adjustable degree 1–6) mapping occupancy → light level
- **Anomaly detection** with toast alerts and a scrollable timeline
- **Zone-level monitoring** with capacity bars, occupancy stats, and click-to-select
- **Trend tracking** of the selected zone's vehicle count and predicted lux over time
- **24-hour historical data table** with actual vs. predicted light levels

---

## 🚀 How to Run

No build step or server required — just open the HTML file:

```bash
# Option 1 — double-click
open "Parking Lot Lighting Forecast/index.html"

# Option 2 — VS Code Live Server
# Right-click index.html → "Open with Live Server"

# Option 3 — Python simple server
cd "Parking Lot Lighting Forecast"
python -m http.server 8080
# then visit http://localhost:8080
```

> **Note:** The project uses Chart.js via CDN and Google Fonts, so an internet connection is needed on first load.

---

## 📂 Project Structure

```
Parking Lot Lighting Forecast/
├── index.html          # Main dashboard page
├── style.css           # Premium dark glassmorphism design
├── data.js             # Synthetic sensor dataset generation
├── regression.js       # Pure-JS polynomial regression (OLS)
├── app.js              # Dashboard logic, charts, alerts
└── README.md           # This file
```

---

## 📊 Where to Get Real-World Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **Melbourne On-Street Parking** | [data.melbourne.vic.gov.au](https://data.melbourne.vic.gov.au/) | 5,000+ IoT sensor bays in Melbourne CBD |
| **Birmingham Parking (UCI)** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham) | Multi-car-park occupancy time series |
| **LA Parking Occupancy** | [data.lacity.org](https://data.lacity.org/) | Meter & lot occupancy from Los Angeles |
| **DIY Sensors** | [arduino.cc](https://www.arduino.cc/) | HC-SR04 / IR break-beam → MQTT → InfluxDB |

### How to Use a Real Dataset

1. **Download a CSV** (e.g., Birmingham Parking from UCI — `dataset.csv`)
2. Convert rows to the format: `{ hour, vehicleCount, lightLevel, zone }`
3. Replace the `generateZoneHistory()` function in `data.js` with a CSV parser
4. The regression model in `regression.js` works with any numeric `(x, y)` pairs

---

## 🧮 How the Polynomial Regression Works

The model fits: **y = c₀ + c₁·x + c₂·x² + … + cₙ·xⁿ**

- **x** = occupancy ratio (vehicleCount / zoneCapacity), range [0, 1]
- **y** = optimal light level in lux
- **Degree** is adjustable via the slider (1 = linear, 3 = cubic, etc.)
- Uses OLS via **Gauss-Jordan elimination** on the normal equations (XᵀX)⁻¹Xᵀy
- Displays the **R² score** to evaluate fit quality

---

## ⚠️ Anomaly Detection

An anomaly is flagged when:
- Vehicle count exceeds **135 %** of zone capacity (simulated sensor spike)
- Probability: ~6–8 % per reading cycle (configurable in `data.js`)

When detected:
- Zone card glows **red** with a pulsing border
- A **toast notification** slides in from the right
- Entry added to the **Anomaly Timeline** panel
- The **live bar chart** highlights the anomalous zone in red

---

## 🎨 Design

- **Dark glassmorphism** with backdrop blur and gradient cards
- **Inter** + **JetBrains Mono** typography
- **Smooth micro-animations** (pulse dots, fade-in timeline, badge bounce)
- **Fully responsive** (desktop → tablet → mobile)

---

## 📜 License

MIT — free for academic and commercial use.
