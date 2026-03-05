// ============================================================
// DATA MODULE — Sensor-Based Vehicle Count Dataset
// ============================================================
// Dataset: Simulated IoT parking-lot sensors.
// Each zone has IR / ultrasonic counters that report every 30 s.
// We aggregate to 1-hour buckets for the regression model.
//
// HOW TO GET A REAL DATASET:
//   • City of Melbourne On-Street Parking Sensor Data
//     https://data.melbourne.vic.gov.au/
//   • Birmingham Parking Dataset (UCI ML Repo)
//     https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham
//   • Los Angeles Parking Occupancy (LA Open Data)
//     https://data.lacity.org/
//   • Or deploy your own sensors (HC-SR04, IR break-beam)
//     and push MQTT → InfluxDB → CSV export.
// ============================================================

const ZONES = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E'];

const ZONE_CAPACITY = {
  'Zone A': 120,
  'Zone B': 90,
  'Zone C': 150,
  'Zone D': 60,
  'Zone E': 100,
};

// Typical daily traffic pattern (multiplier per hour 0-23)
const TRAFFIC_PATTERN = [
  0.05, 0.03, 0.02, 0.02, 0.03, 0.08,   // 0-5   (night)
  0.15, 0.35, 0.65, 0.80, 0.85, 0.75,   // 6-11  (morning rush)
  0.70, 0.72, 0.78, 0.82, 0.88, 0.90,   // 12-17 (afternoon)
  0.75, 0.55, 0.40, 0.25, 0.15, 0.08,   // 18-23 (evening)
];

// Light-level lookup (lux) for a zone at full capacity
const MAX_LUX = {
  'Zone A': 300,
  'Zone B': 250,
  'Zone C': 350,
  'Zone D': 200,
  'Zone E': 280,
};

/**
 * Generate a full 24-hour historical dataset for a given zone.
 * Returns an array of objects: { hour, vehicleCount, lightLevel, isAnomaly }
 */
function generateZoneHistory(zone) {
  const cap = ZONE_CAPACITY[zone];
  const maxLux = MAX_LUX[zone];
  const data = [];

  for (let h = 0; h < 24; h++) {
    const base = Math.round(cap * TRAFFIC_PATTERN[h]);
    const noise = Math.round((Math.random() - 0.5) * cap * 0.08);
    let count = Math.max(0, Math.min(cap, base + noise));

    // Inject anomaly ~8 % of the time
    let isAnomaly = false;
    if (Math.random() < 0.08) {
      count = Math.round(count * (1.4 + Math.random() * 0.4));
      count = Math.min(cap + 30, count); // can exceed capacity (overflow)
      isAnomaly = true;
    }

    // Ideal light level is a polynomial function of occupancy %
    const occ = count / cap;
    const idealLux = Math.round(maxLux * (0.15 + 0.85 * occ));

    data.push({ hour: h, vehicleCount: count, lightLevel: idealLux, isAnomaly });
  }
  return data;
}

/**
 * Generate current "live" sensor reading for all zones.
 */
function generateLiveReading() {
  const now = new Date();
  const hour = now.getHours();
  const readings = {};

  ZONES.forEach(zone => {
    const cap = ZONE_CAPACITY[zone];
    const base = Math.round(cap * TRAFFIC_PATTERN[hour]);
    const noise = Math.round((Math.random() - 0.5) * cap * 0.12);
    let count = Math.max(0, Math.min(cap, base + noise));

    let isAnomaly = false;
    if (Math.random() < 0.06) {
      count = Math.round(count * (1.35 + Math.random() * 0.5));
      count = Math.min(cap + 30, count);
      isAnomaly = true;
    }

    readings[zone] = { vehicleCount: count, capacity: cap, isAnomaly };
  });

  return readings;
}

/**
 * Generate a week's worth of hourly data (for model training).
 * Returns flat array: { day, hour, zone, vehicleCount, lightLevel }
 */
function generateWeeklyDataset() {
  const rows = [];
  for (let d = 0; d < 7; d++) {
    ZONES.forEach(zone => {
      const history = generateZoneHistory(zone);
      history.forEach(h => {
        rows.push({
          day: d,
          hour: h.hour,
          zone,
          vehicleCount: h.vehicleCount,
          lightLevel: h.lightLevel,
          isAnomaly: h.isAnomaly,
        });
      });
    });
  }
  return rows;
}
