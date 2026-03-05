// ============================================================
// DATA MODULE — CSV Loader & Preprocessing
// ============================================================
// This module loads the dataset.csv file and parses it into
// JavaScript objects for the LSTM model and dashboard charts.
//
// The CSV has 2160 rows (90 days × 24 hours) with columns:
//   date, day_of_week, hour, day_type, kWh, temperature_c,
//   humidity_pct, is_night, is_post_event
//
// We parse it manually (no PapaParse needed) since the format
// is simple and predictable.
// ============================================================

// day type constants
const DAY_TYPES = ['Event Day', 'Regular Weekday', 'Weekend'];

/**
 * Load and parse the CSV file.
 * Returns a promise that resolves to an array of data objects.
 */
async function loadCSVData() {
  try {
    const response = await fetch('dataset.csv');
    if (!response.ok) throw new Error('Failed to load dataset.csv');

    const text = await response.text();
    const lines = text.trim().split('\n');

    // first line is the header
    const headers = lines[0].split(',');
    console.log('CSV headers:', headers);

    const data = [];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      // parse CSV line — handle quoted fields (day_type has spaces)
      const values = parseCSVLine(line);

      // map to object
      const row = {
        date: values[0],
        dayOfWeek: parseInt(values[1]),
        hour: parseInt(values[2]),
        dayType: values[3].replace(/"/g, ''),  // remove quotes
        kWh: parseInt(values[4]),
        tempC: parseFloat(values[5]),
        humidity: parseInt(values[6]),
        isNight: parseInt(values[7]) === 1,
        isPostEvent: parseInt(values[8]) === 1,
      };

      data.push(row);
    }

    console.log(`Loaded ${data.length} records from CSV`);
    return data;

  } catch (err) {
    console.error('Error loading CSV:', err);
    throw err;
  }
}

/**
 * Simple CSV line parser that handles quoted fields.
 * We need this because day_type values like "Event Day" have quotes.
 */
function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current.trim());  // last field
  return result;
}

/**
 * Filter data to only night hours (6 PM – 6 AM).
 */
function filterNightHours(data) {
  return data.filter(d => d.isNight);
}

/**
 * Filter to post-event hours.
 */
function filterPostEvent(data) {
  return data.filter(d => d.isPostEvent);
}

/**
 * Calculate basic statistics for a data array.
 * Used for the dashboard summary cards.
 */
function calcStats(data) {
  if (data.length === 0) return { mean: 0, max: 0, min: 0, total: 0, count: 0 };

  const values = data.map(d => d.kWh);
  const sum = values.reduce((a, b) => a + b, 0);

  return {
    mean: Math.round(sum / values.length * 10) / 10,
    max: Math.max(...values),
    min: Math.min(...values),
    total: Math.round(sum),
    count: values.length,
  };
}

/**
 * Group data by day type and return stats for each.
 */
function getStatsByDayType(data) {
  const grouped = {};
  DAY_TYPES.forEach(dt => {
    const filtered = data.filter(d => d.dayType === dt);
    grouped[dt] = {
      stats: calcStats(filtered),
      nightStats: calcStats(filterNightHours(filtered)),
      data: filtered,
    };
  });
  return grouped;
}

/**
 * Get the unique dates in the dataset (for train/test splitting).
 */
function getUniqueDates(data) {
  return [...new Set(data.map(d => d.date))];
}
