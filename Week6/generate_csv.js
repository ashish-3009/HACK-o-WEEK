// ============================================================
// CSV DATASET GENERATOR — Run this with Node.js to create dataset.csv
// Usage: node generate_csv.js
// ============================================================
// This script generates a realistic 90-day hourly electricity
// dataset for a sports facility. The patterns are deterministic
// with controlled noise so the LSTM can learn them well.
// ============================================================

const fs = require('fs');

// Seed-based pseudo-random for reproducibility
let seed = 42;
function seededRandom() {
  seed = (seed * 16807 + 0) % 2147483647;
  return (seed - 1) / 2147483646;
}

function gaussianNoise(mean, stddev) {
  // Box-Muller transform for gaussian noise
  const u1 = seededRandom();
  const u2 = seededRandom();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * stddev;
}

// -- Hourly base patterns (kWh) for each day type --
// These are carefully designed so the LSTM can learn clear patterns

const PATTERNS = {
  'Event Day': [
    14, 12, 10,  9,  9, 10,    // 0-5 AM: overnight baseline
    16, 20, 25, 30, 35, 38,    // 6-11 AM: pre-event prep
    42, 48, 52, 58, 68, 82,    // 12-5 PM: setup + early arrivals
   105,128,142,148,130, 90,    // 6-11 PM: GAME TIME - floodlights max
  ],
  'Regular Weekday': [
     8,  7,  6,  6,  6,  7,    // 0-5 AM: security lights only
    12, 16, 22, 25, 28, 30,    // 6-11 AM: morning sessions
    32, 34, 36, 34, 30, 26,    // 12-5 PM: afternoon use
    38, 45, 40, 32, 20, 12,    // 6-11 PM: evening training
  ],
  'Weekend': [
    10,  9,  8,  7,  7,  8,    // 0-5 AM: baseline
    11, 14, 20, 28, 35, 40,    // 6-11 AM: rec leagues start
    44, 48, 50, 48, 42, 35,    // 12-5 PM: community events
    52, 62, 55, 44, 28, 16,    // 6-11 PM: evening wrapping up
  ]
};

// Week schedule template — determines day types per day of week
// This is realistic: events typically on Wed/Fri/Sat
function getDayType(dayOfWeek, weekNum) {
  // dayOfWeek: 0=Mon, 1=Tue, ..., 6=Sun
  if (dayOfWeek === 6) return 'Weekend';       // Sunday always weekend
  if (dayOfWeek === 5) {                        // Saturday
    return (weekNum % 3 !== 2) ? 'Event Day' : 'Weekend';  // 2 in 3 weeks = event
  }
  if (dayOfWeek === 4) {                        // Friday
    return (weekNum % 2 === 0) ? 'Event Day' : 'Regular Weekday';  // every other
  }
  if (dayOfWeek === 2) {                        // Wednesday
    return (weekNum % 3 === 0) ? 'Event Day' : 'Regular Weekday';  // every 3rd
  }
  return 'Regular Weekday';                     // Mon, Tue, Thu
}

// Generate the full dataset
function generateDataset() {
  const rows = [];
  const startDate = new Date('2025-12-01');
  const numDays = 90;  // 90 days ~= 13 weeks

  for (let d = 0; d < numDays; d++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + d);

    const dayOfWeek = (date.getDay() + 6) % 7;  // 0=Mon convention
    const weekNum = Math.floor(d / 7);
    const dayType = getDayType(dayOfWeek, weekNum);
    const dateStr = date.toISOString().split('T')[0];  // YYYY-MM-DD

    // seasonal temperature trend (winter → spring)
    const seasonBase = 8 + (d / numDays) * 12;  // 8°C to 20°C over 90 days

    for (let hour = 0; hour < 24; hour++) {
      const baseKwh = PATTERNS[dayType][hour];

      // Add controlled noise — small enough that patterns stay learnable
      // stddev = ~5% of base value for weekdays, ~8% for events
      const noiseStd = dayType === 'Event Day' ? baseKwh * 0.06 : baseKwh * 0.04;
      let kWh = Math.round(gaussianNoise(baseKwh, noiseStd));
      kWh = Math.max(3, kWh);  // min 3 kWh (never zero — always some load)

      // Temperature: diurnal cycle + seasonal + noise
      const diurnal = -4 * Math.cos(2 * Math.PI * (hour - 14) / 24);
      const temp = Math.round(gaussianNoise(seasonBase + diurnal, 1.5) * 10) / 10;

      // Humidity: inverse-ish to temperature
      const humidity = Math.round(Math.min(95, Math.max(20, gaussianNoise(65 - diurnal * 3, 5))));

      // Night flag
      const isNight = (hour >= 18 || hour < 6) ? 1 : 0;

      // Post-event: 10PM-4AM after event days
      const isPostEvent = (dayType === 'Event Day' && (hour >= 22 || hour < 4)) ? 1 : 0;

      // Day of week as number (useful feature for model)
      const dayOfWeekNum = date.getDay();  // 0=Sun, 1=Mon, ... 6=Sat

      rows.push({
        date: dateStr,
        day_of_week: dayOfWeekNum,
        hour: hour,
        day_type: dayType,
        kWh: kWh,
        temperature_c: temp,
        humidity_pct: humidity,
        is_night: isNight,
        is_post_event: isPostEvent,
      });
    }
  }

  return rows;
}

// Write to CSV
function writeCSV(rows, filename) {
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(',')];

  for (const row of rows) {
    const values = headers.map(h => {
      const v = row[h];
      // quote strings that contain commas or spaces
      if (typeof v === 'string' && (v.includes(',') || v.includes(' '))) {
        return `"${v}"`;
      }
      return v;
    });
    lines.push(values.join(','));
  }

  fs.writeFileSync(filename, lines.join('\n'), 'utf-8');
  console.log(`✅ Written ${rows.length} rows to ${filename}`);

  // Print some stats
  const dayTypes = {};
  rows.forEach(r => {
    dayTypes[r.day_type] = (dayTypes[r.day_type] || 0) + 1;
  });
  console.log('Day type breakdown (hours):');
  Object.entries(dayTypes).forEach(([dt, count]) => {
    console.log(`  ${dt}: ${count} hours (${count/24} days)`);
  });

  const kWhValues = rows.map(r => r.kWh);
  console.log(`kWh range: ${Math.min(...kWhValues)} - ${Math.max(...kWhValues)}`);
  console.log(`kWh mean: ${(kWhValues.reduce((a,b) => a+b, 0) / kWhValues.length).toFixed(1)}`);
}

// Run
const dataset = generateDataset();
writeCSV(dataset, __dirname + '/dataset.csv');
console.log('\nDone! The CSV is ready for the dashboard.');
