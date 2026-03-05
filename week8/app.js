// ============================================================
// APP.JS — Main Application Logic
// ============================================================
// Wires up the dashboard: trains regression model, renders
// Chart.js charts, streams live sensor data, and fires
// anomaly alerts.
// ============================================================

// ---- Global State ----
let regressionModel = null;
let liveChart = null;
let regressionChart = null;
let trendChart = null;
let anomalyLog = [];
let liveInterval = null;
let currentZone = 'Zone A';
let tickCount = 0;

// ---- DOM Refs ----
const dom = {};
function cacheDom() {
  dom.zoneSelect       = document.getElementById('zone-select');
  dom.liveCanvas       = document.getElementById('live-chart');
  dom.regCanvas        = document.getElementById('regression-chart');
  dom.trendCanvas      = document.getElementById('trend-chart');
  dom.alertContainer   = document.getElementById('alert-container');
  dom.statVehicles     = document.getElementById('stat-vehicles');
  dom.statLux          = document.getElementById('stat-lux');
  dom.statOccupancy    = document.getElementById('stat-occupancy');
  dom.statAnomalies    = document.getElementById('stat-anomalies');
  dom.regressionEq     = document.getElementById('regression-eq');
  dom.regressionR2     = document.getElementById('regression-r2');
  dom.degreeSlider     = document.getElementById('degree-slider');
  dom.degreeValue      = document.getElementById('degree-value');
  dom.simulateBtn      = document.getElementById('simulate-btn');
  dom.resetBtn         = document.getElementById('reset-btn');
  dom.dataTable        = document.getElementById('data-table-body');
  dom.clock            = document.getElementById('clock');
  dom.zoneCards        = document.getElementById('zone-cards');
  dom.alertBadge       = document.getElementById('alert-badge');
  dom.anomalyTimeline  = document.getElementById('anomaly-timeline');
}

// ---- Helpers ----
function formatTime(d) {
  return d.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatHour(h) {
  const ampm = h >= 12 ? 'PM' : 'AM';
  const hr = h % 12 || 12;
  return `${hr} ${ampm}`;
}

// ---- Clock ----
function tickClock() {
  if (dom.clock) dom.clock.textContent = formatTime(new Date());
}

// ---- Train Regression Model ----
function trainModel(degree) {
  const history = generateZoneHistory(currentZone);
  const cap = ZONE_CAPACITY[currentZone];

  const xs = history.map(d => d.vehicleCount / cap);   // occupancy 0-1
  const ys = history.map(d => d.lightLevel);            // lux

  regressionModel = new PolynomialRegression(degree);
  regressionModel.fit(xs, ys);

  const r2 = regressionModel.r2(xs, ys);
  dom.regressionEq.textContent = regressionModel.equationString();
  dom.regressionR2.textContent = r2.toFixed(4);

  renderRegressionChart(xs, ys);
  renderDataTable(history);
}

// ---- Charts ----

// 1. Live bar chart — vehicle counts per zone, refreshed every 2 s
function initLiveChart() {
  const ctx = dom.liveCanvas.getContext('2d');
  liveChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ZONES,
      datasets: [{
        label: 'Vehicle Count',
        data: ZONES.map(() => 0),
        backgroundColor: ZONES.map((_, i) => `hsla(${200 + i * 32}, 85%, 60%, 0.75)`),
        borderColor: ZONES.map((_, i) => `hsla(${200 + i * 32}, 85%, 45%, 1)`),
        borderWidth: 2,
        borderRadius: 8,
      }, {
        label: 'Predicted Light (lux)',
        data: ZONES.map(() => 0),
        backgroundColor: 'hsla(45, 100%, 60%, 0.55)',
        borderColor: 'hsla(45, 100%, 45%, 1)',
        borderWidth: 2,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#c9d1d9', font: { family: 'Inter', size: 12 } } },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.92)',
          titleColor: '#58a6ff',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          cornerRadius: 8,
        },
      },
      scales: {
        x: { ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.4)' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.4)' }, beginAtZero: true },
      },
      animation: { duration: 600, easing: 'easeInOutCubic' },
    },
  });
}

// 2. Regression scatter + fitted curve
function renderRegressionChart(xs, ys) {
  if (regressionChart) regressionChart.destroy();
  const ctx = dom.regCanvas.getContext('2d');

  // Generate smooth curve
  const curveXs = [];
  for (let x = 0; x <= 1.15; x += 0.01) curveXs.push(x);
  const curveYs = regressionModel.predictMany(curveXs);

  regressionChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Observed',
          data: xs.map((x, i) => ({ x, y: ys[i] })),
          backgroundColor: 'hsla(210, 100%, 65%, 0.7)',
          borderColor: 'hsla(210, 100%, 50%, 1)',
          pointRadius: 6,
          pointHoverRadius: 9,
        },
        {
          label: `Poly Degree ${regressionModel.degree}`,
          data: curveXs.map((x, i) => ({ x, y: curveYs[i] })),
          type: 'line',
          borderColor: 'hsla(330, 100%, 60%, 1)',
          borderWidth: 3,
          pointRadius: 0,
          fill: false,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#c9d1d9', font: { family: 'Inter' } } },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.92)',
          titleColor: '#58a6ff',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          cornerRadius: 8,
        },
      },
      scales: {
        x: { title: { display: true, text: 'Occupancy %', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.4)' }, min: 0, max: 1.2 },
        y: { title: { display: true, text: 'Light Level (lux)', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.4)' }, beginAtZero: true },
      },
    },
  });
}

// 3. Trend line chart — last 20 ticks of selected zone
const trendData = { labels: [], vehicles: [], lux: [] };
function initTrendChart() {
  const ctx = dom.trendCanvas.getContext('2d');
  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: trendData.labels,
      datasets: [
        {
          label: 'Vehicles',
          data: trendData.vehicles,
          borderColor: 'hsla(170, 80%, 55%, 1)',
          backgroundColor: 'hsla(170, 80%, 55%, 0.15)',
          fill: true,
          tension: 0.45,
          pointRadius: 3,
          pointBackgroundColor: 'hsla(170, 80%, 55%, 1)',
        },
        {
          label: 'Predicted Lux',
          data: trendData.lux,
          borderColor: 'hsla(45, 100%, 55%, 1)',
          backgroundColor: 'hsla(45, 100%, 55%, 0.10)',
          fill: true,
          tension: 0.45,
          pointRadius: 3,
          pointBackgroundColor: 'hsla(45, 100%, 55%, 1)',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#c9d1d9', font: { family: 'Inter' } } },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.92)',
          titleColor: '#58a6ff',
          bodyColor: '#c9d1d9',
          borderColor: '#30363d',
          borderWidth: 1,
          cornerRadius: 8,
        },
      },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: 'rgba(48,54,61,0.4)' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.4)' }, beginAtZero: true },
      },
      animation: { duration: 400 },
    },
  });
}

// ---- Zone Cards ----
function renderZoneCards(readings) {
  dom.zoneCards.innerHTML = '';
  ZONES.forEach(zone => {
    const r = readings[zone];
    const occ = ((r.vehicleCount / r.capacity) * 100).toFixed(1);
    const cls = r.isAnomaly ? 'zone-card anomaly-glow' : 'zone-card';
    const selected = zone === currentZone ? ' selected' : '';
    dom.zoneCards.innerHTML += `
      <div class="${cls}${selected}" data-zone="${zone}" onclick="switchZone('${zone}')">
        <div class="zone-card-header">
          <span class="zone-name">${zone}</span>
          ${r.isAnomaly ? '<span class="anomaly-badge">⚠ ANOMALY</span>' : ''}
        </div>
        <div class="zone-stat-grid">
          <div><span class="zone-stat-value">${r.vehicleCount}</span><span class="zone-stat-label">Vehicles</span></div>
          <div><span class="zone-stat-value">${occ}%</span><span class="zone-stat-label">Occupancy</span></div>
          <div><span class="zone-stat-value">${r.capacity}</span><span class="zone-stat-label">Capacity</span></div>
        </div>
        <div class="occ-bar"><div class="occ-fill" style="width:${Math.min(100, occ)}%; background:${occ > 90 ? '#f85149' : occ > 70 ? '#d29922' : '#3fb950'}"></div></div>
      </div>`;
  });
}

// ---- Anomaly Alerts ----
function fireAlert(zone, reading) {
  anomalyLog.unshift({ zone, time: new Date(), count: reading.vehicleCount, capacity: reading.capacity });
  if (anomalyLog.length > 50) anomalyLog.pop();

  dom.alertBadge.textContent = anomalyLog.length;
  dom.alertBadge.style.display = 'inline-flex';

  // Toast
  const toast = document.createElement('div');
  toast.className = 'alert-toast';
  toast.innerHTML = `
    <div class="alert-icon">⚠️</div>
    <div class="alert-body">
      <strong>${zone} — Anomaly Detected</strong>
      <span>${reading.vehicleCount} vehicles (cap ${reading.capacity}) at ${formatTime(new Date())}</span>
    </div>
    <button class="alert-close" onclick="this.parentElement.remove()">✕</button>`;
  dom.alertContainer.prepend(toast);
  setTimeout(() => toast.classList.add('show'), 30);
  setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 400); }, 6000);

  renderAnomalyTimeline();
}

function renderAnomalyTimeline() {
  dom.anomalyTimeline.innerHTML = anomalyLog.slice(0, 15).map(a => `
    <div class="timeline-item">
      <span class="tl-dot"></span>
      <span class="tl-zone">${a.zone}</span>
      <span class="tl-detail">${a.count}/${a.capacity} vehicles</span>
      <span class="tl-time">${formatTime(a.time)}</span>
    </div>`).join('');
}

// ---- Data Table ----
function renderDataTable(history) {
  dom.dataTable.innerHTML = history.map(d => {
    const occ = (d.vehicleCount / ZONE_CAPACITY[currentZone] * 100).toFixed(1);
    const predicted = regressionModel ? Math.round(regressionModel.predict(d.vehicleCount / ZONE_CAPACITY[currentZone])) : '—';
    return `<tr class="${d.isAnomaly ? 'row-anomaly' : ''}">
      <td>${formatHour(d.hour)}</td>
      <td>${d.vehicleCount}</td>
      <td>${occ}%</td>
      <td>${d.lightLevel}</td>
      <td>${predicted}</td>
      <td>${d.isAnomaly ? '<span class="badge-anomaly">ANOMALY</span>' : '<span class="badge-normal">Normal</span>'}</td>
    </tr>`;
  }).join('');
}

// ---- Live Update Loop ----
function liveUpdate() {
  tickCount++;
  const readings = generateLiveReading();
  const r = readings[currentZone];

  // Stats
  const occ = ((r.vehicleCount / r.capacity) * 100).toFixed(1);
  const predictedLux = regressionModel ? Math.round(regressionModel.predict(r.vehicleCount / r.capacity)) : 0;

  dom.statVehicles.textContent = r.vehicleCount;
  dom.statLux.textContent = predictedLux + ' lux';
  dom.statOccupancy.textContent = occ + '%';
  dom.statAnomalies.textContent = anomalyLog.length;

  // Update live bar chart
  liveChart.data.datasets[0].data = ZONES.map(z => readings[z].vehicleCount);
  liveChart.data.datasets[1].data = ZONES.map(z => {
    if (!regressionModel) return 0;
    return Math.round(regressionModel.predict(readings[z].vehicleCount / ZONE_CAPACITY[z]));
  });

  // Highlight anomaly bars
  liveChart.data.datasets[0].backgroundColor = ZONES.map((z, i) =>
    readings[z].isAnomaly ? 'hsla(0, 85%, 55%, 0.8)' : `hsla(${200 + i * 32}, 85%, 60%, 0.75)`
  );
  liveChart.update();

  // Trend
  trendData.labels.push(formatTime(new Date()));
  trendData.vehicles.push(r.vehicleCount);
  trendData.lux.push(predictedLux);
  if (trendData.labels.length > 25) {
    trendData.labels.shift(); trendData.vehicles.shift(); trendData.lux.shift();
  }
  trendChart.update();

  // Zone cards
  renderZoneCards(readings);

  // Anomaly alerts
  ZONES.forEach(z => {
    if (readings[z].isAnomaly) fireAlert(z, readings[z]);
  });
}

// ---- Zone Switch ----
function switchZone(zone) {
  currentZone = zone;
  dom.zoneSelect.value = zone;
  trendData.labels.length = 0; trendData.vehicles.length = 0; trendData.lux.length = 0;
  trainModel(parseInt(dom.degreeSlider.value));
}

// ---- Init ----
function init() {
  cacheDom();
  tickClock();
  setInterval(tickClock, 1000);

  // Zone select
  dom.zoneSelect.addEventListener('change', e => switchZone(e.target.value));

  // Degree slider
  dom.degreeSlider.addEventListener('input', e => {
    dom.degreeValue.textContent = e.target.value;
    trainModel(parseInt(e.target.value));
  });

  // Simulate / Reset
  dom.simulateBtn.addEventListener('click', () => {
    if (liveInterval) return;
    liveInterval = setInterval(liveUpdate, 2000);
    dom.simulateBtn.classList.add('active');
    dom.simulateBtn.innerHTML = '<span class="pulse-dot"></span> Streaming';
  });

  dom.resetBtn.addEventListener('click', () => {
    clearInterval(liveInterval);
    liveInterval = null;
    anomalyLog = [];
    trendData.labels.length = 0; trendData.vehicles.length = 0; trendData.lux.length = 0;
    trendChart.update();
    dom.alertBadge.style.display = 'none';
    dom.anomalyTimeline.innerHTML = '';
    dom.simulateBtn.classList.remove('active');
    dom.simulateBtn.innerHTML = '▶ Start Live Feed';
    liveUpdate();
  });

  // Charts
  initLiveChart();
  initTrendChart();

  // First model train
  trainModel(parseInt(dom.degreeSlider.value));

  // Kick off live
  liveUpdate();
  liveInterval = setInterval(liveUpdate, 2000);
  dom.simulateBtn.classList.add('active');
  dom.simulateBtn.innerHTML = '<span class="pulse-dot"></span> Streaming';
}

document.addEventListener('DOMContentLoaded', init);
