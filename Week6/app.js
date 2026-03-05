// ============================================================
// APP MODULE — Sports Facility Night Usage Dashboard
// ============================================================
// Main application logic. Ties together CSV data loading (data.js)
// and the TensorFlow.js LSTM model (model.js), handles chart
// rendering and all UI interactions.
//
// Libraries:
//   - Chart.js (CDN) — for all visualizations
//   - TensorFlow.js (CDN) — for the LSTM model
//
// Dashboard features:
//   1. Load real CSV dataset (2160 rows, 90 days)
//   2. Filter by day type (Event/Weekday/Weekend/All)
//   3. Train LSTM with progress tracking
//   4. Show actual vs predicted with accuracy metrics
//   5. 12-hour post-event forecast
//   6. Training loss + validation loss curves
// ============================================================

// -- Global application state --
let fullDataset = [];      // parsed CSV data
let currentFilter = 'All'; // selected day type filter
let chartInstances = {};   // chart.js instances (for destroying/updating)
let modelResults = null;   // LSTM results after training
let isTraining = false;    // prevent double-clicking train

// hour labels for the x-axis (0-23 → 12AM, 1AM, ..., 11PM)
const HOUR_LABELS = Array.from({ length: 24 }, (_, i) => {
  const ampm = i < 12 ? 'AM' : 'PM';
  const h = i === 0 ? 12 : (i > 12 ? i - 12 : i);
  return `${h}${ampm}`;
});

// =================== INITIALIZATION ===================

document.addEventListener('DOMContentLoaded', async function () {
  console.log("Dashboard initializing...");

  // show loading while CSV loads
  showLoading('Loading dataset.csv...');

  try {
    // load the CSV file
    fullDataset = await loadCSVData();
    addLog(`Loaded ${fullDataset.length} records from dataset.csv`, 'ok');

    // count day types for the log
    const counts = {};
    fullDataset.forEach(d => counts[d.dayType] = (counts[d.dayType] || 0) + 1);
    Object.entries(counts).forEach(([dt, c]) => {
      addLog(`  ${dt}: ${c} hours (${Math.round(c/24)} days)`, 'info');
    });

  } catch (err) {
    console.error('Failed to load CSV:', err);
    addLog('ERROR: Could not load dataset.csv — ' + err.message, 'warn');
    hideLoading();
    return;
  }

  hideLoading();

  // setup clock
  updateClock();
  setInterval(updateClock, 1000);

  // attach event handlers
  setupEventListeners();

  // render the initial dashboard
  renderDashboard();

  console.log("Dashboard ready!");
  addLog('Dashboard ready — select a day type and train the model', 'ok');
});

// =================== CLOCK ===================

function updateClock() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const el = document.getElementById('clock');
  if (el) el.textContent = timeStr;
}

// =================== EVENT LISTENERS ===================

function setupEventListeners() {
  // day type filter dropdown
  const dayTypeSelect = document.getElementById('daytype-select');
  if (dayTypeSelect) {
    dayTypeSelect.addEventListener('change', function () {
      currentFilter = this.value;
      renderDashboard();
      if (modelResults) {
        addLog('Filter changed — click "Retrain" for updated predictions', 'warn');
      }
    });
  }

  // train model button
  const trainBtn = document.getElementById('train-btn');
  if (trainBtn) {
    trainBtn.addEventListener('click', startTraining);
  }

  // reset button
  const resetBtn = document.getElementById('reset-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', resetDashboard);
  }
}

// =================== RENDER DASHBOARD ===================

function renderDashboard() {
  const data = getFilteredData();

  updateStats(data);
  renderHourlyChart(data);
  renderDayTypeChart();
  renderDataTable(data);

  // if model already trained, refresh those charts too
  if (modelResults) {
    renderPredictionChart();
    renderLossChart();
    renderForecastChart();
    updateModelMetrics();
  }
}

function getFilteredData() {
  if (currentFilter === 'All') return fullDataset;
  return fullDataset.filter(d => d.dayType === currentFilter);
}

// =================== STAT CARDS ===================

function updateStats(data) {
  const stats = calcStats(data);
  const nightData = filterNightHours(data);
  const nightStats = calcStats(nightData);

  document.getElementById('stat-records').textContent = data.length.toLocaleString();
  document.getElementById('stat-avg-kwh').textContent = stats.mean + ' kWh';
  document.getElementById('stat-peak').textContent = nightStats.max + ' kWh';

  const nightPct = stats.total > 0
    ? Math.round(nightStats.total / stats.total * 100)
    : 0;
  document.getElementById('stat-night-pct').textContent = nightPct + '%';
}

// =================== HOURLY PATTERN CHART ===================

function renderHourlyChart(data) {
  const ctx = document.getElementById('hourly-chart');
  if (!ctx) return;

  // compute average, max, min kWh per hour
  const hourlyAvg = new Array(24).fill(0);
  const hourlyCounts = new Array(24).fill(0);
  const hourlyMax = new Array(24).fill(0);
  const hourlyMin = new Array(24).fill(Infinity);

  data.forEach(d => {
    hourlyAvg[d.hour] += d.kWh;
    hourlyCounts[d.hour]++;
    if (d.kWh > hourlyMax[d.hour]) hourlyMax[d.hour] = d.kWh;
    if (d.kWh < hourlyMin[d.hour]) hourlyMin[d.hour] = d.kWh;
  });

  for (let i = 0; i < 24; i++) {
    hourlyAvg[i] = hourlyCounts[i] > 0 ? Math.round(hourlyAvg[i] / hourlyCounts[i]) : 0;
    if (hourlyMin[i] === Infinity) hourlyMin[i] = 0;
  }

  // color night hours green, day hours blue
  const barColors = HOUR_LABELS.map((_, i) =>
    (i >= 18 || i < 6) ? 'rgba(52, 211, 153, 0.7)' : 'rgba(96, 165, 250, 0.5)'
  );
  const borderColors = HOUR_LABELS.map((_, i) =>
    (i >= 18 || i < 6) ? 'rgba(52, 211, 153, 1)' : 'rgba(96, 165, 250, 0.8)'
  );

  if (chartInstances.hourly) chartInstances.hourly.destroy();

  chartInstances.hourly = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: HOUR_LABELS,
      datasets: [
        {
          label: 'Avg kWh',
          data: hourlyAvg,
          backgroundColor: barColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4,
          order: 2,
        },
        {
          label: 'Peak kWh',
          data: hourlyMax,
          type: 'line',
          borderColor: 'rgba(248, 113, 113, 0.6)',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          borderDash: [5, 3],
          order: 1,
        }
      ]
    },
    options: getChartOptions('Electricity (kWh)'),
  });
}

// =================== DAY TYPE COMPARISON ===================

function renderDayTypeChart() {
  const ctx = document.getElementById('daytype-chart');
  if (!ctx) return;

  const dayTypeData = {};
  DAY_TYPES.forEach(dt => {
    const filtered = fullDataset.filter(d => d.dayType === dt);
    const hourlyAvg = new Array(24).fill(0);
    const hourlyCounts = new Array(24).fill(0);

    filtered.forEach(d => {
      hourlyAvg[d.hour] += d.kWh;
      hourlyCounts[d.hour]++;
    });

    for (let i = 0; i < 24; i++) {
      hourlyAvg[i] = hourlyCounts[i] > 0 ? Math.round(hourlyAvg[i] / hourlyCounts[i]) : 0;
    }
    dayTypeData[dt] = hourlyAvg;
  });

  if (chartInstances.dayType) chartInstances.dayType.destroy();

  const colors = {
    'Event Day':        { line: '#fbbf24', bg: 'rgba(251, 191, 36, 0.1)' },
    'Regular Weekday':  { line: '#60a5fa', bg: 'rgba(96, 165, 250, 0.1)' },
    'Weekend':          { line: '#a78bfa', bg: 'rgba(167, 139, 250, 0.1)' },
  };

  chartInstances.dayType = new Chart(ctx, {
    type: 'line',
    data: {
      labels: HOUR_LABELS,
      datasets: DAY_TYPES.map(dt => ({
        label: dt,
        data: dayTypeData[dt],
        borderColor: colors[dt].line,
        backgroundColor: colors[dt].bg,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        fill: true,
        tension: 0.4,
      })),
    },
    options: {
      ...getChartOptions('Avg Electricity (kWh)'),
      interaction: { mode: 'index', intersect: false },
    },
  });
}

// =================== LSTM TRAINING ===================

async function startTraining() {
  if (isTraining) return;
  isTraining = true;

  const trainBtn = document.getElementById('train-btn');
  trainBtn.textContent = '⏳ Training LSTM...';
  trainBtn.classList.add('active');

  // show progress bar
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  progressContainer.classList.add('active');
  progressFill.style.width = '0%';

  addLog('Starting TensorFlow.js LSTM training...', 'info');
  addLog(`Filter: ${currentFilter} | Data: ${getFilteredData().length} records`, 'info');
  addLog(`Config: window=${MODEL_CONFIG.windowSize}, lstm=${MODEL_CONFIG.lstmUnits}, epochs=${MODEL_CONFIG.epochs}`, 'info');

  try {
    modelResults = await trainAndPredict(
      fullDataset,
      currentFilter,
      (epoch, total, loss) => {
        const pct = Math.round((epoch / total) * 100);
        progressFill.style.width = pct + '%';
        progressText.textContent = `Epoch ${epoch}/${total} — Loss: ${loss.toFixed(2)}`;
      }
    );

    // log all the metrics
    const m = modelResults.metrics;
    addLog(`✅ Training complete!`, 'ok');
    addLog(`   Accuracy: ${m.accuracy}%`, 'ok');
    addLog(`   R² Score: ${m.r2}`, 'ok');
    addLog(`   RMSE: ${m.rmse} kWh`, 'info');
    addLog(`   MAE: ${m.mae} kWh`, 'info');
    addLog(`   MAPE: ${m.mape}%`, 'info');
    addLog(`   Train: ${modelResults.dataInfo.trainSize} | Test: ${modelResults.dataInfo.testSize}`, 'info');

    // render model charts
    renderPredictionChart();
    renderLossChart();
    renderForecastChart();
    updateModelMetrics();

  } catch (err) {
    console.error('Training failed:', err);
    addLog('❌ Training failed: ' + err.message, 'warn');
  }

  trainBtn.textContent = '🧠 Retrain Model';
  trainBtn.classList.remove('active');
  isTraining = false;

  progressFill.style.width = '100%';
  progressText.textContent = 'Training complete!';
}

// =================== PREDICTION CHART ===================

function renderPredictionChart() {
  if (!modelResults) return;
  const ctx = document.getElementById('prediction-chart');
  if (!ctx) return;

  if (chartInstances.prediction) chartInstances.prediction.destroy();

  // show the last 100 points for clarity (or all if fewer)
  const maxPoints = Math.min(100, modelResults.actuals.length);
  const startIdx = modelResults.actuals.length - maxPoints;
  const actuals = modelResults.actuals.slice(startIdx);
  const predictions = modelResults.predictions.slice(startIdx);
  const labels = actuals.map((_, i) => `T+${startIdx + i + 1}`);

  chartInstances.prediction = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Actual (kWh)',
          data: actuals,
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.08)',
          borderWidth: 2,
          pointRadius: 1.5,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'LSTM Predicted',
          data: predictions,
          borderColor: '#fb923c',
          backgroundColor: 'rgba(251, 146, 60, 0.08)',
          borderWidth: 2,
          pointRadius: 1.5,
          borderDash: [5, 3],
          fill: true,
          tension: 0.3,
        }
      ]
    },
    options: {
      ...getChartOptions('kWh'),
      interaction: { mode: 'index', intersect: false },
      plugins: {
        ...getChartOptions('kWh').plugins,
        legend: {
          labels: { color: '#94a3b8', font: { size: 11 }, usePointStyle: true }
        },
      },
    },
  });
}

// =================== LOSS CHART ===================

function renderLossChart() {
  if (!modelResults) return;
  const ctx = document.getElementById('loss-chart');
  if (!ctx) return;

  if (chartInstances.loss) chartInstances.loss.destroy();

  chartInstances.loss = new Chart(ctx, {
    type: 'line',
    data: {
      labels: modelResults.losses.map((_, i) => i + 1),
      datasets: [
        {
          label: 'Training Loss',
          data: modelResults.losses,
          borderColor: '#f87171',
          backgroundColor: 'rgba(248, 113, 113, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Validation Loss',
          data: modelResults.valLosses,
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251, 191, 36, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
          borderDash: [4, 3],
        }
      ]
    },
    options: {
      ...getChartOptions('Normalized MSE'),
      plugins: {
        ...getChartOptions('Normalized MSE').plugins,
        legend: {
          labels: { color: '#94a3b8', font: { size: 11 }, usePointStyle: true }
        },
      },
      scales: {
        ...getChartOptions('Normalized MSE').scales,
        x: {
          ...getChartOptions('Normalized MSE').scales.x,
          title: { display: true, text: 'Epoch', color: '#475569', font: { size: 11 } }
        }
      }
    },
  });
}

// =================== 12-HOUR FORECAST ===================

function renderForecastChart() {
  if (!modelResults) return;
  const ctx = document.getElementById('forecast-chart');
  if (!ctx) return;

  if (chartInstances.forecast) chartInstances.forecast.destroy();

  // label forecast hours starting from current time
  const now = new Date();
  const forecastLabels = modelResults.forecast.map((_, i) => {
    const h = (now.getHours() + i + 1) % 24;
    const ampm = h < 12 ? 'AM' : 'PM';
    const hr = h === 0 ? 12 : (h > 12 ? h - 12 : h);
    return `${hr}:00 ${ampm}`;
  });

  // color night vs day bars
  const bgColors = modelResults.forecast.map((_, i) => {
    const h = (now.getHours() + i + 1) % 24;
    return (h >= 18 || h < 6)
      ? 'rgba(52, 211, 153, 0.65)' : 'rgba(96, 165, 250, 0.45)';
  });

  chartInstances.forecast = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: forecastLabels,
      datasets: [{
        label: 'Predicted kWh (Post-Event)',
        data: modelResults.forecast,
        backgroundColor: bgColors,
        borderColor: bgColors.map(c =>
          c.includes('211, 153') ? 'rgba(52, 211, 153, 1)' : 'rgba(96, 165, 250, 0.85)'
        ),
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      ...getChartOptions('Predicted kWh'),
      scales: {
        ...getChartOptions('Predicted kWh').scales,
        y: {
          ...getChartOptions('Predicted kWh').scales.y,
          beginAtZero: true,
        }
      },
      plugins: {
        ...getChartOptions('Predicted kWh').plugins,
        tooltip: {
          ...getChartOptions('Predicted kWh').plugins.tooltip,
          callbacks: {
            afterBody: function (ctx) {
              const h = (now.getHours() + ctx[0].dataIndex + 1) % 24;
              return (h >= 18 || h < 6) ? '🌙 Night Hour' : '☀️ Day Hour';
            }
          }
        }
      }
    },
  });
}

// =================== MODEL METRICS ===================

function updateModelMetrics() {
  if (!modelResults) return;

  const m = modelResults.metrics;
  document.getElementById('metric-accuracy').textContent = m.accuracy + '%';
  document.getElementById('metric-r2').textContent = m.r2;
  document.getElementById('metric-rmse').textContent = m.rmse + ' kWh';
  document.getElementById('metric-mae').textContent = m.mae + ' kWh';
  document.getElementById('metric-mape').textContent = m.mape + '%';
  document.getElementById('metric-train').textContent = modelResults.dataInfo.trainSize;
  document.getElementById('metric-test').textContent = modelResults.dataInfo.testSize;
}

// =================== DATA TABLE ===================

function renderDataTable(data) {
  const tbody = document.getElementById('data-table-body');
  if (!tbody) return;

  // show last 72 records (3 days)
  const recentData = data.slice(-72);

  tbody.innerHTML = recentData.map(d => {
    let rowClass = '';
    if (d.isNight) rowClass = 'row-night';
    if (d.dayType === 'Event Day') rowClass = 'row-event';

    let badgeClass = 'badge-weekday';
    if (d.dayType === 'Event Day') badgeClass = 'badge-event';
    if (d.dayType === 'Weekend') badgeClass = 'badge-weekend';

    return `
      <tr class="${rowClass}">
        <td>${d.date}</td>
        <td>${HOUR_LABELS[d.hour]}</td>
        <td><strong>${d.kWh} kWh</strong></td>
        <td><span class="badge ${badgeClass}">${d.dayType}</span></td>
        <td>${d.isNight ? '<span class="badge badge-night">🌙 Night</span>' : '<span class="badge badge-day">☀️ Day</span>'}</td>
        <td>${d.tempC}°C</td>
        <td>${d.humidity}%</td>
      </tr>
    `;
  }).join('');
}

// =================== RESET ===================

function resetDashboard() {
  modelResults = null;
  currentFilter = 'All';

  document.getElementById('daytype-select').value = 'All';

  // reset progress
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  progressContainer.classList.remove('active');
  progressFill.style.width = '0%';
  progressText.textContent = '';

  // clear metrics
  ['metric-accuracy', 'metric-r2', 'metric-rmse', 'metric-mae', 'metric-mape', 'metric-train', 'metric-test'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });

  // destroy model charts
  ['prediction', 'loss', 'forecast'].forEach(key => {
    if (chartInstances[key]) {
      chartInstances[key].destroy();
      chartInstances[key] = null;
    }
  });

  // clear log
  const logScroll = document.getElementById('log-scroll');
  if (logScroll) {
    logScroll.innerHTML = '<p style="color: var(--text-muted); font-size: 0.8rem; padding: 6px 0;">Logs cleared. Click "Train LSTM Model" to start.</p>';
  }

  // reset train button
  document.getElementById('train-btn').textContent = '🧠 Train LSTM Model';

  renderDashboard();
  addLog('Dashboard reset — ready to retrain', 'ok');
}

// =================== UTILITIES ===================

// shared chart options for consistent styling
function getChartOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { size: 11 } }
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        titleColor: '#f0f4f8',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(55, 65, 81, 0.5)',
        borderWidth: 1,
        cornerRadius: 8,
      }
    },
    scales: {
      x: {
        ticks: { color: '#475569', font: { size: 10 }, maxTicksLimit: 20 },
        grid: { color: 'rgba(55, 65, 81, 0.15)' },
      },
      y: {
        ticks: { color: '#475569', font: { size: 10 } },
        grid: { color: 'rgba(55, 65, 81, 0.15)' },
        title: { display: true, text: yLabel, color: '#475569', font: { size: 11 } },
      }
    }
  };
}

// log entry helper
function addLog(message, type = 'info') {
  const logScroll = document.getElementById('log-scroll');
  if (!logScroll) return;

  // remove placeholder if present
  const placeholder = logScroll.querySelector('p');
  if (placeholder) placeholder.remove();

  const time = new Date().toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });

  const typeClass = { 'ok': 'msg-ok', 'warn': 'msg-warn', 'info': 'msg-info' }[type] || 'msg-info';

  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="time">[${time}]</span> <span class="${typeClass}">${message}</span>`;

  logScroll.prepend(entry);

  // keep max 80 entries
  while (logScroll.children.length > 80) {
    logScroll.removeChild(logScroll.lastChild);
  }
}

// loading overlay
function showLoading(msg) {
  const overlay = document.getElementById('loading-overlay');
  const text = document.getElementById('loading-text');
  if (overlay) overlay.classList.add('active');
  if (text) text.textContent = msg || 'Loading...';
}

function hideLoading() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.classList.remove('active');
}
