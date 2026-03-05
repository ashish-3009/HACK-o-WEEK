// ============================================================
// MODEL MODULE — TensorFlow.js LSTM for Electricity Prediction
// ============================================================
// This implements a proper LSTM using TensorFlow.js (loaded via
// CDN in index.html). Using TF.js gives us real backpropagation,
// GPU acceleration, and proper weight optimization.
//
// Architecture:
//   Input (window of kWh values) → LSTM(32 units) → Dense(16) → Dense(1)
//
// We use a sliding window approach:
//   - Take the last `windowSize` hours as input features
//   - Predict the next hour's electricity consumption
//
// The LSTM learns temporal patterns like:
//   - Daily cycles (low at night, peak during events)
//   - Day-type differences (event vs normal)
//   - Post-event ramp-down patterns
//
// Reference: Hochreiter & Schmidhuber, 1997 (LSTM)
// ============================================================

// -- Configuration --
const MODEL_CONFIG = {
  windowSize: 24,       // look back a full 24 hours to capture daily cycles
  lstmUnits: 64,        // increased LSTM capacity
  denseUnits: 32,       // increased dense layer capacity
  epochs: 50,           // training epochs
  batchSize: 32,        // mini-batch size
  learningRate: 0.002,  // slightly faster learning rate
  validationSplit: 0.15, // 15% for validation during training
  trainTestRatio: 0.8,  // 80% train, 20% test
};

/**
 * Normalize an array of values to [0, 1] range.
 * Returns the normalized array plus min/max for denormalization.
 */
function normalizeArray(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;  // avoid division by zero
  const normalized = arr.map(v => (v - min) / range);
  return { normalized, min, max };
}

/**
 * Denormalize a single value back to original scale.
 */
function denormalize(val, min, max) {
  return val * (max - min) + min;
}

/**
 * Create sliding window sequences for LSTM training.
 * 
 * Given hourly kWh data, we create sequences like:
 *   Input: [kWh_t-12, kWh_t-11, ..., kWh_t-1]
 *   Target: kWh_t
 *
 * Each input is shaped as [windowSize, 1] (1 feature = kWh).
 * 
 * @param {Array} kWhValues - raw kWh values in order
 * @param {number} windowSize - number of past hours to look at
 * @returns {Object} - xs (inputs), ys (targets), normalization params
 */
function createWindows(kWhValues, windowSize) {
  // normalize the data first
  const { normalized, min, max } = normalizeArray(kWhValues);

  const xs = [];  // input sequences
  const ys = [];  // target values

  for (let i = windowSize; i < normalized.length; i++) {
    // take a window of `windowSize` past values
    const window = normalized.slice(i - windowSize, i);
    xs.push(window.map(v => [v]));  // shape: [windowSize, 1]
    ys.push(normalized[i]);         // single target value
  }

  return { xs, ys, min, max };
}

/**
 * Build the LSTM model using TensorFlow.js.
 * Architecture: LSTM(32) → Dense(16, relu) → Dense(1)
 */
function buildModel(windowSize) {
  const model = tf.sequential();

  // LSTM layer — this is the core RNN component
  // inputShape: [timesteps, features] = [windowSize, 1]
  model.add(tf.layers.lstm({
    units: MODEL_CONFIG.lstmUnits,
    inputShape: [windowSize, 1],
    returnSequences: false,  // only output the last timestep
  }));

  // Dense hidden layer with ReLU activation
  model.add(tf.layers.dense({
    units: MODEL_CONFIG.denseUnits,
    activation: 'relu',
  }));

  // Output layer — single neuron for kWh prediction
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear',  // regression output, no activation
  }));

  // compile with Adam optimizer and MSE loss (standard for regression)
  model.compile({
    optimizer: tf.train.adam(MODEL_CONFIG.learningRate),
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  return model;
}

/**
 * Main function: train the LSTM model on the dataset and return predictions.
 * 
 * @param {Array} dataset - parsed CSV data from data.js
 * @param {string} dayTypeFilter - 'All', 'Event Day', etc.
 * @param {Function} progressCallback - called with (epoch, totalEpochs, loss)
 * @returns {Object} - predictions, actuals, forecast, metrics, losses
 */
async function trainAndPredict(dataset, dayTypeFilter, progressCallback) {
  // step 1: filter by day type if needed
  let filtered = dataset;
  if (dayTypeFilter !== 'All') {
    filtered = dataset.filter(d => d.dayType === dayTypeFilter);
  }

  // step 2: extract kWh values in order
  const kWhValues = filtered.map(d => d.kWh);
  const windowSize = MODEL_CONFIG.windowSize;

  // step 3: create sliding window sequences
  const { xs, ys, min, max } = createWindows(kWhValues, windowSize);
  console.log(`Created ${xs.length} sequences (window=${windowSize})`);

  // step 4: split into train and test sets
  const splitIdx = Math.floor(xs.length * MODEL_CONFIG.trainTestRatio);
  const trainXs = xs.slice(0, splitIdx);
  const trainYs = ys.slice(0, splitIdx);
  const testXs = xs.slice(splitIdx);
  const testYs = ys.slice(splitIdx);

  console.log(`Train: ${trainXs.length}, Test: ${testXs.length}`);

  // step 5: convert to TensorFlow tensors
  // shape of trainX: [numSamples, windowSize, 1]
  const trainXTensor = tf.tensor3d(trainXs);
  const trainYTensor = tf.tensor1d(trainYs);

  // step 6: build and train the model
  const model = buildModel(windowSize);

  // track losses for the chart
  const losses = [];
  const valLosses = [];

  await model.fit(trainXTensor, trainYTensor, {
    epochs: MODEL_CONFIG.epochs,
    batchSize: MODEL_CONFIG.batchSize,
    validationSplit: MODEL_CONFIG.validationSplit,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        losses.push(logs.loss);
        valLosses.push(logs.val_loss);

        // report progress to the UI
        if (progressCallback) {
          // denormalize the loss for display (approx)
          const realLoss = logs.loss * Math.pow(max - min, 2);
          progressCallback(epoch + 1, MODEL_CONFIG.epochs, realLoss);
        }
      }
    }
  });

  // step 7: make predictions on test set
  const testXTensor = tf.tensor3d(testXs);
  const predTensor = model.predict(testXTensor);
  const predNormalized = await predTensor.data();

  // denormalize predictions and actuals
  const predictions = Array.from(predNormalized).map(v =>
    Math.max(0, Math.round(denormalize(v, min, max)))
  );
  const actuals = testYs.map(v => Math.round(denormalize(v, min, max)));

  // step 8: generate 12-hour post-event forecast
  // use the last windowSize values as seed
  const lastWindow = xs[xs.length - 1].map(v => v[0]);  // get normalized values
  const forecast = [];
  let currentWindow = [...lastWindow];

  for (let i = 0; i < 12; i++) {
    const inputTensor = tf.tensor3d([currentWindow.map(v => [v])]);
    const pred = model.predict(inputTensor);
    const predVal = (await pred.data())[0];

    forecast.push(Math.max(0, Math.round(denormalize(predVal, min, max))));

    // slide the window forward (auto-regressive)
    currentWindow = [...currentWindow.slice(1), predVal];

    // clean up tensors to avoid memory leak
    inputTensor.dispose();
    pred.dispose();
  }

  // step 9: calculate accuracy metrics
  const n = actuals.length;
  let sumSE = 0, sumAE = 0, sumAPE = 0;
  for (let i = 0; i < n; i++) {
    const err = actuals[i] - predictions[i];
    sumSE += err * err;
    sumAE += Math.abs(err);
    sumAPE += Math.abs(err) / (actuals[i] || 1);
  }

  const mse = sumSE / n;
  const rmse = Math.sqrt(mse);
  const mae = sumAE / n;
  const mape = (sumAPE / n) * 100;

  // R² score
  const meanActual = actuals.reduce((a, b) => a + b, 0) / n;
  const ssTot = actuals.reduce((sum, a) => sum + Math.pow(a - meanActual, 2), 0);
  const ssRes = actuals.reduce((sum, a, i) => sum + Math.pow(a - predictions[i], 2), 0);
  const r2 = 1 - (ssRes / (ssTot || 1));

  // accuracy = 100 - MAPE (capped at 0)
  const accuracy = Math.max(0, 100 - mape);

  // step 10: clean up tensors to free memory
  trainXTensor.dispose();
  trainYTensor.dispose();
  testXTensor.dispose();
  predTensor.dispose();

  console.log(`Training complete! RMSE: ${rmse.toFixed(2)}, R²: ${r2.toFixed(4)}, Accuracy: ${accuracy.toFixed(1)}%`);

  return {
    model,
    predictions,
    actuals,
    forecast,
    losses,
    valLosses,
    metrics: {
      rmse: Math.round(rmse * 100) / 100,
      mae: Math.round(mae * 100) / 100,
      mape: Math.round(mape * 100) / 100,
      r2: Math.round(r2 * 10000) / 10000,
      mse: Math.round(mse * 100) / 100,
      accuracy: Math.round(accuracy * 100) / 100,
      meanActual: Math.round(meanActual * 10) / 10,
    },
    dataInfo: {
      totalSamples: kWhValues.length,
      trainSize: trainXs.length,
      testSize: testXs.length,
      windowSize,
      normMin: min,
      normMax: max,
    }
  };
}
