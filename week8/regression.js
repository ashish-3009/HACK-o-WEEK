// ============================================================
// POLYNOMIAL REGRESSION MODULE
// ============================================================
// Implements ordinary-least-squares polynomial regression
// without external libraries.  Used to predict optimal
// light levels from vehicle-count occupancy percentages.
// ============================================================

class PolynomialRegression {
  /**
   * @param {number} degree – polynomial degree (2 = quadratic, 3 = cubic …)
   */
  constructor(degree = 3) {
    this.degree = degree;
    this.coefficients = [];
  }

  // ---- Linear-algebra helpers (Gauss-Jordan elimination) ----

  /** Build the Vandermonde matrix for xs at the given degree */
  _vandermonde(xs) {
    return xs.map(x => {
      const row = [];
      for (let p = 0; p <= this.degree; p++) row.push(Math.pow(x, p));
      return row;
    });
  }

  /** Transpose matrix */
  _transpose(M) {
    const rows = M.length, cols = M[0].length;
    const T = Array.from({ length: cols }, () => new Array(rows));
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) T[j][i] = M[i][j];
    return T;
  }

  /** Multiply two matrices */
  _multiply(A, B) {
    const rA = A.length, cA = A[0].length, cB = B[0].length;
    const C = Array.from({ length: rA }, () => new Array(cB).fill(0));
    for (let i = 0; i < rA; i++)
      for (let j = 0; j < cB; j++)
        for (let k = 0; k < cA; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  }

  /** Multiply matrix by column vector */
  _multiplyVec(A, v) {
    return A.map(row => row.reduce((s, a, i) => s + a * v[i], 0));
  }

  /** Gauss-Jordan inverse of a square matrix */
  _invert(M) {
    const n = M.length;
    const aug = M.map((row, i) => {
      const id = new Array(n).fill(0);
      id[i] = 1;
      return [...row, ...id];
    });

    for (let col = 0; col < n; col++) {
      // partial pivot
      let maxRow = col;
      for (let row = col + 1; row < n; row++)
        if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

      const pivot = aug[col][col];
      if (Math.abs(pivot) < 1e-12) throw new Error('Singular matrix');
      for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = aug[row][col];
        for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
      }
    }
    return aug.map(row => row.slice(n));
  }

  // ---- Public API ----

  /**
   * Fit polynomial: y ≈ c0 + c1·x + c2·x² + … + cn·x^n
   * @param {number[]} xs – independent variable (occupancy %)
   * @param {number[]} ys – dependent variable (lux)
   */
  fit(xs, ys) {
    const X = this._vandermonde(xs);
    const Xt = this._transpose(X);
    const XtX = this._multiply(Xt, X);
    const XtX_inv = this._invert(XtX);
    const Xty = this._multiplyVec(Xt, ys);
    this.coefficients = this._multiplyVec(XtX_inv, Xty);
    return this;
  }

  /**
   * Predict y for a single x value.
   */
  predict(x) {
    return this.coefficients.reduce((sum, c, p) => sum + c * Math.pow(x, p), 0);
  }

  /**
   * Predict for an array of x values.
   */
  predictMany(xs) {
    return xs.map(x => this.predict(x));
  }

  /**
   * R² (coefficient of determination)
   */
  r2(xs, ys) {
    const preds = this.predictMany(xs);
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
    const ssTot = ys.reduce((s, y) => s + (y - mean) ** 2, 0);
    const ssRes = ys.reduce((s, y, i) => s + (y - preds[i]) ** 2, 0);
    return 1 - ssRes / ssTot;
  }

  /** Return human-readable equation string */
  equationString() {
    return this.coefficients
      .map((c, p) => {
        const coef = c.toFixed(4);
        if (p === 0) return coef;
        if (p === 1) return `${coef}·x`;
        return `${coef}·x^${p}`;
      })
      .join(' + ');
  }
}
