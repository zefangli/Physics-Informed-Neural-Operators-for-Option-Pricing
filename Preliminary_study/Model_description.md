## Updated implementation description (with $\tau = T_{\text{years}} = (T-t)/365$)

### Variables (as used in PyTorch)

We work in **strike-normalized** coordinates:

- $M = S/K$ (moneyness)
- $\tau = T_{\text{years}} = (T-t)/365$ (time-to-maturity in **years**)
- $u$: the (date-level) implied volatility surface vector
- $\hat{\sigma} = \mathrm{NN1}(M,\tau,u)$ (model-implied local/effective volatility)
- $\hat{v} = \mathrm{NN2}(M,\tau,r,q,\hat{\sigma})$ (predicted **normalized** call value)
- $\hat{v}_{true} = V/K$ (target; in your data this is `normalized_price`)

Because we normalized by $K$, the Black–Scholes PDE in $(S,t,V)$ can be expressed in $(M,\tau,\hat{v})$ without explicit dependence on $K$ (the scaling cancels).

---

## Loss terms

### 1) Data loss $\mathcal{L}_{data}$

Computed **only** on observed market samples (not on collocation points):

$$\mathcal{L}_{data}=\frac{1}{N_{data}}\sum_{i=1}^{N_{data}}\left(\hat{v}_i-\hat{v}_{true,i}\right)^2$$

---

### 2) PDE loss $\mathcal{L}_{PDE}$ (corrected for $\tau$ measured in years)

**Key point:** Because the standard Black-Scholes PDE expects time $t$ to be measured in **years** (matching the annualized rates $r$, $q$, and volatility $\sigma$), and we have already transformed our data such that $\tau = (T_{days}-t_{days})/365$ is in **years**, the chain rule is straightforward.

Since forward time $t_{years}$ and time-to-maturity $\tau_{years}$ are related simply by $\tau_{years} = T_{years} - t_{years}$, the time derivative transforms as:

$$\frac{\partial}{\partial t_{years}} = -\frac{\partial}{\partial \tau}$$

Substituting this into the normalized Black-Scholes PDE, the residual $\mathcal{R}$ becomes:

$$\mathcal{R}=
-\frac{\partial \hat{v}}{\partial \tau}
+\frac{1}{2}\hat{\sigma}^2 M^2\frac{\partial^2\hat{v}}{\partial M^2}
+(r-q)M\frac{\partial \hat{v}}{\partial M}
-r\hat{v}$$

And the corresponding PDE loss computed over the collocation points is:

$$\mathcal{L}_{PDE}=\frac{1}{N_{colloc}}\sum_{i=1}^{N_{colloc}}\mathcal{R}_i^2$$

---

### 3) No-arbitrage (shape) loss $\mathcal{L}_{arb}$

For European calls in $M=S/K$ coordinates, enforce:

- monotonicity: $\partial_M \hat{v} \ge 0$
- convexity: $\partial_{MM}\hat{v} \ge 0$

Penalty (evaluated on collocation points, and/or optionally also on data points):

$$\mathcal{L}_{arb}
=\frac{1}{N}\sum \left(
\mathrm{ReLU}\left(-\frac{\partial \hat{v}}{\partial M}\right)
+
\mathrm{ReLU}\left(-\frac{\partial^2 \hat{v}}{\partial M^2}\right)
\right)
$$

---

### 4) Terminal condition at $\tau = 0$

Normalized payoff for a call:

$$\hat{v}(M,0)=\max(M-1,0)$$

Loss:

$$\mathcal{L}_{term}
=
\frac{1}{N_{bndry}}\sum \left(\hat{v}(M,0)-\max(M-1,0)\right)^2$$

---

### 5) Lower boundary as $M\to 0$

Call value goes to 0:

$$
\mathcal{L}_{lower}
=
\frac{1}{N_{bndry}}\sum \left(\hat{v}(0,\tau)\right)^2
$$

---

### 6) Upper boundary as $M\to\infty$ (implemented at $M=M_{max}$)

For large $S$, the call approaches discounted forward intrinsic value. In normalized form:

$$
\hat{v}(M,\tau)\approx M e^{-q\tau}-e^{-r\tau}
$$

Loss at $M_{max}$:

$$
\mathcal{L}_{upper}
=
\frac{1}{N_{bndry}}\sum\left(
\hat{v}(M_{max},\tau) - \left(M_{max}e^{-q\tau}-e^{-r\tau}\right)
\right)^2
$$

---

### Total loss

$$
\mathcal{L}_{total}
=
\lambda_1\mathcal{L}_{data}
+\lambda_2\mathcal{L}_{PDE}
+\lambda_3\mathcal{L}_{arb}
+\lambda_4\mathcal{L}_{term}
+\lambda_5\mathcal{L}_{lower}
+\lambda_6\mathcal{L}_{upper}
$$

---

## Training / evaluation pipeline (conceptual)

1. **Chronological split** by date: first 80% train, next 10% val, last 10% test.
2. For each training batch:
   - compute **data loss** on real samples
   - generate **collocation points** $(M,\tau)$ in the same domain as training (now $\tau$ is **years**)
   - generate **boundary/terminal points** and evaluate boundary losses
   - use autograd for $\hat{v}_M, \hat{v}_{MM}, \hat{v}_\tau$
3. Validate on held-out dates using MSE on $\hat{v}$.
4. Test:
   - price fit error on the chronological test set
   - surface sanity checks by evaluating $\hat{\sigma}(M,\tau)$ on a grid
   - Greeks sanity checks using autograd (optional)

***

Does this clear up how the variables map to the PyTorch tensors? Let me know if you need help adjusting the actual Python code where the gradients are calculated!