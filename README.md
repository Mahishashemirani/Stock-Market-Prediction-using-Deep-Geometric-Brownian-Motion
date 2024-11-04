# Stock-Market-Prediction-using-Deep-Geometric-Brownian-Motion

This repository explores the application of **Deep Learning** combined with **Geometric Brownian Motion (GBM)** to model and predict stock prices. The project introduces an extension of GBM by incorporating an **alpha-stable noise term**, aiming to better capture rare, extreme events in stock price dynamics.

## Geometric Brownian Motion (GBM) for Stock Market Predictions

**Geometric Brownian Motion** is a stochastic process widely used in modeling stock prices and financial markets. The GBM model expresses stock prices as a continuous-time stochastic process, defined by the following differential equation:


$dS_t = \mu S_t dt + \sigma S_t dW_t$

where:
- $\( S_t \)$ is the stock price at time \( t \).
- $\( \mu \)$ represents the **drift**, or the expected return rate.
- $\( \sigma \)$ represents the **volatility** of the stock.
- $\( W_t \)$ is a **Wiener process** (or Brownian motion), responsible for introducing random fluctuations.

In this model, **future price changes** are influenced by both a deterministic drift $(\( \mu \))$ and a random noise component scaled by volatility $(\( \sigma \))$, making it a powerful tool for capturing **basic price dynamics** over time.  
This equation can be solved and the solutions is:  
$ S_t = S_0 \exp\(\(\mu - \frac{\sigma ^ 2}{2} \)t + \sigma W_t\)$
GBM is widely used in financial modeling because:
1. It assumes **log-normal distribution** of stock prices, aligning with the historical tendency of stock prices to be positive and skewed.
2. It captures the **random walk behavior** observed in stock prices, which often exhibit volatility and unpredictable fluctuations in the short term.

## Limitations of GBM in Accounting for Rare Events

While GBM effectively models standard market fluctuations, it falls short in representing **rare events** or **extreme price movements** (e.g., market crashes or sudden spikes). This limitation arises from the Wiener process, $\( W_t \)$, which assumes **normal (Gaussian) distribution** of returns.

- **Normal distribution** implies that extreme deviations are highly improbable. In practice, however, financial markets often experience events that are far more extreme than what a normal distribution would predict.
- **Thin tails** of the Gaussian distribution underrepresent the probability of rare events, making GBM less effective in capturing **high-risk, high-impact events** seen in real-world markets.

## Extending GBM with an Alpha-Stable Noise Term

To address the limitations of the Gaussian noise term, this project proposes an alternative noise model: **alpha-stable noise**.

### Why Alpha-Stable Distributions?

**Alpha-stable distributions** are a family of probability distributions characterized by their **heavy tails**, allowing them to model extreme events more effectively than Gaussian distributions. An alpha-stable distribution can capture a **higher likelihood of extreme deviations**, offering a more realistic representation of stock returns.

By replacing the Wiener process $\( W_t \)$ with an **alpha-stable noise term**, the stock price dynamics become:

$dS_t = \mu S_t dt + \sigma S_t dL_t^\alpha$

where:
- $\( L_t^\alpha \)$ is an **alpha-stable Lévy process** with stability parameter $\( \alpha \)$.
- **When $\( \alpha < 2 \)$**, the distribution captures **heavier tails** than a Gaussian, allowing for a higher frequency of extreme changes.

This modified model introduces **flexibility in the noise component** of the price dynamics, enabling it to better reflect **non-Gaussian behaviors** observed in markets, such as **fat tails** and **volatility clustering**. These features are essential for realistic stock market prediction, as they account for the **higher-than-expected probability of large price swings** that occur in real financial markets.

---
