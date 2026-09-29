# Regime-Aware USD/INR Futures Trading with HMM + Safe Reinforcement Learning

A research-driven framework for **30-minute USD/INR currency futures forecasting and risk-aware trade selection**.

The project separates two problems that are often combined in algorithmic trading:

1. **What market regime is likely next?** — handled by a Gaussian Hidden Markov Model (HMM).
2. **Is that forecast worth trading after costs and risk?** — handled by an expert gate and a Fitted-Q reinforcement-learning meta-controller.

The objective is not to force a trade on every prediction. The system is designed to trade **selectively**, account for realistic execution costs, and explicitly penalize adverse risk.

---

## Project at a Glance

| Component | Approach |
|---|---|
| Market | NSE USD/INR currency futures |
| Data source | Zerodha Kite Connect |
| Frequency | 30-minute bars |
| Forecast target | Bearish / Sideways / Bullish |
| Regime model | 3-state Gaussian HMM |
| Final HMM features | Short-horizon, bar-based EMA dynamics |
| Forecasting style | Causal forward filtering |
| Trading gate | Tradability + direction persistence + time-slot prior |
| RL controller | Fitted-Q meta-controller |
| Q-models | Bayesian Ridge regression |
| RL state | 29 features |
| Actions | No trade / 2 contracts / 4 contracts |
| Risk-aware reward | Net P&L − 0.20 × Maximum Adverse Excursion |
| Evaluation | Chronological holdout + walk-forward trading |
| Cost model | Spread, slippage, brokerage, NSE/SEBI/IPFT charges, GST, stamp duty, hedge cost |

---

## Why This Project Exists

Short-horizon FX prediction is difficult for three reasons:

- **Directional signal is weak** relative to market noise.
- **Market behaviour changes over time**, so a single stationary model can struggle.
- A statistically correct prediction may still be **economically useless after transaction costs**.

An early direct-RL approach made this especially clear: allowing an agent to learn direction, timing, sizing and risk simultaneously was too unconstrained for the available data.

This project therefore uses a more controlled architecture:

```text
USD/INR Futures Data
        |
        v
Preprocessing + Causal Features
        |
        v
3-State Gaussian HMM
        |
        v
Bearish / Sideways / Bullish probabilities
        |
        v
Expert Trading Gate
        |
        v
Fitted-Q Meta-Controller
        |
        +----> No Trade
        +----> 2 Contracts
        +----> 4 Contracts
        |
        v
Cost- and Risk-Aware Trading Evaluation
```

The **HMM proposes market structure and direction**.  
The **RL layer decides whether acting on that signal is economically justified**.

---

## Dataset

Historical USD/INR futures data were collected through the **Zerodha Kite Connect API**.

| Property | Value |
|---|---|
| Historical window | 11 Mar 2026 – 11 Sep 2026 |
| Raw dates returned | 82 |
| Active days retained | 54 |
| Bar frequency | 30 minutes |
| Trading grid | 09:00 – 16:30 |
| Expected bars/day | 16 |
| Active-day requirement | ≥12 observed bars and ≥8 actually traded bars |

USD/INR futures expire periodically, so individual contracts were stitched into a continuous series using the actively traded contract.

Missing grid observations were carried forward only to maintain a regular time index and were explicitly flagged so that inactive periods were not silently interpreted as genuine price stability.

---

## Forecast Target

The next market movement is classified into three regimes:

```text
Bearish    Sideways    Bullish
```

Instead of labeling every tiny price movement as directional, the project uses a **causal volatility-sensitive band**.

Conceptually:

```text
future return > +band  -> Bullish
future return < -band  -> Bearish
otherwise              -> Sideways
```

The threshold combines tick-size awareness with a rolling measure of recent absolute returns.

This is important because a movement can be statistically non-zero while still being too small to trade profitably.

---

## Gaussian HMM

A **three-state Gaussian Hidden Markov Model** models latent market regimes.

The final state space is:

```text
Bearish | Sideways | Bullish
```

Several feature representations were evaluated:

| Feature approach | Purpose |
|---|---|
| Return-volatility features | Capture immediate movement, range and realised volatility |
| Day-based EMA trend features | Introduce slower trend information |
| Bar-based EMA dynamics | Capture shorter-horizon intraday trend changes |

The final forecasting configuration used **short-horizon bar-based EMA dynamics**, including information derived from 10-bar and 21-bar EMA windows.

### Why three hidden states?

Increasing the number of HMM states improved statistical fit, but did not consistently improve unseen forecasting performance.

Three states were retained because they provide a useful balance between:

- out-of-sample forecasting performance,
- model simplicity,
- and direct economic interpretation.

---

## Causal Forecasting

Financial backtests are easily inflated by accidental future information.

This project uses **causal forward filtering**: at time `t`, the model can only use information available up to `t`.

For every bar, the HMM produces:

```text
P(Bearish)
P(Sideways)
P(Bullish)
```

The probabilities sum to one and are passed to the downstream trading layer.

No future observations are used to infer the current regime.

---

## HMM Forecasting Results

The forecasting model was trained on **43 active days** and evaluated on **11 later unseen active days**.

| Metric | Result |
|---|---:|
| Test balanced accuracy | **39.3%** |
| Three-class chance level | 33.3% |
| Strong simple baseline | 36.2% |
| Day-block bootstrap CI | 33.1% – 45.4% |
| Permutation p-value | **0.023** |
| Sideways recall | **77.4%** |
| Bullish recall | 38.2% |
| Bearish recall | 2.3% |

The HMM does **not** solve short-horizon FX direction prediction. Its edge is modest.

What it does provide is a structured, causal regime signal—especially useful for identifying **sideways periods where trading may be undesirable**.

---

## From Forecasts to Trades

The project deliberately avoids trading every HMM signal.

The candidate set is reduced through a sequence of filters:

| Stage | Remaining observations |
|---|---:|
| All test bars | 704 |
| Current + next bar actually traded | 595 |
| HMM produced a directional regime | 194 |
| Same direction persisted for ≥4 bars | 82 |
| Historically eligible entry time | 18 |
| Final executed trades | **12** |

The key insight is that these gates do not primarily increase directional accuracy.

They increase **selectivity**.

The mean signal return rose from roughly **0.11 bps** at the directional-HMM stage to about **0.89 bps** for the final executed entries.

---

## Expert Trading Gate

A signal must satisfy all of the following before reaching the RL controller:

- the market observation is tradable,
- the HMM forecast is not sideways,
- the same HMM direction persists for at least four consecutive bars,
- the entry time has a positive historical trading record.

The HMM controls **direction**.

The gate controls **whether the setup is worth evaluating further**.

---

## Fitted-Q Meta-Controller

The reinforcement-learning component is intentionally small.

Instead of a large deep neural network, the project uses **two Bayesian Ridge Q-models**:

```text
Q(s, 2) -> expected utility of trading 2 contracts
Q(s, 4) -> expected utility of trading 4 contracts
```

For every historical eligible opportunity, the environment simulates what would have happened under both position sizes.

Those state-action-reward examples become the training data for the two regressors.

Before each walk-forward trading day, the models are refitted using only completed historical days.

The controller may:

```text
0 contracts -> veto the trade
2 contracts -> reduced position
4 contracts -> full position
```

It **cannot invent a new direction or reverse the HMM forecast**.

---

## Why Bayesian Ridge?

The trading dataset is small, so a high-capacity neural model would be easy to overfit.

Bayesian Ridge is useful because it provides:

- a predicted expected reward,
- an uncertainty estimate around that prediction,
- coefficient regularisation,
- and a comparatively interpretable small-data model.

The controller therefore reasons about both:

```text
expected utility
+
uncertainty in that estimate
```

rather than treating every point prediction as equally reliable.

---

## 29-Feature RL State

Every eligible opportunity is represented by 29 features.

### HMM information

- bearish probability
- sideways probability
- bullish probability
- HMM confidence
- HMM entropy
- estimated directional edge

### Recent market behaviour

- lagged returns 1–4
- recent mean return
- recent volatility

### Calendar context

- time-of-day sine/cosine
- weekday sine/cosine
- week-of-month sine/cosine
- month sine/cosine
- day-of-month sine/cosine

### Historical priors

- class-time prior
- class-slot prior
- class-weekday prior
- class-week-of-month prior

### Persistence and session context

- HMM streak
- bars remaining in the session
- HMM direction

The controller therefore evaluates the HMM signal **in context**, rather than from forecast direction alone.

---

## Risk-Aware Reward

The RL target is not raw profit.

The reward is:

```text
Reward = Net P&L - 0.20 × Maximum Adverse Excursion
```

**Maximum Adverse Excursion (MAE)** is the worst temporary loss experienced while a trade is open.

This means two trades with the same final profit are not considered equally good if one required surviving a much larger drawdown.

---

## Realistic Transaction Costs

The simulator includes:

- half-spread,
- slippage,
- brokerage,
- NSE transaction charges,
- SEBI turnover fees,
- IPFT charges,
- GST,
- stamp duty,
- hedge premium.

This matters because the strategy operates on short-horizon FX moves where trading friction can consume most of the gross edge.

---

## Safety Controls

The trading layer includes additional safeguards:

### Time-slot cooldown

A recently adverse trading slot can be temporarily blocked from generating repeated exposure.

### Synthetic hedge

A protection layer reduces the impact of sufficiently large adverse futures movement at the cost of a hedge premium.

### Thesis-break exit

A trade may be exited when subsequent market information increasingly contradicts the original HMM direction.

Together, these mechanisms are designed to make the system **selective and loss-aware**, not simply aggressive.

---

## Walk-Forward Trading Results

The trading layer was evaluated across **44 sessions**.

| Metric | Expert only | + Cooldown | + Fitted-Q RL |
|---|---:|---:|---:|
| Trades | 14 | 14 | **12** |
| Win rate | 42.9% | 50.0% | **58.3%** |
| Net P&L | ₹81 | ₹532 | **₹716.5** |
| Maximum drawdown | 0.77% | 0.54% | **0.50%** |
| Daily Sharpe | 0.21 | 1.41 | **1.93** |
| Transaction costs | ₹1,639 | ₹1,639 | **₹1,404** |

The main improvement is not more trading.

It is **fewer, more selective trades with better observed risk-adjusted outcomes**.

---

## Reward Breakdown

For the final 12 executed trades:

| Component | Rupees |
|---|---:|
| Gross futures movement | +1,427.4 |
| Hedge payout | +693.6 |
| Execution + hedge costs | -1,404.5 |
| **Net P&L** | **+716.5** |
| MAE risk penalty | -406.0 |
| **Reward / utility** | **+310.5** |

Transaction costs were almost as large as the gross futures contribution.

That is a central result of the project: **a forecasting model should not be judged without the economics of actually trading its predictions**.

---

## What Did the RL Controller Actually Do?

At the final decision stage:

| Decision | Count |
|---|---:|
| Full-size trade | 12 |
| Time-slot cooldown | 3 |
| Confident RL veto | 3 |

The RL layer is therefore best understood as a **meta-controller**.

Its contribution comes largely from deciding when **not** to trade.

---

## Learning Day by Day

The Fitted-Q controller is retrained in walk-forward fashion.

Its available training examples grew from:

```text
39 -> 233
```

over the evaluation period as completed days added new eligible historical opportunities.

Across the 12 executed trades, predicted and realised reward had a correlation of approximately:

```text
r = 0.27
```

This is a modest relationship.

The Q-model should therefore **not** be interpreted as accurately forecasting exact future trade profit.

Its current value is primarily in:

- vetoing unattractive opportunities,
- controlling position size,
- and providing another layer of selectivity.

---

## Did the HMM Actually Help?

A useful attribution test separates executed trades according to whether the HMM direction was correct at entry.

| HMM direction | Trades | Net P&L | Avg. trade |
|---|---:|---:|---:|
| Wrong at entry | 6 | -₹340.2 | -₹56.7 |
| Right at entry | 6 | **+₹1,056.7** | **+₹176.1** |

The HMM was correct on only half of the final entries.

However, the **economic asymmetry** is important:

- correct calls generated much larger gains,
- wrong calls produced smaller losses,
- risk controls helped limit adverse outcomes.

This shows why directional hit rate alone is not enough to judge the complete trading system.

---

## Forecast-Source Ablation

The same trading architecture was also tested with alternative forecast sources.

| Forecast source | Trades | Net P&L | Daily Sharpe |
|---|---:|---:|---:|
| Actual HMM | 12 | ₹716.5 | 1.93 |
| True next-bar label | 6 | ₹216.1 | 3.16 |
| Inverted HMM | 5 | ₹1,104.2 | 2.36 |
| Random direction, seed 0 | 0 | ₹0 | — |
| Random direction, seed 1 | 1 | -₹257.4 | -2.39 |
| Random direction, seed 2 | 0 | ₹0 | — |

These values **should not be read as a ranking of forecast models**.

The expert gate requires the same direction to persist for four bars:

- true labels change frequently,
- random directions rarely persist,
- inverted HMM forecasts preserve the HMM's persistence structure.

The experiment reveals an important property of the architecture:

> The gate currently depends strongly on **forecast persistence**, not only forecast correctness.

A cleaner future ablation should separate persistence from directional quality.

---

## Temporal Behaviour

### Performance by entry time

| Entry time | Trades | Net P&L | Win rate |
|---|---:|---:|---:|
| 10:30 | 8 | **₹566.03** | **62%** |
| 11:00 | 1 | ₹42.66 | 100% |
| 12:30 | 1 | ₹411.93 | 100% |
| 14:00 | 1 | -₹207.45 | 0% |
| 15:00 | 1 | -₹96.68 | 0% |

The 10:30 slot contains most of the trading activity and a substantial portion of the observed profit.

Single-trade slots are not large enough to support reliable conclusions.

### Performance by weekday

| Weekday | Trades | Net P&L |
|---|---:|---:|
| Monday | 4 | -₹405.3 |
| Tuesday | 1 | ₹263.3 |
| Wednesday | 2 | **₹574.8** |
| Thursday | 3 | **₹448.4** |
| Friday | 2 | -₹164.8 |

The sample is too small to claim stable weekday effects, but the variation supports retaining calendar context within the RL state.

---

## Bootstrap Sharpe Analysis

The final strategy produced an observed daily Sharpe of:

```text
1.935
```

A 5,000-resample bootstrap produced:

| Statistic | Value |
|---|---:|
| Observed Sharpe | **1.935** |
| Bootstrap median | **2.172** |
| 95% interval | -2.567 to 6.736 |
| Sharpe ≤ 0 | 19.7% |
| Sharpe > 0 | **80.3%** |

The confidence interval crosses zero, so the current experiment **does not establish persistent profitability**.

At the same time, positive risk-adjusted outcomes dominate the current bootstrap distribution, making the result encouraging enough to justify a larger unseen evaluation.

---

## Key Takeaways

1. **Short-horizon USD/INR forecasting contains only a modest signal.**  
   The HMM beats chance and simple baselines, but the predictive advantage is not large.

2. **Sideways detection is valuable.**  
   Avoiding weak directional conditions can be economically useful when transaction costs are significant.

3. **Accuracy and profitability are different problems.**  
   A roughly 50% directional hit rate can still produce positive economics when gains and losses are asymmetric.

4. **Trading costs matter enormously.**  
   Gross performance without transaction costs would materially overstate the quality of the strategy.

5. **The RL layer adds value mainly through selectivity.**  
   It is currently better viewed as a veto/position-sizing controller than as a precise reward predictor.

6. **Risk controls change the economics of incorrect forecasts.**  
   Hedge and exit logic help reduce the damage from wrong directional calls.

7. **The evidence is promising, not conclusive.**  
   Twelve trades are not enough to claim a persistent market edge.

---

## Research Contributions

This project combines:

- causal three-regime HMM forecasting,
- short-horizon USD/INR futures modelling,
- explicit sideways-market detection,
- persistent-signal expert gating,
- small-data Fitted-Q learning,
- Bayesian uncertainty estimates,
- realistic Indian-market transaction costs,
- MAE-aware reward shaping,
- position-size control,
- cooldown and thesis-break safety mechanisms,
- walk-forward evaluation,
- forecast attribution,
- and bootstrap uncertainty analysis.

The key design principle is:

> **Do not ask reinforcement learning to discover everything. Give it a structured forecast, restrict its authority, charge it realistic costs, and let it learn when not to trade.**

---

## Limitations

This is an experimental research system, not a production trading strategy.

Current limitations include:

- only 54 active days in the forecasting dataset,
- only 12 final executed trades,
- weak bearish recall,
- wide Sharpe confidence interval,
- Q-model reward correlation of only 0.27,
- strong dependence of the expert gate on HMM persistence,
- concentration of profit in a small number of time slots,
- no external macro or cross-asset variables in the final forecasting model,
- and reuse of the July–September period during strategy development.

These constraints are intentionally reported rather than hidden.

---

## Next Steps

Planned extensions include:

- freezing the current HMM + trading policy before collecting additional data,
- evaluating on a completely untouched post-development period,
- extending USD/INR history,
- incorporating volatility-regime information,
- adding exogenous features such as DXY, crude oil, equity-market and macro variables,
- improving bearish-regime identification,
- performing stronger cost and slippage stress tests,
- redesigning the forecast-source ablation so it does not depend on four-bar persistence,
- and evaluating whether the Q-model stabilises as the number of eligible examples grows.

---

## Tech Stack

```text
Python
NumPy
pandas
scikit-learn
Hidden Markov Models
Bayesian Ridge Regression
Zerodha Kite Connect
Jupyter
```

---

## Reproducibility Notes

The project follows several rules intended to keep evaluation realistic:

- chronological rather than random splitting,
- causal features only,
- causal HMM filtering,
- no future labels in model inputs,
- walk-forward Q-model refitting,
- explicit handling of inactive bars,
- detailed transaction-cost accounting,
- and statistical uncertainty reporting.

When reproducing the experiments, API credentials and private brokerage configuration should be provided locally through environment variables and **must not be committed to GitHub**.

---

## Disclaimer

This repository is an academic and experimental research project.

It is **not financial advice**, does not guarantee profitability, and should not be used as the sole basis for live trading decisions. Historical and simulated performance may not generalise to future market conditions.

---

## Authors

Developed as a research project on **regime-aware forecasting and safe reinforcement-learning trading for USD/INR currency futures**.

If you use or build on this work, please cite the repository and the accompanying research paper once published.
