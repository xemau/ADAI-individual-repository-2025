# Demand Forecasting Report - Prophet vs SARIMAX

## 1. Executive Summary

We evaluated two statistical time-series models, **Prophet** and **SARIMAX**, on the AMI monthly demand dataset (around 3 years of monthly data per SKU across multiple product families).

Both models are suitable for structured numerical forecasting. The main conclusions:

- **LLMs should not be used to generate the numerical forecasts.** They are not designed to capture time-dependent structure (trend, seasonality, autocorrelation) in a statistically controlled way and do not provide reliable confidence intervals.
- **Time-series models (Prophet/SARIMAX) should be the forecasting engine.** They provide auditable, reproducible forecasts with clear model assumptions.
- **LLMs are useful as an “insight layer” on top of these forecasts**, translating numbers and uncertainty into human-readable recommendations for planners and management.

From the experiments on SKUs (`SKU1_001`-`SKU1_006`):

- **Prophet** works well for SKUs with clear trend and mild seasonality, with minimal configuration effort.
- **SARIMAX** performs competitively where seasonality is stable and the series is not too noisy, and it offers more classical, interpretable parameters.
- For intermittent and very noisy SKUs, **both models struggle**, and these SKUs require special handling (segmentation, different methods or manual overrides).

Recommended direction for the company:

1. Use **Prophet / SARIMAX as the core forecasting models** per SKU.
2. Wrap them in a **hybrid architecture**: forecasting engine + LLM insight layer.
3. Start with a **simple prototype**, then extend to a multi-model setup and external data integration over time, as described in the AMI case.

---

## 2. Dataset and Methods

### 2.1 Dataset

- **Time period:** approx. January 2022 - December 2024 (36 monthly observations).
- **Granularity:** Monthly demand per SKU (`qty`), with associated unit cost.
- **Coverage:** Multiple SKUs across several product families (FAM1-FAM5).
- **Quality:** No structural missing months; series can be converted to regular monthly frequency.

A subset of SKUs from FAM1 (`SKU1_001`-`SKU1_006`) was selected for detailed testing, covering different patterns (smooth demand, trend, more noisy, intermittent).

### 2.2 Models Implemented

Two notebooks are the core of the experimentation:

- **Prophet notebook (`02_prophet_forecasting.ipynb`)**
  - Loads the time-series data and filters by SKU.
  - Prepares data in Prophet format (`ds`, `y`).
  - Fits a Prophet model with yearly seasonality.
  - Performs a backtest on a holdout window (e.g. last 6 months) and computes MAE/RMSE.
  - Produces 12-month ahead forecasts with uncertainty intervals.
  - Visualises results using interactive Plotly charts.

- **SARIMAX notebook (`03_sarimax_forecasting.ipynb`)**
  - Loads the same data and converts each SKU series to a proper monthly index.
  - Fits a SARIMAX model with a basic seasonal structure (e.g. order (1,0,1), seasonal_order (1,0,1,12)).
  - Performs a backtest on the same holdout horizon with MAE/RMSE.
  - Produces 12-month ahead forecasts with confidence intervals.
  - Saves per-SKU forecast CSVs and a metrics summary in `../artifacts`.

Both notebooks evaluate the models using **out-of-sample backtesting**, not just in-sample fit.

---

## 3. Model Comparison: Prophet vs SARIMAX

### 3.1 Prophet

**Strengths**

- Simple to configure and run: requires minimal manual parameter tuning.
- Built-in handling of trend and yearly seasonality.
- Automatically provides forecast intervals (`yhat_lower`, `yhat_upper`).
- Well-suited as a general-purpose baseline model across many SKUs.

**Where it fits well**

- SKUs with relatively smooth demand and visible trend.
- SKUs with weak-moderate yearly seasonality.
- Situations where the forecasting team wants something robust and easy to maintain.

**Limitations**

- Less transparent in terms of explicit AR/MA structure.
- Structural assumptions (smooth trend + seasonality) are not ideal for highly intermittent or extremely noisy demand.

### 3.2 SARIMAX

**Strengths**

- Classical, well-understood time-series model.
- Explicit AR/MA and seasonal terms allow more detailed control over autocorrelation and seasonality.
- Provides confidence intervals and residual diagnostics.
- Matches “SARIMA-style” solutions often recommended for stable, seasonal demand.

**Where it fits well**

- SKUs with strong, stable seasonal patterns and moderate noise.
- Use cases where the team wants more interpretable parameters and traditional time-series diagnostics.

**Limitations**

- Requires explicit choice of (p, d, q) and seasonal parameters; incorrect choices undermine performance.
- For short or sparse series, parameter estimation can be unstable (as seen in warnings about “too few observations” for seasonal ARMA).
- More effort is needed to tune per SKU or per family.

### 3.3 Observations from the Experiments

From backtests (MAE/RMSE per SKU) and visual inspection:

- On smooth SKUs with clear trend, Prophet generally performs at least as well as SARIMAX and is easier to manage.
- On SKUs with strong yearly seasonality and enough history, SARIMAX can be competitive or slightly better when correctly specified.
- On intermittent or very noisy SKUs, errors are higher for both models; model choice matters less than demand classification and special handling.

**Conclusion:**  
There is no universal “best” model across all SKUs. A **multi-model approach**, where each SKU is evaluated on several models and the best is selected by error metrics, is more appropriate than forcing a single model everywhere.

---

## 4. Role of LLMs in the Forecasting Workflow

The AMI case study explicitly warns against using LLMs as the primary forecasting method for structured time-series data:

- LLMs do not natively model time dependence, trend, and seasonality in a statistically disciplined way.
- They are non-deterministic and do not provide transparent, mathematical uncertainty estimates.
- Their arithmetic and numerical reasoning are not reliable enough for operational forecasting.

However, LLMs are very useful as an **interpretation and communication layer**:

- They can read tables of forecasts, confidence intervals, and backtest errors.
- They can generate narratives for planners and management, for example:
  - “Demand for SKU1_001 is trending upward with moderate forecast error; consider a 10-15% production increase over the next year.”
  - “Forecast uncertainty for SKU1_004 is high; consider higher safety stock or closer monitoring.”
  - “Family FAM1 shows stable demand; current capacity appears sufficient, but FAM3 is more volatile and deserves extra attention.”

In other words, **time-series models do the calculation**, and the **LLM explains what those results mean for inventory, capacity, and purchasing decisions**.

---

## 5. Recommendations for the Company

Based on the experiments with Prophet and SARIMAX and the guidance from the AMI case:

### 5.1 Short Term - Prototype (Option A style)

- Implement a **simple forecasting prototype**:
  - Use Prophet as the default model for all SKUs.
  - For SKUs with clearly strong seasonality and enough data, optionally run SARIMAX as a comparison.
  - Produce 12-month forecasts and confidence intervals.
  - Compute MAE/RMSE backtest metrics per SKU and log them in `../artifacts`.

- Add an **LLM-based insight layer**:
  - Feed the forecast tables and metrics to an LLM.
  - Generate SKU-level and family-level narratives and recommendations for planners and management.

This is fast to implement and already demonstrates a tangible benefit.

### 5.2 Medium Term - Multi-Model Approach (Option B style)

- Move towards a **multi-model per-SKU** solution:
  - For each SKU, run at least Prophet and SARIMAX.
  - Optionally add a third model (for example gradient boosting or random forest on lagged features) if resources allow.
  - Select the “best” model per SKU based on backtest error (for example MAE or RMSE on the last 6-12 months).

- Use the LLM to:
  - Explain which model is used for which SKU and why.
  - Summarize model performance and confidence levels.
  - Prioritize SKUs where forecast risk is highest.

### 5.3 Longer Term - External Data and Scenario Planning (Options C/D style)

- Integrate **external drivers** (where available):
  - Promotions, special events, macroeconomic indicators, market intelligence.
- Build scenario forecasts (base case, optimistic, pessimistic).
- Let the LLM:
  - Explain differences between scenarios.
  - Suggest inventory and capacity strategies under each scenario.

This aligns with the case study’s vision of a more advanced, enterprise-level forecasting solution.

---

## 6. My Recommendation

- **Use Prophet and SARIMAX as the core forecasting tools**. They are mathematically appropriate, auditable, and align with established time-series practice.
- **Do not rely on LLMs to generate the raw forecasts**, but use them to interpret results, communicate risk, and support decision-making.
- Start with a **simple, focused prototype**, then extend into a multi-model and scenario-based solution:
  1. Prototype: one primary model (Prophet) + basic LLM insights.
  2. Multi-model: per-SKU model selection between Prophet, SARIMAX, and others.
  3. Enterprise: integration of external signals and scenario planning.

This approach gives the company a realistic, incremental path from basic forecasting to a more intelligent, explainable demand planning system that combines solid statistical models with modern language models.  