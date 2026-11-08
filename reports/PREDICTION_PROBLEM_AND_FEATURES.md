# Consumer Complaint Trends & Response Intelligence Platform: Prediction Problem & Feature Engineering

**Document Title:** Time-Series Prediction Problem Definition, Partial-Period Handling & Feature Specification  
**Project:** Consumer Complaint Trends and Response Intelligence Platform  
**Track:** Machine Learning  
**Programme:** AnalystLab Africa Experience Lab (Week 4)  
**Author:** Machine Learning Engineering Intern  
**Date:** November 2026  

---

## 1. Problem Understanding & Operational Context (Task A)

### 1.1 The Business & Regulatory Challenge
The Consumer Financial Protection Bureau (CFPB) collects consumer complaints regarding financial products and services and routes them directly to financial institutions for review and remediation. Under federal guidelines, institutions such as **Bank of America** are required to provide a formal, substantive response to the consumer and the CFPB within **15 calendar days** (with a statutory ceiling of 60 days under exceptional circumstances).

When complaint volumes fluctuate unpredictably, customer advocacy and regulatory affairs departments face severe operational stress:
- **SLA Non-Compliance:** Spikes in complaint inflow cause caseworkers to exceed the 15-day timely response threshold, risking supervisory citations, regulatory fines, and public enforcement actions.
- **Suboptimal Staffing:** Inability to anticipate complaint volume leads to either overstaffed call centers during lulls (inflated operational expenditure) or acute understaffing during surges (caseworker burnout and delayed remediations).
- **Unmanaged Product Escalations:** Unmonitored increases in complaints within specific product lines (e.g., checking accounts or credit cards) prevent product managers from addressing underlying systemic operational or IT glitches before they escalate into regulatory enforcement.

### 1.2 The Machine Learning Objective
To design, validate, and deploy a **predictive time-series forecasting prototype** that forecasts future **monthly consumer complaint volumes** for Bank of America:
1. **Total Health System Volume:** To forecast aggregate complaint inflows 1 to 6 months in advance for staffing and capacity planning.
2. **Product-Level Disaggregation:** To forecast monthly volumes for the top 4 product categories (`Checking or savings account`, `Credit card or prepaid card`, `Credit reporting, repair services, or other personal reports`, and `Mortgage`) to support specialized dispute resolution teams.

### 1.3 Operational Decisions Supported by the Model
- **Workforce Capacity & Shift Scheduling:** Proactively allocating caseworkers to dispute queues based on 80% upper prediction bounds.
- **Timely Response Safeguards:** Establishing automated triage prioritization when forecast volume exceeds 1,200 complaints/month.
- **Product Vulnerability Tracking:** Early alerting to business unit heads when a product's projected complaint trajectory deviates from historical baselines.

---

## 2. Dataset Inspection & Quality Assessment (Task B)

### 2.1 Table Schema & Primary Key Verification
The project ingests 62,516 historical complaint records from `Consumer_Complaints.xlsx` (`Data` worksheet):

| Field Name | Data Type | Null Count | Null % | Description / Data Quality Finding |
| :--- | :--- | :--- | :--- | :--- |
| **`Complaint ID`** | `int64` | 0 | 0.0% | **Candidate Primary Key.** Fully verified: 62,516 unique values; 0 duplicates. |
| **`Submitted via`** | `str` | 0 | 0.0% | Submission channel (`Web`: 73%, `Referral`: 15%, `Phone`: 8%, `Postal mail`: 3%). |
| **`Date submitted`** | `datetime64` | 0 | 0.0% | Date consumer submitted complaint to the CFPB (2017-05-01 to 2023-08-28). |
| **`Date received`** | `datetime64` | 0 | 0.0% | Date CFPB sent complaint to Bank of America (2017-05-01 to 2023-08-28). |
| **`State`** | `str` | 0 | 0.0% | Consumer's state of residence (CA, FL, TX, NY are top contributors). |
| **`Product`** | `str` | 0 | 0.0% | Financial product classification (9 distinct products). |
| **`Sub-product`** | `str` | 7 | 0.01% | Detailed product category (e.g., Checking account, General-purpose credit card). |
| **`Issue`** | `str` | 0 | 0.0% | Primary complaint reason (e.g., Managing an account, Problem with a purchase). |
| **`Sub-issue`** | `str` | 10,858 | 17.4% | Granular issue specification (missingness indicates no sub-issue exists). |
| **`Company public response`**| `str` | 2,175 | 3.5% | Standardized public response statement. |
| **`Company response to consumer`** | `str` | 0 | 0.0% | Primary response outcome (`Closed with explanation`: 81%, `Closed with monetary relief`: 13%). |
| **`Timely response?`** | `str` | 1,494 | 2.4% | `Yes` (97.4%) / `No` (0.2%). Missing values (2.4%) represent complaints marked `In progress`. |

### 2.2 Forwarding Interval Analysis
The difference between `Date received` (sent to company) and `Date submitted` (received by CFPB) represents the **CFPB complaint-forwarding interval**, not company resolution time:
- Mean interval: **1.22 days**
- Median interval: **0.00 days** (same-day transmission)
- 75th percentile: **0.00 days**
- Maximum: **275.0 days** (isolated edge cases involving manual verification of paper mail)

---

## 3. Handling Partial Reporting Periods

The dataset covers **1 May 2017 to 28 August 2023**. Special methodological care must be taken regarding the boundary months:

1. **May 2017 (Initial Period):**
   - Analysis shows complaints recorded across **all 31 calendar days** (May 1 to May 31), totaling 744 complaints (daily average: 24.0).
   - This represents a complete, valid calendar month and is fully incorporated into the modeling baseline.
2. **August 2023 (Final Period):**
   - The data cuts off abruptly on **August 28, 2023** (only 28 days observed).
   - Furthermore, due to the CFPB submission-to-forwarding ingestion pipeline, complaints submitted between August 20 and August 28 average only 1 to 5 complaints/day compared to 40–60 complaints/day earlier in the month.
   - Total observed in August 2023 is **717 complaints**.
   - **Methodological Treatment:**
     - Incomplete months must **never** be injected into time-series training as a low-volume anomaly, as this corrupts lag features, momentum indicators, and trend parameters.
     - Therefore, our primary time series is aggregated across the **75 complete, continuous monthly periods (May 2017 through July 2023)**.
     - August 2023 is evaluated separately in a dedicated diagnostic study:
       $$\text{August Prorated Volume} = 717 \times \left(\frac{31}{28}\right) \approx 793.8 \text{ complaints}$$

---

## 4. Time-Series Feature Engineering & Leakage Prevention

### 4.1 Strict Lookahead Leakage Prevention
To guarantee real-world forecasting validity, time-series feature engineering enforces strict temporal isolation:
- All features for predicting complaint volume at time $t$ ($\hat{y}_t$) are computed **strictly from historical observations prior to $t$** ($y_{t-1}, y_{t-2}, \dots$).
- No future observations, test-set statistics, or future rolling averages are ever accessible to the models.
- When generating recursive multi-step forecasts ($t+1, t+2, \dots, t+h$), predicted values from earlier steps are recursively passed into subsequent lag buffers.

### 4.2 Engineered Features Matrix (17 Features)

| Feature Name | Category | Mathematical Definition | Operational Rationale |
| :--- | :--- | :--- | :--- |
| **`lag_1`** | Autoregressive | $y_{t-1}$ | Captures immediate past volume persistence. |
| **`lag_2`** | Autoregressive | $y_{t-2}$ | Captures short-term momentum. |
| **`lag_3`** | Autoregressive | $y_{t-3}$ | Captures quarterly baseline. |
| **`lag_6`** | Autoregressive | $y_{t-6}$ | Captures semi-annual cyclicality. |
| **`lag_12`** | Seasonal Lag | $y_{t-12}$ | Captures annual year-over-year seasonality. |
| **`rolling_mean_3m`** | Rolling Window | $\frac{1}{3} \sum_{i=1}^3 y_{t-i}$ | Smoothed recent quarterly moving average. |
| **`rolling_std_3m`** | Volatility | $\text{std}(y_{t-1}, y_{t-2}, y_{t-3})$ | Measures recent volume volatility. |
| **`rolling_mean_6m`** | Rolling Window | $\frac{1}{6} \sum_{i=1}^6 y_{t-i}$ | Medium-term trend anchor. |
| **`rolling_std_6m`** | Volatility | $\text{std}(y_{t-1}, \dots, y_{t-6})$ | Medium-term volatility buffer. |
| **`rolling_mean_12m`**| Rolling Window | $\frac{1}{12} \sum_{i=1}^{12} y_{t-i}$ | 12-month trailing moving average. |
| **`diff_lag1_lag2`** | Momentum | $y_{t-1} - y_{t-2}$ | Month-over-month rate of change. |
| **`diff_lag1_lag12`**| Seasonal Acceleration | $y_{t-1} - y_{t-12}$ | Year-over-year growth velocity. |
| **`ratio_lag1_to_3m`**| Ratio | $y_{t-1} / (\text{rolling\_mean\_3m} + \epsilon)$ | Relative volume expansion ratio. |
| **`sin_month`** | Cyclical Encoding | $\sin(2\pi \cdot m / 12)$ | Smooth cyclical calendar representation. |
| **`cos_month`** | Cyclical Encoding | $\cos(2\pi \cdot m / 12)$ | Orthogonal cyclical calendar representation. |
| **`quarter`** | Calendar | $\lceil m / 3 \rceil$ | Financial quarter (Q1–Q4). |
| **`trend_index`** | Secular Trend | $t \in [0, 74]$ | Linear time-step capturing long-term banking digitalization & regulatory growth. |

---

## 5. Validation Strategy: Chronological & Rolling-Origin

Two complementary validation protocols were executed:

1. **Chronological Holdout Split (Out-of-Sample Test):**
   - **Training Period:** May 2017 to December 2022 (68 monthly observations, 90.7% of timeline).
   - **Holdout Test Period:** January 2023 to July 2023 (7 monthly observations, 9.3% of timeline).
   - Simulates a real-world enterprise scenario where a model trained on historical data up to year-end 2022 is deployed to forecast 2023 complaint volumes.
2. **Rolling-Origin Cross-Validation (Expanding Window):**
   - Initial minimum training window: 36 months (3 years).
   - Forecast horizon: 6 months ($h = 6$).
   - Step size: 2 months.
   - Evaluated across **17 distinct historical forecasting origins** to ensure the model performs robustly across diverse macroeconomic and regulatory environments (pre-COVID, pandemic, post-pandemic inflation).
