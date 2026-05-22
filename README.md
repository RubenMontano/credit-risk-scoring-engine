# Credit Risk Scoring Engine — P2P Lending · Spain Market

*Engineered by Rubén Montaño Camacho — Economist & Data Scientist*

> **Predictive default detection system** built on CatBoost + SHAP, purpose-engineered for the Spanish subprime lending segment. Deployed as a full-stack risk decision architecture: from raw data ingestion to regulatory-grade explainability and macroeconomic stress testing.

---

## The Business Problem

Bondora, a leading European P2P lending platform, was operating its credit concession engine with a model trained on Nordic and Baltic borrower profiles — then applying it directly to Spanish applicants. The result: **systematic mispricing of risk via Concept Drift**, with default rates that could not be explained, predicted, or defended before regulators.

This project engineers a **market-specific scoring system for Spain**, resolving three critical failure points simultaneously:

| Failure | Root Cause | Solution Delivered |
|---|---|---|
| Discriminatory power near random | Geographic model drift | Spain-only retraining with localized feature space |
| Regulatory non-compliance | Opaque ML decisions | Full SHAP audit layer — per-application traceability |
| Static risk policy | No stress resilience | Macroeconomic shock simulation (Basel III-aligned) |

---

## Architecture at a Glance

```
Raw Bondora Data (179K records)
        │
        ▼
┌─────────────────────────────────┐
│  STAGE 1 · DATA ENGINEERING     │
│  • Geographic filter → Spain    │
│  • Temporal filter → 2013–2020  │
│  • Active loan exclusion        │
│  • Data leakage audit (2 passes)│
│  • Endogeneity purge            │
│  → 24,352 clean records / 35 vars│
└────────────────┬────────────────┘
                 │
        ▼
┌─────────────────────────────────┐
│  STAGE 2 · FEATURE ENGINEERING  │
│  PTI  = MonthlyPayment / Income │
│  DTI  = Liabilities / Income    │
│  Margin = FreeCash / Income     │
└────────────────┬────────────────┘
                 │
        ▼
┌──────────────────────────────────────────────────┐
│  STAGE 3 · MODEL COMPETITION                      │
│                                                    │
│  Benchmark (Bondora PD)   →  AUC: 0.4928 ❌       │
│  Logistic Regression      →  AUC: 0.6522 ⚠️       │
│  CatBoost (optimized)     →  AUC: 0.7071 ✅       │
│                                                    │
│  Hyperparameter search: RandomizedSearchCV (CV=3) │
│  Depth: 8 · Iterations: 500 · LR: 0.05            │
│  Class imbalance: auto_class_weights='Balanced'   │
└────────────────┬─────────────────────────────────┘
                 │
        ▼
┌─────────────────────────────────┐
│  STAGE 4 · EXPLAINABILITY (XAI) │
│  Global: SHAP Summary Plot      │
│  Local:  Per-application audit  │
│  → Regulatory audit-ready       │
└────────────────┬────────────────┘
                 │
        ▼
┌─────────────────────────────────┐
│  STAGE 5 · DECISION FRAMEWORK   │
│  Cut-off optimization           │
│  Internal Rating System (A→E)   │
│  Risk-Based Pricing engine      │
│  Stress Testing (Basel III)     │
└─────────────────────────────────┘
```

---

## Performance Metrics

### Discriminatory Power

| Metric | Bondora Benchmark | Logistic Regression | **CatBoost** |
|---|---|---|---|
| ROC-AUC | 0.4928 | 0.6522 | **0.7071** |
| Gini Coefficient | — | — | **41.42%** |
| KS Statistic | — | — | **0.2900** |
| PR-AUC | — | — | **0.8376** |
| Brier Score | — | — | **0.2135** |

> A Gini >40% in unsecured subprime P2P lending is classified as **highly competitive** in credit risk literature (Baesens et al., 2016). The KS of 0.29 falls within the regulatory tolerance band of 0.20–0.30.

### Decision Threshold Calibration

Two operationally distinct thresholds were derived via cost-matrix optimization:

| Scenario | Threshold | Use Case |
|---|---|---|
| **Stress / Capital Protection** | 9% | Blocks 100% of toxic capital. Crisis-mode policy. |
| **Operational / Profit-Maximizing** | 26% | Blocks 99.1% of toxic capital. Day-to-day deployment. |

### Quantified Financial Impact (Test Set Simulation)

```
Capital blocked from defaulters   →  + €3,311,175
Opportunity cost (rejected solvent) →  - €1,039,964
─────────────────────────────────────────────────
Net estimated savings              →    €2,271,211
```

> Model prevents **99.1% of toxic capital loss** while maintaining commercial viability of the portfolio.

---

## Internal Rating System (Risk-Based Pricing)

| Rating | Predicted PD | Min. Risk Premium | Decision |
|---|---|---|---|
| A | 0% – 5% | 2.56% | ✅ Approved |
| B | 5% – 10% | 8.11% | ✅ Approved |
| C | 10% – 15% | 14.29% | ✅ Approved |
| D | 15% – 20% | 21.21% | ✅ Approved |
| E | 20% – 26% | 29.97% | ✅ Approved |
| — | > 26% | Off-market | ❌ Rejected |

Premia are derived from: `PD_mean × LGD / (1 − PD_mean)`, assuming LGD = 100% (unsecured loan, zero recovery).

---

## Explainability Layer (XAI · SHAP)

The top global risk drivers, as identified by SHAP attribution analysis:

```
HIGH IMPACT (Global)
├── Age                          → Demographic risk segmentation
├── CreditScoreEsMicroL          → External credit bureau signal
├── Education                    → Behavioral stability proxy
├── PreviousRepaymentsBeforeLoan → Primary default mitigator (↓ risk)
├── LiabilitiesTotal             → Debt burden amplifier (↑ risk)
├── Ratio_Deuda_Ingreso_DTI      → Leverage ratio signal
├── AppliedAmount                → Exposure size (↑ risk)
└── IncomeTotal                  → Liquidity shield (↓ risk)
```

**Local explainability** enables per-application audit trails — every approval or denial is fully traceable to the contributing variables, satisfying GDPR Art. 22 and Basel III transparency requirements.

---

## Stress Testing (Basel III / IFRS 9 Aligned)

A multifactor shock was applied to the test portfolio to simulate a severe macroeconomic downturn:

| Shock Variable | Magnitude | Rationale |
|---|---|---|
| IncomeTotal | −20% | Recession / partial unemployment scenario |
| LiabilitiesTotal | +15% | Inflationary debt repricing |
| MonthlyPayment | +10% | Interest rate tightening |

**Result:** The model activated a **mass preventive block** — the vast majority of borderline-approved profiles (Ratings D/E) crossed the 26% default threshold, demonstrating **elastic risk sensitivity** and automatic portfolio de-risking without manual intervention.

> The 26% cut-off functions as a dynamic firewall: it maximizes yield in expansion cycles and autonomously protects capital during recessions.

---

## Data Engineering Highlights

Two-phase leakage elimination was critical to model integrity:

**Phase 1 — Temporal Leakage Purge:**
Variables generated post-origination were excised: `Restructured`, `PrincipalDebtServicingCost`, `InterestAndPenaltyDebtServicingCost`, `PlannedInterestTillDate`.

**Phase 2 — Endogeneity Purge:**
Platform-assigned internal signals removed to avoid circular reasoning: `Rating`, `Interest`, `BidsApi`, `BidsManual`, `BidsPortfolioManager`, `Amount`. The platform's own `ProbabilityOfDefault` was isolated exclusively for benchmark validation.

**Class imbalance (70.4% default rate):** Addressed via `auto_class_weights='Balanced'` in CatBoost — preserving real distribution without synthetic noise injection (SMOTE was explicitly rejected to avoid data distortion).

---

## Tech Stack

```
Language        Python 3.x
ML Framework    CatBoost · Scikit-learn
Explainability  SHAP (SHapley Additive exPlanations)
Data            Pandas · NumPy
Optimization    RandomizedSearchCV · Early Stopping (patience=50)
Evaluation      ROC-AUC · Gini · KS · PR-AUC · Brier Score · Confusion Matrix
Dataset         Bondora P2P Lending — Spain cohort (2013–2020)
```

---

## Repository Structure

```
├── data/
│   └── bondora_spain_clean.csv        # Processed dataset (Spain-only, closed loans)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_xai_shap_analysis.ipynb
│   └── 05_stress_testing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── explainability.py
├── outputs/
│   ├── shap_summary_plot.png
│   ├── roc_curve.png
│   ├── calibration_curve.png
│   └── stress_test_migration.png
└── README.md
```

---

## Known Limitations & Future Roadmap

| Limitation | Proposed Resolution |
|---|---|
| Random train/test split | Out-of-time validation (chronological split) |
| Static stress testing | Agent-Based Modeling for dynamic macro shocks |
| Survivorship bias in closed loans | Survival Analysis (time-to-default modeling) |
| No inferential significance testing | DeLong test for AUC curve comparison |
| Declared income (fraud risk) | PSD2 / Open Banking real-time transaction feeds |
| Fairness risk (age, education) | Fairness-aware ML with constrained loss functions |

---

## Relevance to Target Roles

This project directly demonstrates capabilities required in:

- **Credit Risk / Wholesale Banking:** PD modeling, LGD assumption design, Basel III-aligned validation, internal ratings architecture
- **Financial Crime / AML:** Anomaly scoring, threshold calibration, audit-trail generation via XAI
- **Risk & Regulatory Transformation:** GDPR Art. 22 compliance, EU AI Act (high-risk system classification), explainability-by-design

---

*Dataset source: Siddhartha (2021). Bondora Peer to Peer Lending Loan Data. Kaggle.*
*Methodology references: Baesens et al. (2016), Lundberg & Lee (2017), Prokhorenkova et al. (2018)*

---

## 📬 Contact & Profile

If this work is relevant to a role you're hiring for, I'd welcome the conversation.

**Rubén Montaño Camacho**
Economist & Data Scientist — specializing in Credit Risk, Financial Crime, and predictive architecture for financial institutions.

| | |
|---|---|
| 🔗 LinkedIn | [linkedin.com/in/rubenmontano](https://www.linkedin.com/in/rubenmontano/) |
| ✉️ Email | rmontanocamacho0@gmail.com |
