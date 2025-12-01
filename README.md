# Customer Churn Prediction – tbi Data Science Case Study

A practical and reproducible machine learning project that predicts telecom customer churn to support retention and marketing decision-making.


## Business Objective

The goal is to identify customers at high risk of churn and recommend actionable retention strategies based on data-driven insights.

This project delivers:

- Early detection of churn-risk customers  
- Interpretable drivers for retention strategy  
- Scalable scoring pipeline for deployment (CRM, Marketing Automation)  


## Key Churn Drivers Identified

The most influential predictors discovered in analysis:

- **Contract Type** (month-to-month contracts show highest churn propensity)
- **Payment Method** (electronic check correlates with churn clusters)
- **Tenure** (shorter customer lifetime = higher churn probability)
- **Monthly Charges** (higher bills = clear separation in churn distributions)

These features are prioritized in the final model and should be considered for retention campaigns and marketing incentives.


## Modeling Strategy

We tested multiple algorithms and selected **Gradient Boosting** as champion based on:

1 Highest discrimination (top AUROC, strong KS separation)  
2 Stable recall-precision trade-off for business use  
3 Fast and efficient training performance  

Champion model performance after tuning:

| Metric | Score |
|--------|-------|
| Accuracy | **0.801** |
| Precision | **0.670** |
| Recall | **0.485** |
| AUROC | **0.854** |
| KS | **0.554** |
| Gini | **0.708** |

This indicates strong model ability to differentiate between churned and retained customers.


## Data Preprocessing Decisions

- Converted business binary labels `Yes/No` into numeric 1/0 features  
- Encoded `gender` as 1/0 for model consistency  
- Applied **one-hot encoding** to multi-level categorical features  
  (`internetservice`, `contract`, `paymentmethod`)
- Normalized numeric predictors using **Min-Max scaling (0 to 1)**  
- Handled `totalcharges` for `tenure == 0` customers by assigning **0**  
  (new customers not yet billed – valid business state, not missing data)


## Saved Artifacts

The following artifacts are exported for reproducibility and inference deployment:

- `Churn_Champion_GB.pkl` → Trained Gradient Boosting champion model  
- `Churn_Pipeline.pkl` → Full inference pipeline (encoder + scaler + model)  
- `requirements.txt` → Exact dependency versions for environment setup  


## How to Run

1. Install project dependencies:
```bash
pip install -r requirements.txt

# telco-churn-tbi
