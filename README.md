# E-Commerce Customer Retention Prediction

**Goal:** Predict customer churn and segment users based on behavioral patterns using machine learning (XGBoost).

---

## 🔍 Project Pipeline
1. **Data Acquisition**
   - Pulled e-commerce dataset from Kaggle.
   - Simulated live data via API (JSON from GitHub Gist).

2. **Data Cleaning & Feature Engineering**
   - Normalized text, handled missing values.
   - Created behavioral metrics: RFM Score, Loyalty Score, Avg Spend per Item, etc.

3. **Customer Segmentation**
   - K-Means + Elbow Method to identify 4 customer segments.
   - PCA visualization for interpretable cluster separation.

4. **Modeling**
   - XGBoost classifier for churn prediction.
   - Evaluated via Accuracy, ROC-AUC, Cross-Validation.

5. **Interpretation**
   - Feature importance and SHAP analysis for transparency.

---

## 📈 Key Results
- **ROC-AUC:** 0.87 (± 0.02)
- Top drivers: Loyalty Score, RFM Score, Total Spend.
- Segments reveal 4 distinct customer behaviors.

---

## 🧠 Insights
- High-spend, loyal customers are least likely to churn.  
- Younger customers with low frequency and spend are most at-risk.

---

## 🚀 Future Work
- Expand dataset with transaction history.
- Experiment with LightGBM + AutoML tuning.
- Deploy inference API with FastAPI.

---

## 📂 Project Structure
src/ # all scripts
data/processed/ # cleaned & derived data
reports/plots/ # visualizations
models/ # trained model

## 💡 Key Insights & Recommendations

### Customer Behavior Insights
- **High Loyalty and Spending = Retention:**  
  Customers with strong Loyalty and RFM Scores show the lowest churn risk.  
  These customers maintain frequent purchases, high satisfaction, and premium memberships.

- **Young, Low-Spend Customers = High Churn Risk:**  
  The highest churn probability is among younger users with low purchase frequency and small order values.  
  These users often rely on discounts and show low long-term loyalty.

- **Membership Type Matters:**  
  Gold and Platinum members display consistently higher retention, validating tier-based incentives.  
  Silver members have the most diverse behavior — an opportunity for personalized engagement.

### Strategic Recommendations
- **1. Targeted Retention Campaigns:**  
  Use Loyalty Score thresholds to trigger automated re-engagement offers before churn signals appear.  

- **2. Personalize Discounts:**  
  Offer discount campaigns selectively for low-RFM customers to boost early-stage loyalty rather than across-the-board promotions.

- **3. Enhance Membership Benefits:**  
  Expand perks for Gold/Platinum users and create onboarding incentives for Silver customers to upgrade.

- **4. Monitor Satisfaction Metrics:**  
  Integrate feedback loops — declining satisfaction often precedes reduced spending and churn.

### Business Takeaway
The data suggests a dual strategy: **retain high-value customers through loyalty programs** while **nurturing low-engagement users** via personalized, data-driven campaigns.  
Predictive churn modeling, when coupled with behavioral segmentation, allows proactive retention planning — turning insights into measurable growth.

## 🔮 Future Work

1. **Expand Dataset Depth**  
   Incorporate detailed transaction histories and customer lifetime value (CLV) data to improve churn signal strength and feature diversity.

2. **Model Optimization**  
   Experiment with ensemble methods (LightGBM, CatBoost) and AutoML tuning to further enhance predictive accuracy and generalization.

3. **Temporal Analysis**  
   Add time-based features such as purchase intervals, trend shifts, and rolling averages for a more dynamic customer behavior view.

4. **Deployment & Monitoring**  
   Convert the XGBoost model into an inference API (FastAPI or Flask) and create a live dashboard to monitor churn probabilities in real time.

5. **A/B Testing & Feedback Loop**  
   Integrate churn predictions into targeted retention campaigns and measure actual behavioral outcomes — closing the analytics loop.

---
