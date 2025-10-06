# **PREDICTING CUSTOMERS LIFETIME VALUE (CLV) IN CAR INSURANCE**

## **Description:**
This project focus on discover customers characteristics that lead to high CLV using machine learning model to capture the pattern from current customers dataset. Based on this insight, it will give recommendation of strategy to raise CLV on current customers and focus market on attract new customers by also using the model to predict their CLV.  

## **Key Findings**
- The **Linear Regression** model performed the best with:
  - RMSE : $4,048
  - MAE : $1,729
  - MAPE : ~12%
  - R-squared : 0.65 (moderate explanatory power due to outliers on CLV >$10,000)
- The model face difficulties to capture pattern on customers who predicted having CLV >$10,000 due to its low number data. So, to improve the model performances, it need to collect more customers data who had CLV >$10,000
- The most importances features, based on  that have high impact is Number of Policies, with maximum positive impacts in holding only 2 policies, and Monthly Premium Auto, that highly correlated with customers Vehicle Class and Coverage type

## **Actionable Recommendation**
**Recommendation for Existing Customer:**
1. Encourage Cross-Selling for Customers with One Policy by cross-selling campaigns so they would added one more policy
2. For customers already holding three or more policies, the emphasis should be on simplify their policies by bundling or offering higher coverage or increasing their Monthly Premium Auto and consolidating them into two policies
3. Encourage customers to upgrade their Monthly Premium Auto to at least $100 (the mean of Monthly Premium of customers with CLV values >$10,000)

**Recommendation for New Customer:**
1. Focus acquisition efforts on segments with characteristics predictive of higher CLV. This includes SUV and Luxury car owners Vehicle Class, candidate customer who willing to choose Extended or Premium Coverage, focus on middle to upper-income individuals, especially married and employed or unemployed customers, candidate customers who willing to commit to a Monthly Premium Auto of $100 or more, customers who are willing to hold at least two policies. 
2. Create tailored bundles for customers who own SUVs or luxury vehicles, offering premium or extended coverage that appeals to this segment. This will align with the strategic implication of growing the customer base in these higher-value segments.

## **Conclusion**
By adopting this model, the company could easily predict customers CLV, especially for CLV up to $10,000, and recognize customer who will give CLV more than $10,000 even though the number probably not reliable. 

## **File information**
- **Dataset:** data_customer_lifetime_value.csv
- **Notebook:** Car Insurance CLV Prediction.ipynb
- **Model File:** CLV_prediction_model.sav
- **Python File for Streamlit Deployment:** CLV_Predictor.py (for local deployment, make sure to save this phyton file in similar file directory with the model file)
- **Streamlit App Deployment:** http://192.168.100.254:8501 (Network URL)
