# **PREDICTING CUSTOMERS LIFETIME VALUE (CLV) IN CAR INSURANCE**

## **Description:**
This project leverages regression machine learning models to identify customer characteristics that contribute to high Customer Lifetime Value (CLV) in the car insurance industry. By analyzing customer data, the model predicts CLV for current customers, helping businesses target high-value customers and improve retention. The model also provides recommendations for acquiring new customers likely to generate higher CLV.

## **Key Findings**
- Linear Regression performed the best with the following evaluation metrics:
  - RMSE: $4,048
  - MAE: $1,729
  - MAPE: ~12%
  - R-squared: 0.65 (indicating moderate explanatory power, but affected by outliers in high CLV cases)
- The model struggled to predict CLV for customers with values >$10,000 due to a lack of sufficient data for this group. Collecting more data for high-value customers will improve model accuracy.
- Key features that strongly influence CLV include Number of Policies (most impactful for customers with 2 policies) and Monthly Premium Auto, which is highly correlated with Vehicle Class and Coverage Type.

## **Actionable Recommendation**
**For Existing Customer:**
1. **Encourage Cross-Selling**: For customers with one policy, initiate cross-selling campaigns to add a second policy.
2. **Simplify Multiple Policies**: For customers holding three or more policies, offer bundling options or higher coverage to consolidate them into two policies.
3. **Increase Monthly Premium Auto**: Encourage customers to upgrade their monthly premium to at least $100 (the average for high CLV customers).

**For New Customer:**
1. Focus on acquiring customers who **own SUVs or luxury cars**, prefer **Premium or Extended Coverage**, and are willing to commit to a Monthly Premium Auto of $100 or more.
2. **Tailored Bundles**: Create custom bundles for SUV and luxury car owners, offering premium coverage options.

## **Conclusion**
By adopting this model, the company can predict customer CLV with reasonable accuracy, especially for customers with CLV up to $10,000. While predictions for high CLV (> $10,000) can be improved with more data, the model offers valuable insights for optimizing customer acquisition and retention strategies.

## **File Information**
- **Dataset:** data_customer_lifetime_value.csv
- **Notebook:** Car Insurance CLV Prediction.ipynb
- **Model File:** CLV_prediction_model.sav
- **Python File for Streamlit Deployment:** CLV_Predictor.py (for local deployment, ensure the model file is in the same directory)
- **Streamlit App Deployment:** [Network URL](http://192.168.100.254:8501)

## **Running Streamlit App**
To run the app locally:
1. Open terminal or command prompt.
2. Navigate to the project folder:

   ```bash
   cd path/to/your/folder
   ```
3. Launch the Streamlit app by typing:

   ```bash
   streamlit run CLV_Predictor.py
   ```

This will launch the application locally on your machine.
