<div style="background-color: #212946; padding: 10px; color: #F62196; text-align: center;">

#### Codeup Project
***
</div>

<div style="background-color: #212946; padding: 10px; text-align: center; color: #18C0C4;">

<br>

# **Project Telco** 
<br>

</div>

## Learn to Discern What Turns Customers to Churn

- Discover drivers of churn of Telco customers

- Use drivers to develop a machine learning models to identify drivers of churn

- Churn as a customer ending their contract or not renewing their contract with Telco

### Project Description

Telco, a telecommunications enterprise, provides a wide array of services catering to a diverse clientele. This endeavor delves into the exploration of distinct elements influencing customer churn. The goal is to ascertain whether any of these factors amplify or diminish the probability of customers discontinuing their services.

### Project Goal

- Identify the drivers of churn among Telco customers.

- Develop a machine learning model to classify customer churn, distinguishing between contract renewals and contract terminations.

- Enhance our understanding of which customer attributes contribute to or mitigate customer churn.

### Initial Thoughts

I hypothesize that the drivers of churn will likely involve dissatisfied customers. Specific services or the lack thereof might be influencing customers to churn. 

## The Plan

1. **Data Acquisition**

   - Obtain data from the Codeup MySQL Database.

2. **Data Preparation**

   - Create new engineered columns from the existing data.

3. **Data Exploration**

   - Explore the data to identify potential drivers of churn by answering initial questions:
     - Is churn independent from payment type?
     - Is churn independent from internet service type?
     - Is churn independent from paperless billing?
     - Are there variations in churn based on monthly charges?

4. **Model Development**

   - Utilize the insights gained from the exploration to build predictive models.
   - Evaluate model performance on training and validation data.
   - Select the best-performing model based on accuracy.
   - Validate the chosen model using the test data.

5. **Conclusion Drawing**

   - Summarize the findings and insights.

## Data Dictionary (Exploration)

| Feature               | Values                      | Definition                            |
| :-------------------- | --------------------------- | :-------------------------------------|
| *index:* customer_id   | Alpha-numeric              | Unique ID for each customer           |
| mailed_check_payment   | True=1/False=0             | Whether customer is/has feature name  |
| e_check_payment        | True=1/False=0             | Whether customer is/has feature name  |
| credit_card_payment    | True=1/False=0             | Whether customer is/has feature name  |
| bank_transfer_payment  | True=1/False=0             | Whether customer is/has feature name  |
| two_year_contact       | True=1/False=0             | Whether customer is/has feature name  |
| one_year_contract      | True=1/False=0             | Whether customer is/has feature name  |
| internet_service_type  | True=1/False=0             | Whether customer is/has feature name  |
| month_to_month_contract| True=1/False=0             | Whether customer is/has feature name  |
| automatic_payments     | True=1/False=0             | Whether customer is/has feature name  |
| churn (target)         | True=1/False=0             | Whether customer is/has feature name  |
| dsl_internet           | True=1/False=0             | Whether customer is/has feature name  |
| paperless_billing      | True=1/False=0             | Whether customer is/has feature name  |
| fiber_optic_internet   | True=1/False=0             | Whether customer is/has feature name  |
| streaming_tv           | True=1/False=0             | Whether customer is/has feature name  |
| tech_support           | True=1/False=0             | Whether customer is/has feature name  |
| device_protection      | True=1/False=0             | Whether customer is/has feature name  |
| online_backup          | True=1/False=0             | Whether customer is/has feature name  |
| online_security        | True=1/False=0             | Whether customer is/has feature name  |
| multiple_lines         | True=1/False=0             | Whether customer is/has feature name  |
| phone_service          | True=1/False=0             | Whether customer is/has feature name  |
| kids                   | True=1/False=0             | Whether customer is/has feature name  |
| married                | True=1/False=0             | Whether customer is/has feature name  |
| senior_citizen         | True=1/False=0             | Whether customer is/has feature name  |
| streaming_movies       | True=1/False=0             | Whether customer is/has feature name  |
| tenure_years           | Numeric Normalized (0 - 1) | Tenure normalized with MinMaxScaler() |
| total_add_ons	       | Numeric Normalized (0 - 1) | Add ons normalized with MinMaxScaler()|
| tenure_months          | Numeric Normalized (0 - 1) | Tenure normalized with MinMaxScaler() |
| monthly_charges        | Numeric Normalized (0 - 1) | Charges normalized with MinMaxScaler()|
| total_charges          | Numeric Normalized (0 - 1) | Charges normalized with MinMaxScaler()|

## Steps to Reproduce

1. Clone this repository.

2. If you have access to the Codeup MySQL DB:
   - Save **env.py** in the repository with `user`, `password`, and `host` variables.
   - Ensure the **env.py** has the appropriate database connection.
   - RandomState 123 is predefined in the functions
   - Run the notebook.

3. If you don't have access:
   - Request access from Codeup.
   - Follow step 2 after obtaining access.

# Conclusions

### Takeaways and Key Findings

- Customers without tech support are churning more than those without
- Payment type, especially electronic check, is a significant driver of churn.
- The influence of fiber optic internet service on churn is surprising given its high speed.
- Paperless billing increases churn, with many churn cases having it enabled.
- Churn rates tend to rise with higher monthly charges.

### Recommendations

- Investigate and address issues related to electronic check payments.
- Analyze potential problems with fiber optic internet service.
- Offer tech support for free or cheap if it means retaining customers.

### Next Steps

- Given more time, delve into the reasons behind the high monthly charges contributing to customer churn.
- Tune the hyperparameters to potentially find a better tuned model
