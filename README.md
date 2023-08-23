<div style="background-color: #212946; padding: 10px; color: #F62196; text-align: center;">

#### Codeup Project
***
</div>

<div style="background-color: #212946; padding: 10px; text-align: center; color: #18C0C4;">

<br>

# **Project Telcoco** 
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

'|    | Column Name             |   Number of Unique Values | Unique Values                                                           |   Number of Null Values |\n|---:|:------------------------|--------------------------:|:------------------------------------------------------------------------|------------------------:|\n|  0 | gender                  |                         2 | [0 1]                                                                   |                       0 |\n| 26 | mailed_check_payment    |                         2 | [1 0]                                                                   |                       0 |\n| 25 | e_check_payment         |                         2 | [0 1]                                                                   |                       0 |\n| 24 | credit_card_payment     |                         2 | [0 1]                                                                   |                       0 |\n| 23 | bank_transfer_payment   |                         2 | [0 1]                                                                   |                       0 |\n| 22 | two_year_contact        |                         2 | [0 1]                                                                   |                       0 |\n| 21 | one_year_contract       |                         2 | [1 0]                                                                   |                       0 |\n| 20 | month_to_month_contract |                         2 | [0 1]                                                                   |                       0 |\n| 17 | automatic_payments      |                         2 | [0 1]                                                                   |                       0 |\n| 16 | churn                   |                         2 | [0 1]                                                                   |                       0 |\n| 27 | dsl_internet            |                         2 | [1 0]                                                                   |                       0 |\n| 13 | paperless_billing       |                         2 | [1 0]                                                                   |                       0 |\n| 28 | fiber_optic_internet    |                         2 | [0 1]                                                                   |                       0 |\n| 11 | streaming_tv            |                         2 | [1 0]                                                                   |                       0 |\n| 10 | tech_support            |                         2 | [1 0]                                                                   |                       0 |\n|  9 | device_protection       |                         2 | [0 1]                                                                   |                       0 |\n|  8 | online_backup           |                         2 | [1 0]                                                                   |                       0 |\n|  7 | online_security         |                         2 | [0 1]                                                                   |                       0 |\n|  6 | multiple_lines          |                         2 | [0 1]                                                                   |                       0 |\n|  5 | phone_service           |                         2 | [1 0]                                                                   |                       0 |\n|  3 | kids                    |                         2 | [1 0]                                                                   |                       0 |\n|  2 | married                 |                         2 | [1 0]                                                                   |                       0 |\n|  1 | senior_citizen          |                         2 | [0 1]                                                                   |                       0 |\n| 12 | streaming_movies        |                         2 | [0 1]                                                                   |                       0 |\n| 18 | tenure_years            |                         6 | [0.  0.2 1.  0.8 0.4 0.6]                                               |                       0 |\n| 19 | total_add_ons           |                         8 | [0.57142857 0.28571429 0.71428571 0.42857143 1.         0.14285714      |                       0 |\n|    |                         |                           |  0.         0.85714286]                                                 |                         |\n|  4 | tenure_months           |                        73 | [0.125      0.05555556 0.18055556 0.04166667 0.98611111 0.875           |                       0 |\n|    |                         |                           |  0.09722222 0.90277778 0.75       1.         0.06944444 0.77777778      |                         |\n|    |                         |                           |  0.47222222 0.01388889 0.625      0.69444444 0.31944444 0.76388889      |                         |\n|    |                         |                           |  0.36111111 0.95833333 0.51388889 0.68055556 0.91666667 0.93055556      |                         |\n|    |                         |                           |  0.27777778 0.59722222 0.81944444 0.16666667 0.375      0.02777778      |                         |\n|    |                         |                           |  0.34722222 0.40277778 0.19444444 0.48611111 0.88888889 0.54166667      |                         |\n|    |                         |                           |  0.55555556 0.15277778 0.08333333 0.41666667 0.97222222 0.79166667      |                         |\n|    |                         |                           |  0.80555556 0.22222222 0.44444444 0.45833333 0.13888889 0.29166667      |                         |\n|    |                         |                           |  0.84722222 0.20833333 0.61111111 0.30555556 0.33333333 0.26388889      |                         |\n|    |                         |                           |  0.65277778 0.86111111 0.63888889 0.72222222 0.11111111 0.83333333      |                         |\n|    |                         |                           |  0.66666667 0.38888889 0.56944444 0.73611111 0.94444444 0.43055556      |                         |\n|    |                         |                           |  0.5        0.23611111 0.25       0.70833333 0.52777778 0.58333333      |                         |\n|    |                         |                           |  0.        ]                                                            |                         |\n| 14 | monthly_charges         |                      1585 | [0.47114428 0.41442786 0.55373134 ... 0.73134328 0.50298507 0.49353234] |                       0 |\n| 15 | total_charges           |                      6531 | [0.06831476 0.06245394 0.03233811 ... 0.08554025 0.53284474 0.42690678] |                       0 |'




## Steps to Reproduce

1. Clone this repository.

2. If you have access to the Codeup MySQL DB:
   - Save **env.py** in the repository with `user`, `password`, and `host` variables.
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
