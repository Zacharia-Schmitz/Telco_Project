<div style="background-color: #212946; padding: 10px; color: #F62196; text-align: center;">

<b>*Codeup Project*</b>
***
</div>

<div style="background-color: #212946; padding: 10px; text-align: center; color: #18C0C4;">

<br>
<br>
<br>

# **Project Telcoco** 
<br>
<br>
<br>

![image.png](attachment:image.png) 

<br>
<br>
<br>

</div>

## Learn to Discern What Turns Customers to Churn

### Project Description

Telco is a telecommunications company that offers a variety of services to a diverse customer base. This project focuses on investigating the different factors that contribute to customer churn, aiming to determine whether any of these factors increase or decrease the likelihood of customers churning.

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

| Feature               | Values                  | Definition                                    |
| :-------------------- | ----------------------- | :-------------------------------------------- |
| customer_id           | Alpha-numeric           | Unique identifier for each customer          |
| gender                | Female/Male             | Customer's gender                            |
| senior_citizen        | True=1/False=0          | Whether the customer is a senior citizen     |
| partner               | True=1/False=0          | Whether the customer has a partner          |
| dependents            | True=1/False=0          | Whether the customer has dependents          |
| ...                   | ...                     | ...                                          |
| churn (target)        | True=1/False=0          | Whether the customer has churned             |
| Additional Features   | True=1/False=0          | Encoded values for categorical data         |

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
