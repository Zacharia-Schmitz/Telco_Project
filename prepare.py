import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from env import db_url
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def get_telco_data():
    """
    This function retrieves the telco data from a CSV file if it exists,
    otherwise it retrieves the data from a SQL database and saves it to a CSV file.
    This function assumes the user has an env file formatted as:
    __________________________________________________
    user = "username"
    password = "password"
    host = "host"

    f"mysql+pymysql://{user}:{password}@{host}/{db}"
    __________________________________________________

    Args:
    - None

    Returns:
    - df_telco: pandas dataframe containing the telco data
    """
    filename = "telco.csv"
    if os.path.isfile(filename):
        df_telco = pd.read_csv(filename)
        print("CSV Found")
    else:
        query = """
        SELECT *
        FROM customers
        JOIN contract_types
        ON customers.contract_type_id = contract_types.contract_type_id
        JOIN internet_service_types
        ON customers.internet_service_type_id = internet_service_types.internet_service_type_id
        JOIN payment_types
        ON customers.payment_type_id = payment_types.payment_type_id;"""
        connection = db_url("telco_churn")
        df_telco = pd.read_sql(query, connection)
        df_telco.to_csv(filename, index=False)
        print("CSV Created")
    return df_telco


def prep_telco(df_telco):
    # Drop the joiner columns from SQL and definitely useless columns
    df_telco = df_telco.drop(
        columns=[
            "internet_service_type_id",
            "contract_type_id",
            "payment_type_id",
            "contract_type_id.1",
            "internet_service_type_id.1",
            "payment_type_id.1",
        ]
    )
    # Fill NA in internet service type with what the rest of the table uses, 'No internet service'.
    df_telco["internet_service_type"].fillna("No internet service", inplace=True)

    # As we've noticed before, ['total_charges'] is detected as an object, but holds mainly numbers. We'll use pd.to_numeric() to force everything to a number,
    # and anything that it cannot convert, we'll make it Null and investigate those.
    df_telco["total_charges"] = pd.to_numeric(
        df_telco["total_charges"], errors="coerce"
    )
    # Total charges = 0 also has Tenure = 0. We'll assume they're new and haven't been charged
    df_telco["total_charges"].fillna(0, inplace=True)

    # Automatic payments could be important for churn, so we will separate it from the payment method
    df_telco["automatic_payments"] = (
        df_telco["payment_type"] == "Bank transfer (automatic)"
    ) | (df_telco["payment_type"] == "Credit card (automatic)")

    # Now that it is separated, we remove 'automatic_payment' from the payment types. Made casing uniform
    df_telco["payment_type"].replace(
        {
            "Electronic check": "electronic check",
            "Mailed check": "mailed check",
            "Credit card (automatic)": "credit card",
            "Bank transfer (automatic)": "bank transfer",
        },
        inplace=True,
    )

    # Change senior_citizen to Yes/No for better visuals
    df_telco["senior_citizen"].replace({0: "No", 1: "Yes"}, inplace=True)

    # Bin the tenure into groups of 12 to separate by year
    df_telco["tenure_years"] = df_telco["tenure"].apply(lambda x: int((x - 1) / 12))

    # Change from True / False to Yes / No for visuals and uniformity, also for tablewide binary conversion later
    df_telco["automatic_payments"].replace({False: "No", True: "Yes"}, inplace=True)

    # Create a total add ons that counts all of the add ons
    df_telco["total_add_ons"] = df_telco[
        [
            "phone_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]
    ].apply(lambda x: (x == "Yes").sum(), axis=1)

    # Due to needing the customer_id later, we can't drop it. We'll assign it as
    # the index to keep it out of the way
    df_telco.set_index("customer_id", inplace=True)

    # Change churn to binary
    df_telco["churn"].replace({"No": 0, "Yes": 1}, inplace=True)

    # Rename some columns
    df_telco.rename(
        columns={
            "tenure": "tenure_months",
            "partner": "married",
            "dependents": "kids",
        },
        inplace=True,
    )
    return df_telco


def check_columns(df_telco):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    and the number of null values in the column. The resulting dataframe
    is sorted by the 'Number of Unique Values' column in ascending order.

    Args:
    - df_telco: pandas dataframe

    Returns:
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df_telco.columns:
        # Append the column name, number of unique values, unique values, and number of null values to the data list
        data.append(
            [
                column,
                df_telco[column].nunique(),
                df_telco[column].unique(),
                df_telco[column].isna().sum(),
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', and 'Number of Null Values'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
        ],
    ).sort_values(by="Number of Unique Values")


def telco_ml(df_telco):
    """
    This function takes in a pandas dataframe and returns
    a modified version of the dataframe with binary values
    for certain columns.

    Args:
    - df_telco: pandas dataframe

    Returns:
    - df_telco: pandas dataframe
    """
    # Replace values with binary values
    df_telco.replace(
        {
            "No internet service": 0,
            "No phone service": 0,
            "No": 0,
            "Yes": 1,
            "Male": 1,
            "Female": 0,
        },
        inplace=True,
    )

    # 3 Categories (but we want to keep all 3)
    categorical = ["contract_type", "payment_type", "internet_service_type"]

    # Get dummies for categorical columns
    cat = pd.get_dummies(df_telco[categorical], drop_first=False, dtype="int")

    # Rename columns for uniformity
    cat.rename(
        columns={
            "contract_type_Month-to-month": "month_to_month_contract",
            "contract_type_One year": "one_year_contract",
            "contract_type_Two year": "two_year_contact",
            "payment_type_bank transfer": "bank_transfer_payment",
            "payment_type_credit card": "credit_card_payment",
            "payment_type_electronic check": "e_check_payment",
            "payment_type_mailed check": "mailed_check_payment",
            "internet_service_type_DSL": "dsl_internet",
            "internet_service_type_Fiber optic": "fiber_optic_internet",
        },
        inplace=True,
    )

    # Concatenate the original dataframe with the categorical dataframe
    df_telco = pd.concat([df_telco, cat], axis=1)

    # Drop redundant columns
    df_telco.drop(
        columns=[
            "internet_service_type_0",
            "contract_type",
            "internet_service_type",
            "payment_type",
        ],
        inplace=True,
    )
    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # select the columns to scale
    cols_to_scale = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "total_add_ons",
        "tenure_years",
    ]

    # fit the scaler to the selected columns
    scaler.fit(df_telco[cols_to_scale])

    # transform the selected columns
    df_telco[cols_to_scale] = scaler.transform(df_telco[cols_to_scale])

    return df_telco


def split_telco(df_telco):
    """
    Splits a DataFrame into training, validation, and test sets, stratifying on the 'churn' variable.

    Parameters:
    df_telco (pandas.DataFrame): The DataFrame to split.

    Returns:
    tuple: A tuple containing the training, validation, and test DataFrames.
    """
    train_validate, test = train_test_split(
        df_telco, test_size=0.2, random_state=123, stratify=df_telco["churn"]
    )
    train, validate = train_test_split(
        train_validate,
        test_size=0.3,
        random_state=123,
        stratify=train_validate["churn"],
    )
    print(
        f"train: {len(train)} ({round(len(train)/len(df_telco)*100)}% of {len(df_telco)})"
    )
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df_telco)*100)}% of {len(df_telco)})"
    )
    print(
        f"test: {len(test)} ({round(len(test)/len(df_telco)*100)}% of {len(df_telco)})"
    )

    return train, validate, test
