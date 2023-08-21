from sklearn.model_selection import train_test_split
from env import db_url
import os
import pandas as pd


def split_data(df, target):
    """
    take in a DataFrame and return train, validate, and df_telco DataFrames; stratify on a specified variable.
    return train, validate, df_telco DataFrames.
    """
    train_validate, df_telco = train_test_split(
        df, test_size=0.2, random_state=123, stratify=df[target]
    )
    train, validate = train_test_split(
        train_validate, test_size=0.3, random_state=123, stratify=train_validate[target]
    )
    print(f"train: {len(train)} ({round(len(train)/len(df), 2)*100}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df), 2)*100}% of {len(df)})"
    )
    print(
        f"df_telco: {len(df_telco)} ({round(len(df_telco)/len(df), 2)*100}% of {len(df)})"
    )
    return train, validate, df_telco


def get_telco_data():
    """
    get telco data will query the telco database and return all the relevant churn data within

    arguments: none

    return: a pandas dataframe
    """
    filename = "telco.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
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
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df


def pull_prep_telco(binary=False):
    """
    This function prepares the telco dataset for classification exercises.

        It drops unnecessary columns,
        converts the 'total_charges' column to numeric,
        separates automatic payments from payment types,
        changes senior citizen to Yes/No,
        calculates tenure in months and years,
        changes True/False values to Yes/No,
        creates a new column 'total_add_ons' that counts all of the add ons.
    """

    # Use CSV or pull data from SQL
    df_telco = get_telco_data()

    # Drop the joiner columns from SQL and definitely useless columns
    df_telco = df_telco.drop(
        columns=[
            "customer_id",
            "internet_service_type_id",
            "contract_type_id",
            "payment_type_id",
            "contract_type_id.1",
            "internet_service_type_id.1",
            "payment_type_id.1",
        ]
    )

    # As we've noticed before, ['total_charges'] is detected as an object, but holds mainly numbers. We'll use pd.to_numeric() to force everything to a number,
    # and anything that it cannot convert, we'll make it Null and investigate those.
    df_telco["total_charges"] = pd.to_numeric(
        df_telco["total_charges"], errors="coerce"
    )
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

    # Calculate tenure in months
    df_telco["tenure (months)"] = df_telco["tenure"] * 12

    # Bin the tenure into groups of 12 to separate by year
    df_telco["tenure (years)"] = df_telco["tenure"].apply(
        lambda x: int((x - 1) / 12) + 1
    )

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

    # Replace no X service with None to not have them represented in the data
    df_telco.replace(
        {"No internet service": None, "No phone service": None}, inplace=True
    )
    if binary == False:
        return df_telco
    else:
        df_telco.replace(
            {"Male": 1, "Female": 0, "Yes": 1, "No": 0, "No internet service": None},
            inplace=True,
        )
        df_telco[
            [
                "multiple_lines",
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
            ]
        ] = df_telco[
            [
                "multiple_lines",
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
            ]
        ].astype(
            "Int64"
        )
        return df_telco


def check_columns(df):
    """
    This function takes in a pandas DataFrame and prints out the name of each column,
    the number of unique values in each column, and the unique values themselves.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to be checked.
    """
    for column in df.columns:
        print(f"{column} ({df[column].nunique()})")
        print(f"Unique Values: {df[column].unique()}")
        print(f"Null Values: {df[column].isna().sum()}")
        print("")
