# from env import db_url
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# import pandas as pd


# def grab_sql(db, query, filename):
#     """
#     Grab data from Codeup's MySQL server utilizing db_url with credentials from a personal env file.
#     Check if filename file exists. If it does not exist, it will create one.

#     This function selects ALL data from a table with no limit on size.

#     Args:
#         db (str): The name of the database.
#         table (str): The name of the table within that database.
#         filename (str): The name of the file to be created.
#         query (str, optional): The SQL query to execute. Defaults to "SELECT * FROM <table>".

#     Returns:
#         pandas.DataFrame: A pandas dataframe containing the data from the selected table.
#     """
#     connection = db_url(db)
#     df = pd.read_sql(query, connection)
#     df.to_csv(filename, index=False)
#     return df


# # 1. Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame.
# # Obtain your data from the Codeup Data Science Database.


# # def get_titanic_data():
# #     """
# #     This function retrieves the titanic data from the 'passengers' table in the 'titanic_db' database and returns it as a pandas DataFrame.
# #     """
# #     query = "select * from passengers"
# #     df = grab_sql("titanic_db", query, "titanic.csv")
# #     return df


# # 2. Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame.
# # The returned data frame should include the actual name of the species in addition to the species_ids.
# # Obtain your data from the Codeup Data Science Database.


# # def get_iris_data():
# #     """
# #     This function retrieves the iris dataset from a SQL database and returns it as a pandas DataFrame.

# #     Returns:
# #     - df (pandas DataFrame): The iris dataset with species names and measurements.
# #     """
# #     query = """
# #         SELECT species.species_name, measurements.*
# #         FROM measurements
# #         JOIN species
# #         ON measurements.species_id = species.species_id;"""

# #     df = grab_sql("iris_db", query, "iris.csv")
# #     return df


# # 3. Make a function named get_telco_data that returns the data from the telco_churn database in SQL.
# # In your SQL, be sure to join contract_types, internet_service_types, payment_types tables with the customers table,
# # so that the resulting dataframe contains all the contract, payment, and internet service options.
# # Obtain your data from the Codeup Data Science Database.


# # def get_telco_data():
# #     """
# #     This function retrieves data from the 'customers' table in the 'telco_churn' database, and joins it with the 'contract_types', 'internet_service_types', and 'payment_types' tables. The resulting dataframe contains information about customers' contracts, internet service types, payment types, and other relevant information.

# #     Returns:
# #     - df: a pandas DataFrame containing the joined data from the 'customers', 'contract_types', 'internet_service_types', and 'payment_types' tables.
# #     """
# #     query = """
# #         SELECT *
# #         FROM customers
# #         JOIN contract_types
# #         ON customers.contract_type_id = contract_types.contract_type_id
# #         JOIN internet_service_types
# #         ON customers.internet_service_type_id = internet_service_types.internet_service_type_id
# #         JOIN payment_types
# #         ON customers.payment_type_id = payment_types.payment_type_id;"""

# #     df = grab_sql("telco_churn", query, "telco.csv")
# #     return df


# # 4. Once you've got your get_titanic_data, get_iris_data, and get_telco_data functions written, now it's time to add caching to them.
# # To do this, edit the beginning of the function to check for the local filename of telco.csv, titanic.csv, or iris.csv.
# # If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe,
# # then write the dataframe to a .csv file with the appropriate name.


# def get_telco_data():
#     """
#     get telco data will query the telco database and return all the relevant churn data within

#     arguments: none

#     return: a pandas dataframe
#     """
#     filename = "telco.csv"
#     if os.path.isfile(filename):
#         df = pd.read_csv(filename)
#     else:
#         query = """
#         SELECT *
#         FROM customers
#         JOIN contract_types
#         ON customers.contract_type_id = contract_types.contract_type_id
#         JOIN internet_service_types
#         ON customers.internet_service_type_id = internet_service_types.internet_service_type_id
#         JOIN payment_types
#         ON customers.payment_type_id = payment_types.payment_type_id;"""
#         connection = db_url("telco_churn")
#         df = pd.read_sql(query, connection)
#         df.to_csv(filename, index=False)
#     return df


# def get_iris_data():
#     """
#     get iris will query the iris database and return all the data within

#     arguments: none

#     return: a pandas dataframe
#     """
#     filename = "iris.csv"
#     if os.path.isfile(filename):
#         df = pd.read_csv(filename)
#     else:
#         query = """
#         SELECT species.species_name, measurements.*
#         FROM measurements
#         JOIN species
#         ON measurements.species_id = species.species_id;"""
#         connection = db_url("iris_db")
#         df = pd.read_sql(query, connection)
#         df.to_csv(filename, index=False)
#     return df


# def get_titanic_data():
#     """
#     get titanic data will query the titanic database and return all the data within

#     arguments: none

#     return: a pandas dataframe
#     """
#     filename = "titanic.csv"
#     if os.path.isfile(filename):
#         df = pd.read_csv(filename)
#     else:
#         query = "SELECT * FROM passengers"
#         connection = db_url("titanic_db")
#         df = pd.read_sql(query, connection)
#         df.to_csv(filename, index=False)
#     return df


# def split_data(df, target):
#     """
#     take in a DataFrame and return train, validate, and test DataFrames; stratify on a specified variable.
#     return train, validate, test DataFrames.
#     """
#     train_validate, test = train_test_split(
#         df, test_size=0.2, random_state=123, stratify=df[target]
#     )
#     train, validate = train_test_split(
#         train_validate, test_size=0.3, random_state=123, stratify=train_validate[target]
#     )
#     print(f"train: {len(train)} ({round(len(train)/len(df), 2)*100}% of {len(df)})")
#     print(
#         f"validate: {len(validate)} ({round(len(validate)/len(df), 2)*100}% of {len(df)})"
#     )
#     print(f"test: {len(test)} ({round(len(test)/len(df), 2)*100}% of {len(df)})")

#     return train, validate, test
