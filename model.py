import pandas as pd
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def create_train_validate_test(train, validate, test, target_variable):
    # We'll do exploration and train our model on the train data
    X_train = train.drop(columns=[target_variable])
    y_train = train[target_variable]

    # We tune our model on validate, since it will be out-of-sample until we use it.
    X_validate = validate.drop(columns=[target_variable])
    y_validate = validate[target_variable]

    # Keep the test separate, for our final out-of-sample dataset, to see how well our tuned model performs on new data.
    X_test = test.drop(columns=[target_variable])
    y_test = test[target_variable]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def plot_churn_proportion(train):
    # Create a figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Calculate the proportion of churn
    prop_response = train["churn"].value_counts(normalize=True)

    # Create a bar plot showing the percentage of churn
    prop_response.plot(kind="bar", ax=ax)

    # Add value counts to the top of each bar
    for i, v in enumerate(prop_response):
        ax.text(
            i - 0.1, v - 0.05, str(round(v * 100)) + "%", fontsize=20, color="black"
        )

    # Set labels and titles
    ax.set_title(
        "Proportion of Observations for 'churn' ",
        fontsize=15,
        loc="center",
    )
    ax.set_xlabel("churn", fontsize=12)
    ax.set_ylabel("proportion of observations", fontsize=12)
    ax.tick_params(rotation="auto")

    # Show the plot
    plt.show()


def knn_metrics(X_train, y_train, X_validate, y_validate, features):
    start_time = time.time()

    knnmetrics = []

    for k in range(1, 30):
        # KNN
        # model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[features], y_train)
        # Accuracies
        ytr_acc = knn.score(X_train[features], y_train).round(10)
        yv_acc = knn.score(X_validate[features], y_validate).round(10)
        # Make it into a DF
        output = {
            "model": knn,
            "features": features,
            "train_acc": ytr_acc,
            "validate_acc": yv_acc,
        }
        knnmetrics.append(output)

    end_time = time.time()

    df = pd.DataFrame(knnmetrics)
    df["avg_score"] = (df["train_acc"] + df["validate_acc"]) / 2
    df = df.sort_values(
        ["validate_acc", "train_acc"], ascending=[False, True]
    ).reset_index()

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df.train_acc, label="Train", marker="o")
    plt.plot(df.index, df.validate_acc, label="Validate", marker="o")
    plt.fill_between(df.index, df.train_acc, df.validate_acc, alpha=0.2)
    plt.xlabel("Model Number as Index in DF", fontsize=10)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Classification Model Performance: KNN", fontsize=18)
    plt.legend(title="Scores", fontsize=12)
    plt.show()

    print(f"Total Models Ran: {len(df)}")
    print("Processing Time:", round(end_time - start_time, 2), "seconds")
    print("--------------------------------")
    print(f"Best KNN Hyperparameters:")
    print(df.iloc[0]["model"])
    print("--------------------------------")
    print(f"Best KNN Average Accuracy:")
    print(df.iloc[0]["avg_score"])


def logistic_regression(X_train, y_train, X_validate, y_validate, features):
    start_time = time.time()

    logregmetrics = []

    # Define hyperparameters
    C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalty = ["l2"]
    solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]

    # Generate all possible combinations of hyperparameters
    hyperparameters = list(itertools.product(C, penalty, solver))

    for hyperparameter in hyperparameters:
        # Logistic Regression
        # model
        logreg = LogisticRegression(
            C=hyperparameter[0],
            penalty=hyperparameter[1],
            solver=hyperparameter[2],
            random_state=1,
        )
        logreg.fit(X_train[features], y_train)
        # Accuracies
        ytr_acc = logreg.score(X_train[features], y_train).round(10)
        yv_acc = logreg.score(X_validate[features], y_validate).round(10)
        # Make it into a DF
        output = {
            "model": logreg,
            "features": features,
            "train_acc": ytr_acc,
            "validate_acc": yv_acc,
            "hyperparameters": hyperparameter,
        }
        logregmetrics.append(output)

    end_time = time.time()

    df = pd.DataFrame(logregmetrics)
    df["avg_score"] = (df["train_acc"] + df["validate_acc"]) / 2
    df = df.sort_values(
        ["validate_acc", "train_acc"], ascending=[False, True]
    ).reset_index()

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df.train_acc, label="Train", marker="o")
    plt.plot(df.index, df.validate_acc, label="Validate", marker="o")
    plt.fill_between(df.index, df.train_acc, df.validate_acc, alpha=0.2)
    plt.xlabel("Model Number as Index in DF", fontsize=10)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Classification Model Performance: LR", fontsize=18)
    plt.legend(title="Scores", fontsize=12)
    plt.show()
    print(
        "For this we used SK Learn itertools to run through iterations of hyperparamters."
    )
    print("In this specific model, we tuned C level and solver.")
    print("As it cycled through the iterations and it hit different hyperparameters,")
    print("you can notice a dips in performance with a big drop off at model 30.")
    print("")
    print(f"Total Models Ran: {len(df)}")
    print("Processing Time:", round(end_time - start_time, 2), "seconds")
    print("--------------------------------")
    print(f"Best LR Hyperparameters:")
    print("C=0.1 - lower values help to prevent overfitting")
    print("Solver=liblinear - uses coordinate descent to optimize cost function")
    print("--------------------------------")
    print(f"Best LR Average Accuracy:")
    print(df.iloc[0]["avg_score"])
    print("")
    print("----------------------------------------------------------------")
    print("")
    print("Severe Drop Off at Model 30")
    print("Model 30 Hyperparameters:")
    print("C=0.0001 - drop off was most likely due to strong regularization of data")
    print("Solver='newton-cg'")


def random_forest(X_train, y_train, X_validate, y_validate, features):
    start_time = time.time()

    rfmetrics = []

    # Define hyperparameters
    n_estimators = [10, 50, 100, 200, 500]
    max_depth = [None, 5, 10, 20, 50]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    # Generate all possible combinations of hyperparameters
    hyperparameters = list(
        itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    )

    for hyperparameter in hyperparameters:
        # Random Forest
        # model
        rf = RandomForestClassifier(
            n_estimators=hyperparameter[0],
            max_depth=hyperparameter[1],
            min_samples_split=hyperparameter[2],
            min_samples_leaf=hyperparameter[3],
            random_state=1,
        )
        rf.fit(X_train[features], y_train)
        # Accuracies
        ytr_acc = rf.score(X_train[features], y_train).round(10)
        yv_acc = rf.score(X_validate[features], y_validate).round(10)
        # Make it into a DF
        output = {
            "model": rf,
            "features": features,
            "train_acc": ytr_acc,
            "validate_acc": yv_acc,
            "hyperparameters": hyperparameter,
        }
        rfmetrics.append(output)

    end_time = time.time()

    df = pd.DataFrame(rfmetrics)
    df["avg_score"] = (df["train_acc"] + df["validate_acc"]) / 2
    df = df.sort_values(
        ["validate_acc", "train_acc"], ascending=[False, True]
    ).reset_index()

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df.train_acc, label="Train", marker="o")
    plt.plot(df.index, df.validate_acc, label="Validate", marker="o")
    plt.fill_between(df.index, df.train_acc, df.validate_acc, alpha=0.2)
    plt.xlabel("Model Number as Index in DF", fontsize=10)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Classification Model Performance: Random Forest", fontsize=18)
    plt.legend(title="Scores", fontsize=12)
    plt.show()
    print(
        "Hyperparameters tuned n_estimators, max_depth, min_samples_split, min_samples_leaf"
    )
    print("As the hyperparamters got larger, the overfitting was more apparent.")
    print("")
    print(f"Total Models Ran: {len(df)}")
    print("Processing Time:", round(end_time - start_time, 2), "seconds")
    print("--------------------------------")
    print(f"Best Random Forest Hyperparameters:")
    print("   n_estimators=10")
    print("   max_depth=5")
    print("   min_samples_split=2")
    print("   min_samples_leaf=2")
    print("--------------------------------")
    print(f"Best Random Forest Average Accuracy:")
    print(df.iloc[0]["avg_score"])


def decision_tree(X_train, y_train, X_validate, y_validate, features):
    start_time = time.time()

    dtmetrics = []

    # Define hyperparameters
    criterion = ["gini", "entropy"]
    splitter = ["best", "random"]
    max_depth = [None, 5, 10, 20, 50]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    # Generate all possible combinations of hyperparameters
    hyperparameters = list(
        itertools.product(
            criterion, splitter, max_depth, min_samples_split, min_samples_leaf
        )
    )

    for hyperparameter in hyperparameters:
        # Decision Tree
        # model
        dt = DecisionTreeClassifier(
            criterion=hyperparameter[0],
            splitter=hyperparameter[1],
            max_depth=hyperparameter[2],
            min_samples_split=hyperparameter[3],
            min_samples_leaf=hyperparameter[4],
            random_state=1,
        )
        dt.fit(X_train[features], y_train)
        # Accuracies
        ytr_acc = dt.score(X_train[features], y_train).round(10)
        yv_acc = dt.score(X_validate[features], y_validate).round(10)
        # Make it into a DF
        output = {
            "model": dt,
            "features": features,
            "train_acc": ytr_acc,
            "validate_acc": yv_acc,
            "hyperparameters": hyperparameter,
        }
        dtmetrics.append(output)

    end_time = time.time()

    df = pd.DataFrame(dtmetrics)
    df["avg_score"] = (df["train_acc"] + df["validate_acc"]) / 2
    df = df.sort_values(
        ["validate_acc", "train_acc"], ascending=[False, True]
    ).reset_index()

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df.train_acc, label="Train", marker="o")
    plt.plot(df.index, df.validate_acc, label="Validate", marker="o")
    plt.fill_between(df.index, df.train_acc, df.validate_acc, alpha=0.2)
    plt.xlabel("Model Number as Index in DF", fontsize=10)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Classification Model Performance: Decision Tree", fontsize=18)
    plt.legend(title="Scores", fontsize=12)
    plt.show()
    print("For DecisionTree we tuned criterion, splitter, max_depth, min_samples_split")
    print("min_samples_leaf")
    print("")
    print(f"Total Models Ran: {len(df)}")
    print("Processing Time:", round(end_time - start_time, 2), "seconds")
    print("--------------------------------")
    print(f"Best Decision Tree Hyperparameters:")
    print(df.iloc[0]["hyperparameters"])
    print("--------------------------------")
    print(f"Best Decision Tree Average Accuracy:")
    print(df.iloc[0]["avg_score"])


def best_rfc_test(X_train, y_train, X_test, y_test, features):
    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=1,
    )
    rf.fit(X_train[features], y_train)
    t = rf.score(X_test[features], y_test)
    print("Random Forest Model on test set:")
    print(f"Accuracy on Test: {round(t*100,5)}")
