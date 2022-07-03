# Importing the pandas library and giving it the alias pd.
import json

import pandas as pd

# Importing the numpy library and giving it the alias np.
import numpy as np
# Importing the mean function from the numpy library.
from numpy import mean

# Importing the necessary libraries for the model.
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Importing the standard error of the mean function from the scipy library.
from scipy.stats import sem

# Importing the pyplot module from the matplotlib library.
from matplotlib import pyplot


def createDF(file_name):
    """
    It reads the csv file, renames the columns, creates a binary column for the mood, and creates a column for the
    length of the lyrics

    :param file_name: the name of the file you want to import
    :return: A dataframe with the columns: lyrics, song name, valence, mood, length
    """
    # Import CSV file
    df = pd.read_csv(file_name)

    return df


# evaluate a model with a given number of repeats
def evaluate_model(x, y, repeats):
    """
    It creates a logistic regression model and evaluates it using repeated k-fold cross-validation

    :param x: The input data
    :param y: The target variable
    :param repeats: the number of times to repeat the cross-validation procedure
    :return: The accuracy of the model
    """
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
    # create model
    model = LogisticRegression()
    # evaluate model
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def crossValidation(x, y):
    """
    It will evaluate the model using a given number of repeats and summarize the results

    :param x: The input data
    :param y: The target variable
    """
    # configurations to test
    repeats = range(1, 10)
    results = list()
    print("Cross Validation scores (Mean & Standard Deviation")
    for r in repeats:
        # evaluate using a given number of repeats
        scores = evaluate_model(x, y, r)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)

    # plot the results
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()


def hyperparameterTuning(x_raw_train, y_train):
    """
    It takes in the training data and the target variable, and returns the best hyperparameters for the model

    :param x_raw_train: the training data
    :param y_train: the target variable
    :return: The best hyperparameters for the model
    """
    # define model/create instance
    lr = LogisticRegression()
    # tuning weight for minority class then weight for majority class will be 1-weight of minority class
    # Setting the range for class weights
    weights = np.linspace(0.0, 0.99, 500)
    # specifying all hyperparameters with possible values
    param = {'C': [0.1, 0.5, 1, 10, 15, 20], 'penalty': ['l1', 'l2'],
             "class_weight": [{0: x, 1: 1.0 - x} for x in weights]}
    # create 5 folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Gridsearch for hyperparam tuning
    model = GridSearchCV(estimator=lr, param_grid=param, scoring="f1", cv=folds, return_train_score=True)
    # train model to learn relationships between x and y
    model.fit(x_raw_train, y_train)

    # print best hyperparameters
    print("Best F1 score: ", model.best_score_)

    # Writing Json data into a file
    with open('Logistic Regression Scores.txt', 'w') as outfile:
        json.dump(model.best_score_, outfile)

    print("Best hyperparameters: ", model.best_params_)

    # Writing Json data into a file
    with open('Logistic Regression Best Parameters.txt', 'w') as outfile:
        json.dump(model.best_params_, outfile)

    return model.best_params_


def modelAfterFineTuning(best_params, x_raw_train, y_train):
    """
    It takes the best parameters from the grid search, and uses them to build a new model

    :param best_params: {'C': 0.1, 'class_weight': 'balanced', 'penalty': 'l1'}
    :param x_raw_train: The raw training data
    :param y_train: The target variable
    :return: The model after fine tuning
    """
    # Building Model again with best params
    lr2 = LogisticRegression(class_weight=best_params['class_weight'], C=best_params['C'],
                             penalty=best_params['penalty'])
    lr2.fit(x_raw_train, y_train)

    return lr2


def evaluation(lr2, x_raw_test, y_test):
    """
    It takes the model, the test data, and the test labels as input and prints out the confusion matrix, ROC-AUC score,
    accuracy score, precision score, recall score, and f1 score.

    :param lr2: the model
    :param x_raw_test: The test data
    :param y_test: The actual values of the target variable
    """
    # predict probabilities on Test and take probability for class 1([:1])
    y_pred_prob = lr2.predict_proba(x_raw_test)[:, 1]
    y_predict = lr2.predict(x_raw_test)

    df_metrics = pd.DataFrame([])
    metrics = {}
    metrics['Roc-Auc'] = roc_auc_score(y_test, y_pred_prob)
    metrics['accuracy'] = accuracy_score(y_test, y_predict)
    metrics['precision'] = precision_score(y_test, y_predict)
    metrics['recall'] = recall_score(y_test, y_predict)
    metrics['f1'] = f1_score(y_test, y_predict, average='macro')
    df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics, index=['Logistic Regression']).T], axis=1)

    df_metrics.to_csv("5_Logistic Regression Results After Fine Tuning.csv")


def fineTuning():
    """
    This function takes in the data from the csv file, splits it into training and testing data, and then uses the training
    data to find the best parameters for the model. It then uses the best parameters to create a new model and evaluates it
    using the testing data.
    """
    file_name = '3_Optimization Data.csv'
    lg_data = createDF(file_name)

    x = lg_data.drop('Mood', 1)
    y = lg_data['Mood']

    crossValidation(x, y)

    x_raw_train, x_raw_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    best_params = hyperparameterTuning(x_raw_train, y_train)
    lr2 = modelAfterFineTuning(best_params, x_raw_train, y_train)
    evaluation(lr2, x_raw_test, y_test)
