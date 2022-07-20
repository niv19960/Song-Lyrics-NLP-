# Importing the pandas library and giving it the alias pd.
import json

import matplotlib
import pandas as pd

# Importing the mean function from the numpy library.
from numpy import mean

# Importing the necessary libraries for the model.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

# Importing the standard error of the mean function from the scipy library.
from scipy.stats import sem

# Importing the pyplot module from the matplotlib library.
from matplotlib import pyplot, pyplot as plt

# Importing the seaborn library and giving it the alias sns.
import seaborn as sns


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
    model = RandomForestClassifier()
    # evaluate model
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def crossValidation(x, y):
    """
    It takes the X and Y values as input and then runs the evaluate_model function for different values of repeats.

    The evaluate_model function is defined below.

    :param x: The input data
    :param y: The target variable
    """

    # configurations to test
    repeats = range(1, 6)
    results = list()
    print("Cross Validation scores (Mean & Standard Deviation)")
    for r in repeats:
        # evaluate using a given number of repeats
        scores = evaluate_model(x, y, r)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)

    # Writing Json data into a file
    with open('Random forest Mean & Std.txt', 'w') as outfile:
        json.dump(results, outfile)

    # plot the results
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()


def HyperRF(x_train, x_test, y_train, y_test):
    """
    The function takes in the training and testing data and returns the best classifier

    :param x_train: The training data
    :param x_test: The test data
    :param y_train: The training data's labels
    :param y_test: The actual values of the target variable
    :return: The best classifier
    """
    max_depth = [None, 32, 64]
    n_estimators = [100, 128, 256]

    # bootstrap= [True, False]
    # max_depth= [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    # max_features= ['auto', 'sqrt']
    # min_samples_leaf= [1, 2, 4]
    # min_samples_split= [2, 5, 10]
    # n_estimators= [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

    # Build the grid search
    dfrst = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, cv=10)
    grid_results = grid.fit(x_train, y_train)

    # Summarize the results in a readable format
    print("Best: {0}, using {1}".format(grid_results.cv_results_['mean_test_score'], grid_results.best_params_))
    results_df = pd.DataFrame(grid_results.cv_results_)
    print(results_df)
    results_df.to_csv('Grid Search Results.csv')

    # Writing Json data into a file
    with open('Random forest Best Parameters.txt', 'w') as outfile:
        json.dump(grid_results.best_params_, outfile)

    # Extract the best decision forest
    best_clf = grid_results.best_estimator_
    y_pred = best_clf.predict(x_test)

    # Create a confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # Create heatmap from the confusion matrix
    class_names = [False, True]  # name  of classes
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = [0.5, 1.5]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    return best_clf


def evaluation(model, x_test, y_test):
    """
    It takes the model, the test data and the test labels as input and returns the metrics of the model

    :param model: The model object
    :param x_test: The test data
    :param y_test: The actual values of the target variable
    """

    # predict probabilities on Test and take probability for class 1([:1])
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    y_predict = model.predict(x_test)

    df_metrics = pd.DataFrame([])
    metrics = {}
    metrics['Roc-Auc'] = roc_auc_score(y_test, y_pred_prob)
    metrics['accuracy'] = accuracy_score(y_test, y_predict)
    metrics['precision'] = precision_score(y_test, y_predict)
    metrics['recall'] = recall_score(y_test, y_predict)
    metrics['f1'] = f1_score(y_test, y_predict, average='macro')
    df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics, index=['Random forest']).T], axis=1)

    df_metrics.to_csv("5_Random forest Results After Fine Tuning.csv")


def fineTuning():
    """
    This function takes in the dataframe created from the csv file, splits it into training and testing data, and then uses
    the hyperparameter tuning function to find the best parameters for the random forest model
    """
    file_name = '3_Optimization Data.csv'
    RF_data = createDF(file_name)

    x = RF_data.drop(columns='Mood', axis=1)
    y = RF_data['Mood']

    crossValidation(x, y)
    print("step 1")

    x_train = pd.read_csv("x_train.csv")
    x_test = pd.read_csv("x_test.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")

    model = HyperRF(x_train, x_test, y_train, y_test)

    evaluation(model, x_test, y_test)

