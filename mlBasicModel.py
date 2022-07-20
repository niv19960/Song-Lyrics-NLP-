# It imports the pandas library and renames it to pd.
import pandas as pd

# Importing the train_test_split, TfidfVectorizer, and Pipeline functions from the sklearn library.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Importing the models that we are going to use.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB

# Importing the functions that we are going to use to evaluate the models.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix

# It imports the pyplot module from the matplotlib library and renames it to plt.
from matplotlib import pyplot as plt


def createDF(file_name):
    """
    It reads the csv file, renames the columns, creates a binary column for the mood, and creates a column for the
    length of the lyrics

    :param file_name: the name of the file you want to import
    :return: A dataframe with the columns: lyrics, song name, valence, mood, length
    """
    # Import CSV file
    df = pd.read_csv(file_name, usecols=['lyrics', 'Mood'])

    return df


def TFIDF(df):
    """
    It takes in a dataframe, and returns a dataframe with the lyrics column removed, and replaced with a TFIDF vectorized
    version of the lyrics column.

    :param df: the dataframe that contains the lyrics' column
    :return: A dataframe with the lyrics vectorized and the mood and length_log columns.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    x = vectorizer.fit_transform(df['lyrics'])

    print(vectorizer.get_feature_names_out())

    vectorizer_df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    df.drop('lyrics', axis=1, inplace=True)  # Consist of 'mood' & 'length_log'
    result = pd.concat([df, vectorizer_df], axis=1)

    return result


def trainTestSplit(df):
    """
    It takes in a dataframe, drops the 'Mood' column, and then splits the dataframe into training and testing data

    :param df: the dataframe
    :return: X_raw_train, X_raw_test, y_train, y_test
    """
    x = df.drop('Mood', 1)
    y = df['Mood']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    x_train.to_csv("x_train.csv")
    x_test.to_csv("x_test.csv")
    y_train.to_csv("y_train.csv")
    y_test.to_csv("y_test.csv")

    return x_train, x_test, y_train, y_test


def trainModel(model, x_train, y_train):
    """
    It takes in a model, x_train, and y_train, and returns a trained model

    :param model: the model to be trained
    :param x_train: The training data
    :param y_train: The labels of the training data
    :return: The classifier is being returned.
    """

    classifier = Pipeline([('clf', model)])
    classifier.fit(x_train, y_train)

    return classifier


def confusionMatrix(classifier, x_test, y_test):
    """
    It takes a classifier, a test set, and a test set's labels, and plots a confusion matrix

    :param classifier: the classifier object
    :param x_test: The test data
    :param y_test: The actual labels of the test data
    """

    plot_confusion_matrix(classifier, x_test, y_test)
    plt.show()


def basicModelPipeline():  # (ml_data):
    """
    It takes the dataframe created in the previous step, performs TFIDF on it, splits it into train and test sets, and then
    trains and tests 8 different models on the data
    """

    file_name = '2_ML Data.csv'
    ml_data = createDF(file_name)
    df_after_TFIDF = TFIDF(ml_data)
    df_after_TFIDF.to_csv("3_Optimization Data.csv")
    x_train, x_test, y_train, y_test = trainTestSplit(df_after_TFIDF)

    ml_models = {
        'LogReg': LogisticRegression(),
        'LinearSVC': LinearSVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Naive Bayes': MultinomialNB()
    }

    df_metrics = pd.DataFrame([])

    for model in ml_models:
        classifier = trainModel(ml_models[model], x_train, y_train)
        y_predict = classifier.predict(x_test)

        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_predict)
        metrics['precision'] = precision_score(y_test, y_predict)
        metrics['recall'] = recall_score(y_test, y_predict)
        metrics['f1'] = f1_score(y_test, y_predict, average='macro')
        df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics, index=[model]).T], axis=1)

    print(df_metrics)
    df_metrics.to_csv("4_Models Results Before Fine Tuning.csv")
