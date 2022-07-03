# DataFrame
import nltk
import pandas as pd
import numpy as np

# NLP
import string
import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def length(text):
    """
    "Return the length of the text."

    The function takes a single argument, text, and returns the length of the text

    :param text: The text to be analyzed
    :return: The length of the text.
    """
    return len(text)


def characterSubstitute(text):
    """
    It takes a string, removes all newline characters, removes all instances of the string "Hook 1", and replaces all
    multiple spaces with a single space

    :param text: the text to be processed
    :return: the text with the newlines, carriage returns, and Hook 1 removed.
    """
    regex = r'\n|\r|Hook 1|[0-9]|chorus'
    text = re.sub(regex, " ", text)
    text = re.sub(' +', ' ', text)
    return text


def removePunctuation(text):
    """
    * The function takes in a string and returns a new string which doesn't contain any punctuation.

    * For example, calling the function with the string `"Let's try, Mike."` should return `"Lets try Mike"`

    :param text: The text whose punctuations are to be removed
    :return: The text stripped of punctuation marks
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def applyStopWords(text):
    """a function for removing the stopword"""
    stop_words = ['a', 'an', 'above', 'and', 'any', 'as', 'at', 'of', 'that', 'the', 'to']
    # sw = stopwords.words('english')
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    # joining the list of words with space separator
    return " ".join(text)


def stemming(text):
    """
    It takes a string of text, splits it into words, and then returns a string of text where each word is stemmed

    :param text: The text that you want to stem
    :return: the stemmed words in the text.
    """
    stemmer = SnowballStemmer("english")
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)


def createDF(file_name):
    """
    It reads the csv file, renames the columns, creates a binary column for the mood, and creates a column for the
    length of the lyrics

    :param file_name: the name of the file you want to import
    :return: A dataframe with the columns: lyrics, song name, valence, mood, length
    """
    # Import CSV file
    df = pd.read_csv(file_name, sep=',', index_col=[0])

    # Rename columns names: "seq": "lyrics", "song": "song name", "label":"valence"
    df.rename(columns={"seq": "lyrics", "song": "song name", "label": "valence"}, inplace=True)

    # Create binary column: 1 represent "happy" mood while 0 represent "sad column"
    df['Mood'] = np.where(df['valence'] > 0.5, 1, 0)

    # Create 'length' column that represent the lyrics' number of words
    df['length'] = df['lyrics'].apply(length)
    df['length_log'] = np.log(df['length'])

    return df


def dataCleansing(df):
    """
    It takes in a dataframe, cleanses the lyrics' column, and returns a dataframe.

    :param df: the dataframe that contains the lyrics
    :return: A dataframe with the lyrics' column cleaned.
    """
    # Substitute special regex/characters
    df['lyrics'] = df['lyrics'].apply(characterSubstitute)

    # Remove punctuation
    df['lyrics'] = df['lyrics'].apply(removePunctuation)

    # Lowercase all words
    df['lyrics'] = df['lyrics'].apply(lambda x: x.lower())

    # Keep song with lyrics length between 500 and 2000
    df = df[(df['length'] < 2000) & (df['length'] > 500)]

    # Drop Duplicates
    df = df.drop_duplicates(subset=['lyrics'])  # , inplace=True)

    # Remove StopWords
    df['lyrics'] = df['lyrics'].apply(applyStopWords)

    # Stemming
    df['lyrics'] = df['lyrics'].apply(stemming)

    return df


def downSampling(df):
    """
    It takes a dataframe as input, and returns a dataframe with the
    same number of rows as the input dataframe, but with the same number of rows for each class

    :param df: The dataframe you want to down sample
    :return: A dataframe with the same number of negative and positive moods.
    """
    requires_n = df['Mood'].value_counts().min()
    negative_mood = df[df['Mood'] == 0].sample(n=requires_n)
    positive_mood = df[df['Mood'] == 1].sample(n=requires_n)

    down_sampling_data = pd.concat([negative_mood, positive_mood])

    # The frac keyword argument specifies the fraction of rows to return to the random sample, so frac=1 means to
    # return all rows (in random order).
    down_sampling_data = down_sampling_data.sample(frac=1)
    return down_sampling_data


def handlingData():
    """
    It takes the raw data, cleans it, and down sample it
    :return: ml_data
    """
    file_name = '1_First Data.csv'
    data = createDF(file_name)
    cleaned_data = dataCleansing(data)
    ml_data = downSampling(cleaned_data)
    ml_data.to_csv("2_ML data.csv")

    return ml_data
