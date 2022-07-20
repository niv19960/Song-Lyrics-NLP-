# Import libraries
# Data structures
import pandas as pd
import numpy as np

# Pre-processing and Tokenization
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

# Embedding
# Using GloVe Embeddings
# download glove and unzip it.
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip

import zipfile

with zipfile.ZipFile('/tmp/glove.6B.zip', 'r') as zip_ref:
    zip_ref.extractall('/tmp/glove')


# Build the model
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger

# Plot results
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sn

# Load the model
import tensorflow as tf
from keras.models import load_model


# Calling all the functions in the order they need to be called.
def neuralNetwork():
    file_name = '2_ML data'
    ml_data = createDF(file_name)
    X_data_seq_padded, y_data, word_index, max_num_tokens = processTrainingData(ml_data)
    vocab_num_words, embedding_dim, embedding_matrix = preTrainedGloVeEmbedding(word_index)
    X_train_pad, X_test_pad, y_train, y_test = trainTestSplit(X_data_seq_padded, y_data)
    model, history, X_test_pad, y_test = SecModel(X_train_pad, X_test_pad, y_train, y_test, vocab_num_words,
                                                  embedding_dim, embedding_matrix, max_num_tokens)
    plot_model(model, show_shapes=True, rankdir="LR")
    plotTrainingModelaccuracy(history)
    plotConfusionMatrix(model, X_test_pad, y_test)
    saveModelInfo(model, history)


def createDF(file_name):
    """
    It reads the csv file, renames the columns, creates a binary column for the mood, and creates a column for the
    length of the lyrics

    :param file_name: the name of the file you want to import
    :return: A dataframe with the columns: lyrics, song name, valence, mood, length
    """
    # Import CSV file
    df = pd.read_csv(file_name, usecols=['lyrics', 'Mood'])

    return (df)


def processTrainingData(df):
    """
    It takes in a dataframe of lyrics and their corresponding moods, and returns a list of tokenized lyrics, a list of
    moods, a dictionary of words and their corresponding integer values, and the maximum number of tokens in a lyric

    :param df: the dataframe containing the lyrics and moods
    :return: X_data_seq_padded is a list of lists of integers, where each integer represents a word token.
    y_data is a list of integers, where each integer represents a mood.
    tokenizer.word_index is a dictionary, where each key is a word token and each value is an integer.
    max_num_tokens is an integer representing the maximum
    """
    # Initializes training data, stop words list and translator for stripping punctuation
    X_data = []
    y_data = df["Mood"]
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)

    for lyric in df.lyrics:
        # Tokenize each lyric, and set all characters to lower-case
        tokens = word_tokenize(lyric)
        tokens = [word.lower() for word in tokens]

        # Remove punctuation
        tokens_nopunc = [word.translate(translator) for word in tokens]

        # Remove non-alphabetic tokens
        words = [word for word in tokens_nopunc if word.isalpha()]

        # Remove stop words from the lyric
        words = [word for word in words if not word in stop_words]

        # Append to training data
        X_data.append(words)

    # Map each word token in the training data to an integer

    Vocab_size = 1000
    oov_token = "<OOV>"

    # For each training example, maps each word token to an integer
    tokenizer = Tokenizer(num_words=Vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_data)
    X_data_seq = tokenizer.texts_to_sequences(X_data)

    # Pad sequences shorter than max length
    max_num_tokens = max([len(tokenized_lyric) for tokenized_lyric in X_data])
    X_data_seq_padded = pad_sequences(X_data_seq, maxlen=max_num_tokens)

    return X_data_seq_padded, y_data, tokenizer.word_index, max_num_tokens


def preTrainedGloVeEmbedding(word_index):
    """
    The function takes in the word index of the tokenizer and returns the number of words in the vocabulary, the dimension
    of the embedding, and the embedding matrix

    :param word_index: The word index of the tokenizer
    :return: The number of words in the vocabulary, the dimension of the embedding, and the embedding matrix
    """
    # Loads pre-trained GloVe word embeddings

    # Load in GloVe file and initialize embedding index
    embeddings_index = {}
    file = open('/tmp/glove/glove.6B.100d.txt')
    embeddings_index = {}

    for line in file:
        # Add each embedding to the embedding index
        embedding = line.split()
        embeddings_index[embedding[0]] = np.asarray(embedding[1:])

    file.close()

    # Map GloVe word embeddings to each word in the tokenizer word index to create a matrix of word embeddings

    # Initialize embedding matrix
    embedding_dim = 100
    vocab_num_words = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_num_words, embedding_dim))

    # Populate embedding matrix
    for word, i in word_index.items():

        if i > vocab_num_words:
            continue

        # Assign corresponding GloVe embedding to the given word
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # Words not found in the embedding are assigned a zero vector by default
            embedding_matrix[i] = embedding_vector

    return vocab_num_words, embedding_dim, embedding_matrix


def trainTestSplit(X_data_seq_padded, y_data):
    """
    It shuffles the data, and splits it into train and test sets

    :param X_data_seq_padded: the padded sequences of the data
    :param y_data: the mood labels
    :return: X_train_pad, X_test_pad, y_train, y_test
    """
    # Shuffles the data, and splits it into train and test sets

    VALIDATION_SPLIT = 0.3

    # Shuffles data and labels
    word_indices = np.arange(X_data_seq_padded.shape[0])
    np.random.shuffle(word_indices)
    X_data_seq_padded = X_data_seq_padded[word_indices]
    moods = np.array(y_data)
    moods = moods[word_indices]

    # Binarizes the mood labels
    encoder = LabelBinarizer()
    moods = encoder.fit_transform(moods.tolist())

    # Splits the dataset into train and test sets
    num_validation_samples = int(VALIDATION_SPLIT * X_data_seq_padded.shape[0])
    X_train_pad = X_data_seq_padded[:-num_validation_samples]
    y_train = moods[:-num_validation_samples]

    X_test_pad = X_data_seq_padded[-num_validation_samples:]
    y_test = moods[-num_validation_samples:]
    return X_train_pad, X_test_pad, y_train, y_test


def SecModel(X_train_pad, X_test_pad, y_train, y_test, vocab_num_words, embedding_dim, embedding_matrix, max_num_tokens):
    """
    It defines a model, compiles it, and trains it

    :param X_train_pad: The padded training data
    :param X_test_pad: the padded test set
    :param y_train: the training labels
    :param y_test: the test set labels
    :param vocab_num_words: The number of words in the vocabulary
    :param embedding_dim: The dimension of the embedding matrix
    :param embedding_matrix: the embedding matrix we created earlier
    :param max_num_tokens: The maximum number of tokens in a tweet
    :return: The model, the history of the model, the X_test_pad and the y_test
    """
    # Defines the Binary model, compiles and trains it

    model = Sequential()
    embedding_layer = Embedding(vocab_num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_num_tokens,
                                trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Saves a History object for plotting
    history = model.fit(X_train_pad, y_train, batch_size=128, epochs=10, validation_data=(X_test_pad, y_test),
                        verbose=2)

    return model, history, X_test_pad, y_test


def plotConfusionMatrix(model1, X_test_pad, y_test):
    """
    It takes in a model, a test set, and the test labels, and then plots a confusion matrix

    :param model1: the model you want to plot the confusion matrix for
    :param X_test_pad: the padded test data
    :param y_test: The actual labels of the test set
    """
    # Gets predicted labels from model
    y_pred = np.around(model1.predict(X_test_pad)).astype(int).flatten()
    print(y_pred.shape)
    # Generates confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Formats and displays the confusion matrix
    figure(num=None, figsize=(3, 2), dpi=300)
    df_cm = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 13}, cmap=plt.cm.Blues, fmt='g', cbar=False)
    plt.title('Binary Classification Confusion Matrix', fontsize=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.show()


def plotTrainingModelaccuracy(history):
    """
    Plot accuracy as a function of training epoch
    """

    figure(num=None, figsize=(3, 2), dpi=300)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.grid(True)
    plt.title('Binary Classification Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def saveModelInfo(model, history):
    """
    Take the model and the history results of it as parameters.
    Save the model
    Save the results and weights of the model
    """

    # Create history df and save as csv file
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Save weights
    weight = model.get_weights()
    np.savetxt('weights.csv', weight, fmt='%s', delimiter=',')

    # Save results of the model
    filename = 'log.csv'
    history_logger = CSVLogger(filename, separator=",", append=True)

    # Save the model itself
    model.save("lyrics_mood_model")


