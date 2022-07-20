# Song-Lyrics-NLP-
Sentiment classification in music is a common problem with numerous applications. Many companies in many fields are interested in satisfying their audience with the right music. One central feature is the positiveness of a song. In this project, we try to create a model which predicts if a song's lyric is positive or not.
<br>
Valence: Describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).


# Repo intro
Valence prediction from lyrics- building a classifier that predicts whether a song is positive or negative.


## Classic model:
Please activate the 'StandAloneModel' with the '3_Optimization Data.csv' file.
All results (before and after fine tuning) are in results file.

## Neural Network model:
Please activate the 'NNModel' with the '2_ML data.csv' file. Loading the weights from model file. The results are in the results file. 
<br>
Neural Network for lyrics mood Binary- Classification in Keras.
The approach is to use a Neural Network which uses pre-trained GloVe embeddings, bidirectional-LSTM layer to classify whether the song is happy or sad.
# Installation | Requirements

Enviornment: PyCharm | Jupyter Notebook
<br>
Python = 3.9
<br>
pandas~=1.4.3
<br>
scikit-learn~=1.1.1
<br>
matplotlib~=3.5.2
<br>
nltk~=3.7
<br>
numpy~=1.23.0
<br>
scipy~=1.8.1
<br>
TensorFlow = V2
<br>
keras~=2.9.0
<br>
Glove

# Quickstart


# Resources
https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence
