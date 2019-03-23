#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:44:27 2019

@author: roope
"""
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# # # # IMBD Large Movie Review Dataset - building LSTM sentiment classification model # # # #

# # # # LOADING DATA # # # #
import os
import numpy as np

# Directory where the IMBD large movie review dataset lays
root_folder = "/home/roope/projects/datasets/movie_reviews/aclImdb_v1/aclImdb"
train_folder_neg = os.path.join(root_folder, "train","neg")
train_folder_pos = os.path.join(root_folder, "train","pos")

test_folder_neg = os.path.join(root_folder, "test", "neg")
test_folder_pos = os.path.join(root_folder, "test","pos")

# txt files in a specified folder to python list
def files_to_list(directory=None):
    filenames = os.listdir(directory)
    # creating list where the content of txt-files will be stored
    file_contents = []    
    for filename in filenames:
        path = os.path.join(directory, filename)
        # get input
        with open(path, 'r') as input_file:
            input_text = input_file.read()
            # append the input into list
            file_contents.append(input_text)        
            
    return file_contents
        
train_neg = files_to_list(train_folder_neg)
train_pos = files_to_list(train_folder_pos)
test_neg = files_to_list(test_folder_neg)
test_pos = files_to_list(test_folder_pos)

# # # # CREATING LABELS (1 FOR POSITIVE 0 FOR NEGATIVE) # # # #
# first positives then negatives
train_ylabel = [1] * 12500 + [0] * 12500
test_ylabel = [1] * 12500 + [0] * 12500

# # # # CREATING TRAINING AND TEST SETS FOR X # # # #
# first positives, then negatives
train_X = train_neg + train_pos
test_X = test_neg + test_pos

# let's change these to numpy arrays

train_X = np.asarray(train_X)
test_X = np.asarray(test_X)

# # # # VERSION 1 FOR CLASSIFICATION MODEL # # # #
# Let's first convert the input sentence into the word vector representation
# Let's use GloVe embeddings (50-dimensional) that's already pretrained
# lets download from https://nlp.stanford.edu/projects/glove/ - the wikipedia pretrained
   
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

words_to_index, index_to_words, word_to_vec_map = read_glove_vecs("/home/roope/Downloads/glove.twitter.27B/glove.twitter.27B.50d.txt")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.activations import sigmoid

def find_longest_sentence(searched_list = None):
    longest_sentence = 0
    for sentence in searched_list:
        if len(sentence) > longest_sentence:
            longest_sentence = len(sentence)
    return longest_sentence

train_X_longest = find_longest_sentence(train_X)
test_X_longest = find_longest_sentence(test_X)

# # # # Tokenizing sentences # # # #
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """    
    m = X.shape[0]                                   # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))   
    summa = 0
    for i in range(m):                               # loop over training examples        
        # tokenize the ith training sentence
        sentence_words = tokenizer.tokenize(X[i].lower())
        # Initialize j to 0
        j = 0        
        # Loop over the words of sentence_words
        for z in range(0,max_len):
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[sentence_words[z]]
            except:
                summa += 1
                print(summa)
                X_indices[i, j] = word_to_index['unk']
                
            # Increment j to j + 1
            j = j + 1
    return X_indices

# # # # train_X to index matrix using abofe function # # # #
train_X_indices = sentences_to_indices(train_X, words_to_index, max_len=128)
# 242 450 unknown wordia..
train_X_indices.size # 3 200 000 sanaa yhteensä (ei lähdetä muuttelemaan vielä kuitenkaan tokenizeria)

# # # # CREATING PRETRAINED EMBEDDING LAYER WITH GLOVE VECTOR # # # #


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for (word, index) in word_to_index.items():
        try:
            emb_matrix[index, :] = word_to_vec_map[word]
        except:
            emb_matrix[index, :] = word_to_vec_map["unk"]
            print(index)
    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)

def model_v1(input_shape, word_to_vec_map, words_to_index):
    """
    Function creating model_v1's graph for sentiment classification
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    # defining inputs in the graph as sentence indices
    sentence_indices = Input(input_shape, dtype='int32')
    
    # creating embedding layer pretrained with GloVe vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)
    
    # propagating sentence indices through the embedding layer
    embeddings = embedding_layer(sentence_indices)
    
    # propagating the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    # adding dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a 1 of 5-dimensional vectors.
    X = Dense(1)(X)
    # Add a softmax activation
    X = Activation('sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    
    return model

model = model_v1((128,), word_to_vec_map, words_to_index)
model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_X_indices 

# let's set a seed for shuffling
train_ylabel = np.asarray(train_ylabel)
train_ylabel = train_ylabel[:,np.newaxis]

# shuffling the data
from sklearn.utils import shuffle

train_X_indices_shuffled, train_ylabel_shuffled = shuffle(train_X_indices, train_ylabel)

# # # # Fitting the model / training # # # #
model.fit(train_X_indices_shuffled, train_ylabel_shuffled, epochs = 50, batch_size = 32, shuffle=True)


# # # # PREDICTIONS WITH TEST DATA # # # #
test_X_indices = sentences_to_indices(test_X, words_to_index, max_len=128)
y_test = np.asarray(test_ylabel)
y_test = y_test[:,np.newaxis] 

loss, acc = model.evaluate(test_X_indices, y_test)
print()
print("Test accuracy = ", acc)

# # # # LETS TEST WITH TEXTBLOB # # # #
from textblob import TextBlob

# let's get sentiments of the test_X with textblo
sentiments = []
for text in test_X:
    blob = TextBlob(text)
    sentiments.append(blob.sentiment.polarity)

# mean sentiment
def mean_of_list(L = None):
    total = 0 
    for item in L:
        total = total + item
    return total / len(L)

mean_sentiment = mean_of_list(sentiments) # about 0.1036 is the mean sentiment, let's use this

def median_of_list(L):
    L.sort()
    if (len(L) % 2) == 0:
        median_below = L[(len(L) // 2) -1]
        median_above = L[(len(L) // 2)]
        median = (median_below + median_above) / 2
        return median
    else: 
        median = L[(len(L) // 2)]
        return median
    
# sentiment (polarity) changing to 1 or 0 based on mean_sentiment
def predicting_sentiment(sentiments = None):
    for i in range(len(sentiments)):
        if sentiments[i] > -0.104374:
            sentiments[i] = 1
        else:
            sentiments[i] = 0
    return sentiments

preds = predicting_sentiment(sentiments)

# predictions and true values
sentiment_predictions = np.asarray(preds)
y_test_vector = np.asarray(test_ylabel)

def finding_accuracy(predictions = None, true_values = None):
    right_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == true_values[i]:
            right_predictions += 1
    return right_predictions / len(predictions)

accuracy = finding_accuracy(sentiment_predictions, y_test_vector)
# TEXT BLOB HANKALA ARVIOIDA ACCURACYÄ, KOSKA ON JATKUMO SCORE - MISTÄ LAITETAANP OIKKI?
