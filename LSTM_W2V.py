
# coding: utf-8

# In[1]:


import csv
import random
import gensim
import re
import nltk
import keras

import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Activation 
from keras.layers import Layer, Concatenate, Flatten, Bidirectional
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Optimizer





# In[2]:


#gpu allowed
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)


# In[3]:


article_bodies = pd.read_csv("Dataset/article_body_texts.csv", encoding='utf-8')
train_data= pd.read_csv("Dataset/train_data.csv", encoding='utf-8')
validation_data= pd.read_csv("Dataset/validation_data.csv", encoding='utf-8')
test_data= pd.read_csv("Dataset/test_data.csv", encoding='utf-8')


# In[4]:


train_merged = pd.merge(train_data, article_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
train_merged = train_merged.drop('Body ID',1)
#WORK = test_merged.values.tolist()

validation_merged = pd.merge(validation_data, article_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
validation_merged = validation_merged.drop('Body ID',1)

test_merged = pd.merge(test_data, article_bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
test_merged = test_merged.drop('Body ID',1)


# In[5]:


train_merged


# In[6]:


# Converting all sentence to string
train_merged['Headline'] = train_merged['Headline'].apply(lambda x: str(x))
train_merged['articleBody'] = train_merged['articleBody'].apply(lambda x: str(x))

validation_merged['Headline'] = validation_merged['Headline'].apply(lambda x: str(x))
validation_merged['articleBody'] = validation_merged['articleBody'].apply(lambda x: str(x))

test_merged['Headline'] = test_merged['Headline'].apply(lambda x: str(x))
test_merged['articleBody'] = test_merged['articleBody'].apply(lambda x: str(x))


# In[7]:


print(train_merged.shape)
print(validation_merged.shape)
print(test_merged.shape)


# In[8]:


sent_len_1 = lambda x:len(x)
train_merged['Headline_length'] = train_merged.Headline.apply(sent_len_1)
train_merged[train_merged['Headline_length']<5]['Headline'].tail()

sent_len_2 = lambda x:len(x)
train_merged['articleBody_length'] = train_merged.articleBody.apply(sent_len_2)
train_merged[train_merged['articleBody_length']<500]['articleBody'].tail()

sent_len_3 = lambda x:len(x)
validation_merged['Headline_length'] = validation_merged.Headline.apply(sent_len_3)
validation_merged[validation_merged['Headline_length']<0]['Headline'].tail()

sent_len_4 = lambda x:len(x)
validation_merged['articleBody_length'] = validation_merged.articleBody.apply(sent_len_4)
validation_merged[validation_merged['articleBody_length']<0]['articleBody'].tail()

sent_len_5 = lambda x:len(x)
test_merged['Headline_length'] = test_merged.Headline.apply(sent_len_5)
test_merged[test_merged['Headline_length']<0]['Headline'].tail()

sent_len_6 = lambda x:len(x)
test_merged['articleBody_length'] = test_merged.articleBody.apply(sent_len_6)
test_merged[test_merged['articleBody_length']<0]['articleBody'].tail()


# In[9]:


# Headline having lesser than 20 characters, articleBody_length  can be discarded - noisy data
indices_1 = train_merged[train_merged['Headline_length']<5].index
train_merged.drop(indices_1, inplace=True)
indices_2 = train_merged[train_merged['articleBody_length']<500].index
train_merged.drop(indices_2, inplace=True)

train_merged.reset_index(inplace=True, drop=True)
train_merged.shape


# In[10]:


train_merged


# In[11]:


#no filter on validation set
indices_3 = validation_merged[validation_merged['Headline_length']<0].index
validation_merged.drop(indices_3, inplace=True)
indices_4 = validation_merged[validation_merged['articleBody_length']<0].index
validation_merged.drop(indices_4, inplace=True)

validation_merged.reset_index(inplace=True, drop=True)
validation_merged.shape


# In[12]:


#no filter on test set
indices_5 = test_merged[test_merged['Headline_length']<0].index
test_merged.drop(indices_5, inplace=True)
indices_6 = test_merged[test_merged['articleBody_length']<0].index
test_merged.drop(indices_6, inplace=True)

test_merged.reset_index(inplace=True, drop=True)
test_merged.shape


# In[13]:


# Pre-processing involves removal of puctuations and converting text to lower case

#Since only 10 percent of the sentences > 16 words, we should be safe using MAX_HEAD_LEN as 20
train_headline_seq = [text_to_word_sequence(sent) for sent in train_merged['Headline']]
print('90th Percentile Train Headline Length:', np.percentile([len(seq) for seq in train_headline_seq], 90))


#Since only 10 percent of the sentences > 712 words, we should be safe using MAX_BODY_LEN as 700
train_body_seq = [text_to_word_sequence(sent) for sent in train_merged['articleBody']]
print('90th Percentile Train Body Length:', np.percentile([len(seq) for seq in train_body_seq], 90))

#Since only 10 percent of the sentences > 16 words, we should be safe using MAX_HEAD_LEN as 20
validation_headline_seq = [text_to_word_sequence(sent) for sent in validation_merged['Headline']]
print('90th Percentile Validation Headline Length:', np.percentile([len(seq) for seq in validation_headline_seq], 90))


#Since only 10 percent of the sentences > 742 words, we should be safe using MAX_BODY_LEN as 700
validation_body_seq = [text_to_word_sequence(sent) for sent in validation_merged['articleBody']]
print('90th Percentile Validation Body Length:', np.percentile([len(seq) for seq in validation_body_seq], 90))

#Since only 15 percent of the sentences > 19 words, we should be safe using MAX_HEAD_LEN as 20
test_headline_seq = [text_to_word_sequence(sent) for sent in test_merged['Headline']]
print('95th Percentile Test Headline Length:', np.percentile([len(seq) for seq in test_headline_seq], 95))


#Since only 15 percent of the sentences > 705 words, we should be safe using MAX_BODY_LEN as 700
test_body_seq = [text_to_word_sequence(sent) for sent in test_merged['articleBody']]
print('95th Percentile Test Body Length:', np.percentile([len(seq) for seq in test_body_seq], 95))


# In[52]:


# These are some hyperparameters that can be tuned
MAX_HEAD_VOCAB_SIZE = 4000
MAX_BODY_VOCAB_SIZE = 25000

MAX_HEAD_LEN = 20
MAX_BODY_LEN = 700


EMBEDDING_DIM = 300
BATCH_SIZE = 500
N_EPOCHS = 5


dense_1_unit = 32
dense_2_unit = 8

dropout_1_rate = 0.6
dropout_2_rate = 0.6

LSTM_DIM = 256

af1 = 'sigmoid'
af2 = 'sigmoid'
af_output = 'softmax'


# In[16]:


train_headline_list = [' '.join(word_tokenize(x) [:MAX_HEAD_VOCAB_SIZE]) for x in train_merged['Headline']] 
print(train_headline_list[:2])

train_body_list = [' '.join(word_tokenize(x)[:MAX_BODY_VOCAB_SIZE]) for x in train_merged['articleBody']]
print(train_body_list[:2])

validation_headline_list = [' '.join(word_tokenize(x)[:MAX_HEAD_VOCAB_SIZE]) for x in validation_merged['Headline']]
print(validation_headline_list[:2])

validation_body_list = [' '.join(word_tokenize(x)[:MAX_BODY_VOCAB_SIZE]) for x in validation_merged['articleBody']]
print(train_body_list[:2])

test_headline_list = [' '.join(word_tokenize(x)[:MAX_HEAD_VOCAB_SIZE]) for x in test_merged['Headline']]
print(test_headline_list[:2])

test_body_list = [' '.join(word_tokenize(x)[:MAX_BODY_VOCAB_SIZE]) for x in test_merged['articleBody']]
print(test_body_list[:2])


# In[17]:


# Tokenize headline and article body without symbol and puncuation
filter_list = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

train_headline_tokenizer = Tokenizer(num_words = MAX_HEAD_VOCAB_SIZE,filters=(filter_list))
train_headline_tokenizer.fit_on_texts(train_headline_list)
print("Number of words in train headline vocabulary:", len(train_headline_tokenizer.word_index))

train_body_tokenizer = Tokenizer(num_words = MAX_BODY_VOCAB_SIZE, filters=(filter_list))
train_body_tokenizer.fit_on_texts(train_body_list)
print("Number of words in train body vocabulary:", len(train_body_tokenizer.word_index))

validation_headline_tokenizer = Tokenizer(num_words = MAX_HEAD_VOCAB_SIZE,filters=(filter_list))
validation_headline_tokenizer.fit_on_texts(validation_headline_list)
print("Number of words in validation headline vocabulary:", len(validation_headline_tokenizer.word_index))

validation_body_tokenizer = Tokenizer(num_words = MAX_BODY_VOCAB_SIZE, filters=(filter_list))  
validation_body_tokenizer.fit_on_texts(validation_body_list)
print("Number of words in validation body vocabulary:", len(validation_body_tokenizer.word_index))

test_headline_tokenizer = Tokenizer(num_words = MAX_HEAD_VOCAB_SIZE,filters=(filter_list))
test_headline_tokenizer.fit_on_texts(test_headline_list)
print("Number of words in test headline vocabulary:", len(test_headline_tokenizer.word_index))

test_body_tokenizer = Tokenizer(num_words = MAX_BODY_VOCAB_SIZE, filters=(filter_list))  
test_body_tokenizer.fit_on_texts(test_body_list)
print("Number of words in test body vocabulary:", len(test_body_tokenizer.word_index))



# In[18]:


# Limit vocab and idx-word dictionary
train_headline_word_index = {k: v for k, v in train_headline_tokenizer.word_index.items() if v < MAX_HEAD_VOCAB_SIZE}
#train_headline_idx_to_word = dict((v,k) for k,v in train_headline_word_index.items())

train_body_word_index = {k: v for k, v in train_body_tokenizer.word_index.items() if v < MAX_BODY_VOCAB_SIZE}
#train_body_idx_to_word = dict((v,k) for k,v in train_body_word_index.items())

validation_headline_word_index = {k: v for k, v in validation_headline_tokenizer.word_index.items() if v < MAX_HEAD_VOCAB_SIZE}
# validation_headline_idx_to_word = dict((v,k) for k,v in validation_headline_word_index.items())

validation_body_word_index = {k: v for k, v in validation_body_tokenizer.word_index.items() if v < MAX_BODY_VOCAB_SIZE}
# validation_body_idx_to_word = dict((v,k) for k,v in validation_body_word_index.items())

test_headline_word_index = {k: v for k, v in test_headline_tokenizer.word_index.items() if v < MAX_HEAD_VOCAB_SIZE}
# test_headline_idx_to_word = dict((v,k) for k,v in train_headline_word_index.items())

test_body_word_index = {k: v for k, v in test_body_tokenizer.word_index.items() if v < MAX_BODY_VOCAB_SIZE}
# test_body_idx_to_word = dict((v,k) for k,v in test_body_word_index.items())


# In[19]:


train_headline_word_index


# In[20]:


print(len(train_headline_word_index))
print(len(train_body_word_index))
print(len(validation_headline_word_index))
print(len(validation_body_word_index))
print(len(test_headline_word_index))
print(len(test_body_word_index))


# In[21]:


# define stopworks 
stop_words_nltk = list(stopwords.words('english'))
stop_words = list(stop_words_nltk) 

# remove stop works in dict
for i in train_headline_word_index:
    if i in stop_words:
        train_headline_word_index[i] = 0

for i in train_body_word_index:
    if i in stop_words:
        train_body_word_index[i] = 0

for i in validation_headline_word_index:
    if i in stop_words:
        validation_headline_word_index[i] = 0

for i in validation_body_word_index:
    if i in stop_words:
        validation_body_word_index[i] = 0

for i in test_headline_word_index:
    if i in stop_words:
        test_headline_word_index[i] = 0

for i in test_body_word_index:
    if i in stop_words:
        test_body_word_index[i] = 0


# In[22]:


# Fransfer words into index.
# Input size are utlized as MAX_HEAD_LEN and MAX_BODY_LEN
# If input headline/body length less than MAX_HEAD_LEN/MAX_BODY_LEN, index is padded with 0

train_headline_X = train_headline_tokenizer.texts_to_sequences(train_headline_list) 
train_headline_X = pad_sequences(train_headline_X, maxlen=MAX_HEAD_LEN, padding='post', truncating='post')

train_body_X = train_body_tokenizer.texts_to_sequences(train_body_list)
train_body_X = pad_sequences(train_body_X, maxlen=MAX_BODY_LEN, padding='post', truncating='post')

validation_headline_X = validation_headline_tokenizer.texts_to_sequences(validation_headline_list)
validation_headline_X = pad_sequences(validation_headline_X, maxlen=MAX_HEAD_LEN, padding='post', truncating='post')

validation_body_X = validation_body_tokenizer.texts_to_sequences(validation_body_list)
validation_body_X = pad_sequences(validation_body_X, maxlen=MAX_BODY_LEN, padding='post', truncating='post')

test_headline_X = test_headline_tokenizer.texts_to_sequences(test_headline_list)
test_headline_X = pad_sequences(test_headline_X, maxlen=MAX_HEAD_LEN, padding='post', truncating='post')

test_body_X = test_body_tokenizer.texts_to_sequences(test_body_list)
test_body_X = pad_sequences(test_body_X, maxlen=MAX_BODY_LEN, padding='post', truncating='post')




# In[23]:


print(train_headline_X.shape)
print(train_body_X.shape)
print(validation_headline_X.shape)
print(validation_body_X.shape)
print(test_headline_X.shape)
print(test_body_X.shape)


# In[24]:


print(train_headline_list[:2])


# In[25]:


print(train_headline_X[:2])


# In[26]:


#embedding matrix
#load Word2vec embeddings
W2V_DIR = '/Users/TerryNg/Documents/University_of_Waterloo/MSCI_641/Project/code/GoogleNews-vectors-negative300 (1).bin.gz'




# Load the word2vec embeddings 
embeddings = gensim.models.KeyedVectors.load_word2vec_format(W2V_DIR, binary=True)


# In[27]:


# Create an embedding matrix containing only the word's in our vocabulary
# If the word does not have a pre-trained embedding, then randomly initialize the embedding

train_body_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(train_body_tokenizer.word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0


for word, i in train_body_tokenizer.word_index.items(): # i=0 is the embedding for the zero padding   
    try:
        train_body_embeddings_vector = embeddings[word]
    except KeyError:
        train_body_embeddings_vector = None
    if train_body_embeddings_vector is not None:
        train_body_embeddings_matrix[(i)] = train_body_embeddings_vector

        
# del embeddings


# In[28]:


train_body_embeddings_matrix.shape


# In[29]:


train_headline_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(train_headline_tokenizer.word_index)+1, EMBEDDING_DIM)) 

for word, i in train_headline_tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        train_headline_embeddings_vector = embeddings[word]
    except KeyError:
         train_headline_embeddings_vector = None
    if  train_headline_embeddings_vector is not None:
        train_headline_embeddings_matrix[i] =  train_headline_embeddings_vector
        
# del embeddings


# In[30]:


# Create an embedding matrix containing only the word's in our vocabulary
# If the word does not have a pre-trained embedding, then randomly initialize the embedding

validation_body_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(validation_body_tokenizer.word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0


for word, i in validation_body_tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        validation_body_embeddings_vector = embeddings[word]
    except KeyError:
        validation_body_embeddings_vector = None
    if validation_body_embeddings_vector is not None:
        validation_body_embeddings_matrix[(i)] = validation_body_embeddings_vector

        
# del embeddings


# In[31]:


validation_headline_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(validation_headline_tokenizer.word_index)+1, EMBEDDING_DIM)) 

for word, i in validation_headline_tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        validation_headline_embeddings_vector = embeddings[word]
    except KeyError:
         validation_headline_embeddings_vector = None
    if  validation_headline_embeddings_vector is not None:
        validation_headline_embeddings_matrix[i] =  validation_headline_embeddings_vector
        
# del embeddings


# In[32]:


# Create an embedding matrix containing only the word's in our vocabulary
# If the word does not have a pre-trained embedding, then randomly initialize the embedding

test_body_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(test_body_tokenizer.word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0


for word, i in test_body_tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        test_body_embeddings_vector = embeddings[word]
    except KeyError:
        test_body_embeddings_vector = None
    if test_body_embeddings_vector is not None:
        test_body_embeddings_matrix[(i)] = test_body_embeddings_vector

        
# del embeddings


# In[33]:


test_headline_embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(test_headline_tokenizer.word_index)+1, EMBEDDING_DIM)) 

for word, i in test_headline_tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        test_headline_embeddings_vector = embeddings[word]
    except KeyError:
         test_headline_embeddings_vector = None
    if  test_headline_embeddings_vector is not None:
        test_headline_embeddings_matrix[i] =  test_headline_embeddings_vector
        
# del embeddings


# In[34]:


print(train_body_embeddings_matrix.shape)
print(validation_body_embeddings_matrix.shape)
print(test_body_embeddings_matrix.shape)

print(train_headline_embeddings_matrix.shape)
print(validation_headline_embeddings_matrix.shape)
print(test_headline_embeddings_matrix.shape)


# In[35]:


y_train = train_merged['Stance']
train_Y = []

for stance in y_train:
    if stance == 'agree':
        train_Y.append(0)
    if stance == 'disagree':
        train_Y.append(0)
    if stance == 'discuss':
        train_Y.append(0)
    if stance == 'unrelated':
        train_Y.append(1)
        
y_validation = validation_merged['Stance']
validation_Y = []

for stance in y_validation:
    if stance == 'agree':
        validation_Y.append(0)
    if stance == 'disagree':
        validation_Y.append(0)
    if stance == 'discuss':
        validation_Y.append(0)
    if stance == 'unrelated':
        validation_Y.append(1)


# In[36]:


# y_train = train_merged['Stance']
# train_Y = []

# for stance in y_train:
#     if stance == 'agree':
#         train_Y.append(0)
#     if stance == 'disagree':
#         train_Y.append(1)
#     if stance == 'discuss':
#         train_Y.append(2)
#     if stance == 'unrelated':
#         train_Y.append(3)
        
# y_validation = validation_merged['Stance']
# validation_Y = []

# for stance in y_validation:
#     if stance == 'agree':
#         validation_Y.append(0)
#     if stance == 'disagree':
#         validation_Y.append(1)
#     if stance == 'discuss':
#         validation_Y.append(2)
#     if stance == 'unrelated':
#         validation_Y.append(3)


# In[37]:


len(train_headline_tokenizer.word_index)+1


# In[53]:


#For headline

# Input layer


headline_input = Input(shape =(MAX_HEAD_LEN, ), 
                       name='headline_input')

# Embedding lookup layer
headline_emb_look_up = Embedding(input_dim=len(train_headline_tokenizer.word_index)+1,
                                 output_dim=EMBEDDING_DIM,
                                 weights = [train_headline_embeddings_matrix], 
                                 trainable=False, 
                                 name='headline_word_embedding_layer', 
                                 mask_zero=True) # trainable=True results in overfitting
                                

headline_emb_1 = headline_emb_look_up(headline_input)


# LSTM layer
# Can try Bidirectional-LSTM
headline_LSTM = LSTM(LSTM_DIM, return_sequences=False, name='headline_lstm_layer')(headline_emb_1)



# In[54]:


# For body

# Input layer
body_input = Input(shape =(MAX_BODY_LEN, ), 
                   name='body_input')

# Embedding lookup layer
body_emb_look_up = Embedding(input_dim=len(train_body_tokenizer.word_index)+1,
                             output_dim=EMBEDDING_DIM,
                             weights = [train_body_embeddings_matrix], 
                             trainable=False, 
                             name='body_word_embedding_layer', 
                             mask_zero=True) # trainable=True results in overfitting
                            

body_emb_1 = body_emb_look_up(body_input)



# LSTM layer
# Can try Bidirectional-LSTM
body_LSTM = LSTM(LSTM_DIM, return_sequences=False, name='body_lstm_layer')(body_emb_1)


# In[55]:


# Two main layer

# concatenate two inputs for first main hidden layer
merged_input = Concatenate(name='LSTM_concat')([headline_LSTM, body_LSTM])

# Dense layer 1
dense_1 = Dense(units=dense_1_unit, 
                name='dense_1')(merged_input)

# Dropout layer 1
dropout_1 = Dropout(rate = dropout_1_rate, 
                          name='dropout_1')(dense_1)

# #flatten layer
# flatten_1 = Flatten(name='flatten_1')(dropout_1)

# Activation layer 1 of second main hidden layer
activation_1 = Activation(activation=af1, 
                          name='activation_1')(dropout_1)

# Dense layer 2
dense_2 = Dense(units=dense_2_unit, 
                name='dense_2')(activation_1)

# Dropout layer 2
dropout_2 = Dropout(rate = dropout_2_rate, 
                    name='dropout_2')(dense_2)

# #flatten layer 2
# flatten_2 = Flatten(name='flatten_1')(dropout_2)

# Activation layer 2 for ouput layer
activation_2 = Activation(activation = af2, 
                          name='activation_2')(dropout_2)

# Output_layer
output_prob = Dense(units=1, 
                    activation = af_output, 
                    name='output_layer')(activation_2)


# In[56]:


model = Model(inputs=[headline_input, body_input], 
              output=output_prob, 
              name='LSTM_classification')

model.summary()


# In[57]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[58]:


model.fit(x = [train_headline_X, train_body_X], 
          y = train_Y, 
          batch_size=BATCH_SIZE, 
          epochs=N_EPOCHS, 
          validation_data=([validation_headline_X, validation_body_X], validation_Y))

