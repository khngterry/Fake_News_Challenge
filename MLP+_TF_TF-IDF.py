
# coding: utf-8

# In[1]:


import random
import re
import nltk
import tensorflow as tf
import numpy as np


from csv import DictReader

from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


#gpu allowed
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)


# In[3]:


# Class of dataset being input
class Dataset:

    #read CSV files
    def read(self, filename):

        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows
    
    
    def __init__(self, instances, ArticalBody):

        # Load data
        self.instances = self.read(instances)
        bodies = self.read(ArticalBody)
        self.headlines = {}
        self.bodies = {}

        # Process instances
        for i in self.instances:
            if i['Headline'] not in self.headlines:
                headline_length = len(self.headlines)
                self.headlines[i['Headline']] = headline_length
            i['Body ID'] = int(i['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']



# In[4]:


# stop works
stop_words_nltk = list(stopwords.words('english'))
#stop_words_sklearn = list(stop_words.ENGLISH_STOP_WORDS)
stop_words = stop_words_nltk #+ stop_words_sklearn

# Label of classes
label_dict = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
label_class = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}


# In[5]:


#random function

r = random.Random()


# In[6]:


# Hyperparameters
freq_N = 5000
hidden_size_1 = 100
#hidden_size_2 = 50
drop_out_rate = 0.7
l2_reg = 0.0001
learn_rate = 0.05
clip_ratio = 5
batch_size_train = 500
epochs = 50


# In[7]:


# Set file names
train_file = "Dataset/train_data.csv"
train_bodies = "Dataset/article_body_texts.csv"
test_file = "Dataset/validation_data.csv"
test_bodies = "Dataset/article_body_texts.csv"


# In[8]:


# open data sets
train_data = Dataset(train_file, train_bodies)
test_data = Dataset(test_file, test_bodies)
train_len = len(train_data.instances)


# In[9]:


#  Head and body of train set

train_heads = []
train_heads_track = {}
train_bodies = []
train_bodies_track = {}
train_body_ids = []

for i in train_data.instances:
    head = i['Headline']
    body_id = i['Body ID']
    if head not in train_heads_track:
        train_heads.append(head)
        train_heads_track[head] = 1
    if body_id not in train_bodies_track:
        train_bodies.append(train_data.bodies[body_id])
        train_bodies_track[body_id] = 1
        train_body_ids.append(body_id)


# In[10]:


train_id_dict = {}

# Create body ID dictionary for train set
for i, element in enumerate(train_heads + train_body_ids):
    train_id_dict[element] = i


# In[11]:


# Create Trem-frequency vectorizers for train set

bow_vectorizer = CountVectorizer(max_features = freq_N,
                                 stop_words=stop_words)
train_bow = bow_vectorizer.fit_transform(train_heads + train_bodies)

tf_vectorizer = TfidfTransformer(use_idf=False).fit(train_bow)
tf = tf_vectorizer.transform(train_bow).toarray()


# In[12]:


# Create Trem-frequency-inverse-document-frequency vectorizers for both train set and test set
test_heads =[]
test_bodies = []

tfidf_vectorizer = TfidfVectorizer(max_features = freq_N, 
                                   stop_words=stop_words).fit(train_heads + train_bodies + test_heads + test_bodies)  


# In[13]:


# Generate Term-frequency and Trem-frequency-inverse-document-frequency features from train set

train_head_tfidf_track = {}
train_body_tfidf_track = {}

train_set = []
train_cos_track = {}
train_stance = []

for i in train_data.instances:
    head = i['Headline']
    body_id = i['Body ID']
    
    #Term-frequency feature for headlines
    train_head_tf = tf[train_id_dict[head]].reshape(1, -1)
    
    #Term-frequency feature for body
    train_body_tf = tf[train_id_dict[body_id]].reshape(1, -1)
    
    #Trem-frequency-inverse-document-frequency
    if head not in train_head_tfidf_track:
        train_head_tfidf = tfidf_vectorizer.transform([head]).toarray()
        train_head_tfidf_track[head] = train_head_tfidf
    else:
        train_head_tfidf = train_head_tfidf_track[head]
        
    if body_id not in train_body_tfidf_track:
        train_body_tfidf = tfidf_vectorizer.transform([train_data.bodies[body_id]]).toarray()
        train_body_tfidf_track[body_id] = train_body_tfidf
    else:
        train_body_tfidf = train_body_tfidf_track[body_id]
        
    if (head, body_id) not in train_cos_track:
        train_tfidf_cos = cosine_similarity(train_head_tfidf,  train_body_tfidf)[0].reshape(1, 1)
        train_cos_track[(head, body_id)] = train_tfidf_cos
    else:
        train_tfidf_cos = train_cos_track[(head, body_id)]
        
        
    #input feature from training set
    input_feat_vec = np.squeeze(np.c_[train_head_tf, train_body_tf, train_tfidf_cos])
    train_set.append(input_feat_vec)
    


# In[14]:


print(len(input_feat_vec))


# In[15]:


# Generate Trem-frequency-inverse-document-frequency features from test set

test_set = []
test_heads_track = {}
test_bodies_track = {}
test_cos_track = {}
test_stance = []

# Process test set
for i in test_data.instances:
    head = i['Headline']
    body_id = i['Body ID']
    
    #Trem-frequency-inverse-document-frequency
    if head not in test_heads_track:
        test_head_bow = bow_vectorizer.transform([head]).toarray()
        test_head_tf = tf_vectorizer.transform(test_head_bow).toarray()[0].reshape(1, -1)
        test_head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
        test_heads_track[head] = (test_head_tf, test_head_tfidf)
    else:
        test_head_tf = test_heads_track[head][0]
        test_head_tfidf = test_heads_track[head][1]
        
    if body_id not in test_bodies_track:
        test_body_bow = bow_vectorizer.transform([test_data.bodies[body_id]]).toarray()
        test_body_tf = tf_vectorizer.transform(test_body_bow).toarray()[0].reshape(1, -1)
        test_body_tfidf = tfidf_vectorizer.transform([test_data.bodies[body_id]]).toarray().reshape(1, -1)
        test_bodies_track[body_id] = (test_body_tf, test_body_tfidf)
    else:
        test_body_tf = test_bodies_track[body_id][0]
        test_body_tfidf = test_bodies_track[body_id][1]
        
    if (head, body_id) not in test_cos_track:
        test_tfidf_cos = cosine_similarity(test_head_tfidf, test_body_tfidf)[0].reshape(1, 1)
        test_cos_track[(head, body_id)] = test_tfidf_cos
    else:
        test_tfidf_cos = test_cos_track[(head, body_id)]
        
    #input feature from testing set 
    test_feat_vec = np.squeeze(np.c_[test_head_tf, test_body_tf, test_tfidf_cos])
    test_set.append(test_feat_vec)


# In[16]:


np.shape(test_head_tf)


# In[17]:


# labelled class from train set

for i in train_data.instances:
    train_stance.append(label_dict[i['Stance']])


# In[18]:


# TensorFlow training model
# Feed-forwad neural network
import tensorflow as tf

# Input Setting
feature_size = len(train_set[0])
input_features = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances = tf.placeholder(tf.int64, [None], 'stances')
keep_prob = tf.placeholder(tf.float32)

# batch size
batch_size = tf.shape(input_features)[0]



# 1st Hidden layer:

#contribu module
contrib_module = tf.contrib.layers.linear(input_features, hidden_size_1)

# Dense
dense_1 = tf.layers.dense(contrib_module, units=10)
# Dropout
dropout_1 = tf.nn.dropout(dense_1, keep_prob=keep_prob)
# Flatten
flatten_1 = tf.nn.dropout(tf.contrib.layers.linear(dropout_1, 4), keep_prob=keep_prob)
# Activation: relu
activation_1 = tf.nn.relu(flatten_1)

# Reshape
logits_1 = tf.reshape(activation_1, [batch_size, 4])

####
# # 2nd Hidden layer:
# # accuracy decreases with 2nd hidden layer ---> overfit issue
# # with 2 hidden layer, 'agree' and 'disagree' cannot be classified

# dense_2 = tf.layers.dense(logits_1, units=10)
# # Activation: relu
# activation_2 = tf.nn.relu(dense_2)
# # Dropout
# dropout_2 = tf.nn.dropout(activation_2, keep_prob=keep_prob)
# # Flatten
# flatten_2 = tf.nn.dropout(tf.contrib.layers.linear(dropout_2, 4), keep_prob=keep_prob)
# # Reshape
# logits_2 = tf.reshape(flatten_2, [batch_size, 4])
#####

# L2 Regularization for the MLP weights to the objective
tf_vars = tf.trainable_variables()


# Compute overall loss
# Calculate L2 loss
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_reg


#Computes cross entropy between logits and labels.
sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=stances)

#Computes the sum of loss
loss = tf.reduce_sum(sparse + l2_loss)

                                                                    
# Define prediction
softmaxed_logits = tf.nn.softmax(logits_1)
predict = tf.argmax(softmaxed_logits, 1)


# In[19]:


# Define optimisation function

# Use Adam as gradiant-based optimizatior
opti_alg = tf.train.AdamOptimizer(learn_rate)

# perform gradients computation
gradients_1 = tf.gradients(loss, tf_vars)

#gradient clipping preventing exploding gradients
gradient_clipping, _ = tf.clip_by_global_norm(gradients_1, clip_ratio)

# optimization function
opt_func = opti_alg.apply_gradients(zip(gradient_clipping, tf_vars))


# In[20]:


# Train the network

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        total_loss = 0
        indices = list(range(train_len))
        r.shuffle(indices)

        for i in range(train_len // batch_size_train):
            batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
            batch_features = [train_set[i] for i in batch_indices]
            batch_stances = [train_stance[i] for i in batch_indices]

            batch_feed_dict = {input_features : batch_features, stances: batch_stances, keep_prob: drop_out_rate}
            _, current_loss = sess.run([opt_func, loss], feed_dict=batch_feed_dict)
            total_loss += current_loss


    # Predict
    test_feat_dict = {input_features: test_set, keep_prob: 1.0}
    test_pred = sess.run(predict, feed_dict=test_feat_dict)




# In[21]:


np.shape(train_set)


# In[22]:


# scoring system
# only for validation set

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


# Confusion Matrix will be printed out on result


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


# The report_score function is based off the original scorer provided in the FNC-1 dataset repository 
# This will print a final score your classifier.

def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score


if __name__ == "__main__":
    actual = [0,0,0,0,1,1,0,3,3]
    predicted = [0,0,0,0,1,1,2,3,3]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])


# In[23]:


# labelled class from test set
  
for i in test_data.instances:
    test_stance.append(label_dict[i['Stance']])


# In[24]:


report_score([LABELS[e] for e in test_stance],[LABELS[e] for e in test_pred])


# In[390]:


# Initialise
test_predictions = []
for stance in test_pred:
     test_predictions.append(label_class [stance])


# In[104]:


np.savetxt("predictions.csv",test_predictions,fmt='%s')


# In[ ]:




