
# coding: utf-8

# In[1]:


import csv
import math
import numpy as np
import random
import re
import pickle
from tqdm import tqdm

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words


from sklearn import linear_model
from collections import defaultdict





# In[2]:


#Read and process dataset

# Initialise
bodies = []
train_unrelated = []
train_discuss = []
train_agree = []
train_disagree = []

#Read Article Body
f_bodies = open('Dataset/article_body_texts.csv', 'r', encoding='utf-8')
csv_bodies = csv.DictReader(f_bodies)

for row in csv_bodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(bodies):
        bodies += [None] * (body_id + 1 - len(bodies))
    bodies[body_id] = row['articleBody']
    
f_bodies.close()


#Read Train set
train_stances = open('Dataset/train_data.csv', 'r', encoding='utf-8')
csv_stances = csv.DictReader(train_stances)

#mapping
for row in csv_stances:
    body = bodies[int(row['Body ID'])]
    if row['Stance'] == 'unrelated':
        train_unrelated.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'discuss':
        train_discuss.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'agree':
        train_agree.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'disagree':
        train_disagree.append((row['Headline'], body, row['Stance']))

train_stances.close()

print('\tUnrltd\tDiscuss\t Agree\tDisagree')
print('Train\t', len(train_unrelated), '\t', len(train_discuss), '\t', len(train_agree), '\t', len(train_disagree))


# In[4]:


train_all  = (train_unrelated + train_discuss + train_agree + train_disagree) 
random.Random(0).shuffle(train_all)
train_all = np.array(train_all)

print('Train (Total)', train_all.shape)


# In[5]:


# tokenise text excluding punctuation, symbols,digits and stopwords

pattern = re.compile("[^a-zA-Z0-9 ]+")  

stop_words_nltk = list(stopwords.words('english'))
stop_words_sklearn = list(stop_words.ENGLISH_STOP_WORDS)
stop_words = stop_words_nltk + stop_words_sklearn


def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text


# In[6]:


# compute term frequency

def extract_tf(text, ngram=1):
    words = tokenise(text)
    ret = defaultdict(float)
    for i in range(len(words)):
        for j in range(1, ngram+1):
            if i - j < 0:
                break
            word = [words[i-k] for k in range(j)]
            ret[word[0] if ngram == 1 else tuple(word)] += 1.0
    return ret
    
for i in range(5):
    print(train_all[i, 0], extract_tf(train_all[i, 0]))
print(train_all[0, 0], extract_tf(train_all[0, 0], ngram=2))


# In[7]:


#create a corpus with all headlines and article bodies

corpus = np.r_[train_all[:, 1], train_all[:, 0]]  


# In[8]:


#compute ducument frequency of every word in corpus

df = defaultdict(float)
for doc in tqdm(corpus):
    words = tokenise(doc)
    seen = set()
    for word in words:
        if word not in seen:
            df[word] += 1.0
            seen.add(word)


# In[9]:


# compute inverse document frequency of every word in corpus

num_docs = corpus.shape[0]
idf = defaultdict(float)
for word, val in tqdm(df.items()):
    idf[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf


# In[10]:


# pre-trained 50-dimensional word embedding vector from GloVe 

f_glove = open("glove_6B/glove.6B.50d.txt", "rb") 
glove_vectors = {}
for line in tqdm(f_glove):
    glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))

print(glove_vectors['glove'])


# In[11]:


# Extract cosine similarity feature from GLoVe vectors
# GLoVe vectors for each word = tf-idf of word * GLoVe of word / total tf-idf)


def doc_to_glove(doc):
    doc_tf = extract_tf(doc)
    doc_tf_idf = defaultdict(float)
    for word, tf in doc_tf.items():
        doc_tf_idf[word] = tf * idf[word]
        
    doc_vector = np.zeros(glove_vectors['glove'].shape[0])
    if np.sum(list(doc_tf_idf.values())) == 0.0:  
        return doc_vector
    
    for word, tf_idf in doc_tf_idf.items():
        if word in glove_vectors:
            doc_vector += glove_vectors[word] * tf_idf
    doc_vector /= np.sum(list(doc_tf_idf.values()))
    return doc_vector

def dot_product(vec1, vec2):
    sigma = 0.0
    for i in range(vec1.shape[0]):  
        sigma += vec1[i] * vec2[i]
    return sigma
    
def magnitude(vec):
    return np.sqrt(np.sum(np.square(vec)))
        
def cosine_similarity(doc):
    headline_vector = doc_to_glove(doc[0])
    body_vector = doc_to_glove(doc[1])
    
    if magnitude(headline_vector) == 0.0 or magnitude(body_vector) == 0.0:  
        return 0.0
    
    return dot_product(headline_vector, body_vector) / (magnitude(headline_vector) * magnitude(body_vector))

for i in range(5):
    print(cosine_similarity(train_all[i]), train_all[i, 2])




# In[13]:


# Extract Kullback–Leibler Divergence features

def divergence(lm1, lm2):
    sigma = 0.0
    for i in range(lm1.shape[0]):  
        sigma += lm1[i] * np.log(lm1[i] / lm2[i])
    return sigma

def kl_divergence(doc, eps=0.1):
    # Convert headline and body to 1-gram representations
    tf_headline = extract_tf(doc[0])
    tf_body = extract_tf(doc[1])
    
    # Convert dictionary tf representations to vectors
    words = set(tf_headline.keys()).union(set(tf_body.keys()))
    vec_headline, vec_body = np.zeros(len(words)), np.zeros(len(words))
    i = 0
    for word in words:
        vec_headline[i] += tf_headline[word]
        vec_body[i] = tf_body[word]
        i += 1
    
    # Compute a 1-gram language model for headline and body
    lm_headline = vec_headline + eps
    lm_headline /= np.sum(lm_headline)
    lm_body = vec_body + eps
    lm_body /= np.sum(lm_body)
    
    
    return divergence(lm_headline, lm_body)

for i in range(5):
    print(kl_divergence(train_all[i]), train_all[i, 2])


# In[14]:


# Extract ngram_overlap features
# It means Returns how many times n-grams (up to 3-gram) that occur both in headline and article body.

def ngram_overlap(doc):
    tf_headline = extract_tf(doc[0], ngram=5)
    tf_body = extract_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_headline.keys():
        if words in tf_body:
            matches += tf_body[words]
            
    # normalise for document length
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  

for i in range(5):
    print(ngram_overlap(train_all[i]), train_all[i, 2])


# In[15]:


# Concatenate three features as input feature vectors

ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec

# extracted feature vectors
x_test = np.array([to_feature_array(doc) for doc in tqdm(train_all)])
print(x_test[:10])


# In[16]:


label_to_int = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']

#labelled output
y_test = np.array([label_to_int[i] for i in train_all[:, 2]])
print(y_test[:10])


# In[31]:


from sklearn.ensemble import GradientBoostingClassifier
GradientBoosting_clf = GradientBoostingClassifier()
GradientBoosting_clf.fit(x_test, y_test)


# In[32]:


saved_GradientBoosting_classifier = open('saved_classifier/GradientBoosting_classifier_50_0.1.pkl', 'wb') 
pickle.dump(GradientBoosting_clf, saved_GradientBoosting_classifier)
saved_GradientBoosting_classifier.close()


# In[ ]:


# # beside gradient Boosting, linear regression and gaussian naive bayes can be tried
# # these three classifiers give similar accuracy

# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# regr.fit(x_test, y_test)

# saved_linear_model_classifier = open('saved_classifier/linear_model_classifier_50_0.1.pkl', 'wb') 
# pickle.dump(regr, saved_linear_model_classifier)
# saved_linear_model_classifier.close()

# from sklearn.naive_bayes import GaussianNB
# GaussianNB_clf = GaussianNB()
# GaussianNB_clf.fit(x_test, y_test)

# saved_GaussianNB_classifier = open('saved_classifier/GaussianNB_classifier_50_0.1.pkl', 'wb') 
# pickle.dump(GaussianNB_clf, saved_GaussianNB_classifier)
# saved_GaussianNB_classifier.close()

