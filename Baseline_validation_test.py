
# coding: utf-8

# In[1]:


import csv
import math
import numpy as np
import pandas as pd
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
val_unrelated = []
val_discuss = []
val_agree = []
val_disagree = []

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
val_stances = open('Dataset/validation_data.csv', 'r', encoding='utf-8')
csv_stances = csv.DictReader(val_stances)

#mapping
for row in csv_stances:
    body = bodies[int(row['Body ID'])]
    if row['Stance'] == 'unrelated':
        val_unrelated.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'discuss':
        val_discuss.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'agree':
        val_agree.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'disagree':
        val_disagree.append((row['Headline'], body, row['Stance']))

val_stances.close()


print('\tUnrltd\tDiscuss\t Agree\tDisagree')
print('Train\t', len(val_unrelated), '\t', len(val_discuss), '\t', len(val_agree), '\t', len(val_disagree))


# In[4]:


val_all  = (val_unrelated + val_discuss + val_agree + val_disagree) 
random.Random(0).shuffle(val_all)
val_all = np.array(val_all)


print('Val (Total)', val_all.shape)



# In[5]:


# tokenise text excluding punctuation, symbols, digits and stopwords

pattern = re.compile("[^a-zA-Z0-9 ]+") 

stop_words_nltk = list(stopwords.words('english'))
stop_words_sklearn = list(stop_words.ENGLISH_STOP_WORDS)
stop_words = stop_words_nltk + stop_words_sklearn



def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text



# In[6]:


# compute frequency of term

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
    print(val_all[i, 0], extract_tf(val_all[i, 0]))
print(val_all[0, 0], extract_tf(val_all[0, 0], ngram=2))


# In[7]:


#create a corpus with all headlines and article bodies

corpus = np.r_[val_all[:, 1], val_all[:, 0]] 


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

print(list(df.items())[:10])


# In[9]:


# compute inverse document frequency of every word in corpus

num_docs = corpus.shape[0]
idf = defaultdict(float)
for word, val in tqdm(df.items()):
    idf[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf

print(list(idf.items())[:10])


# In[10]:


# pre-trained 50-dimensional word embedding vector from GLoVe 

f_glove = open("glove_6B/glove.6B.50d.txt", "rb")
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
    if np.sum(list(doc_tf_idf.values())) == 0.0:  # edge case: document is empty
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
    # unrelated should have lower than rest
    print(cosine_similarity(val_all[i]), val_all[i, 2])




# In[12]:





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
    print(kl_divergence(val_all[i]), val_all[i, 2])


# In[14]:


# Extract ngram_overlap features
# It means Returns how many times n-grams (up to 3-gram) that occur both in headline and article body.

def ngram_overlap(doc):
    tf_headline = extract_tf(doc[0], ngram=3)
    tf_body = extract_tfdoc_to_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_headline.keys():
        if words in tf_body:
            matches += tf_body[words]
            
    # normalise for document length
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  

for i in range(5):
    print(ngram_overlap(val_all[i]), val_all[i, 2])


# In[15]:


# Concatenate three features as input feature vectors

ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec

# extracted feature vectors
x_test = np.array([to_feature_array(doc) for doc in tqdm(val_all)])
print(x_test[:5])


# In[16]:


label_to_int = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']

#labelled output
y_test = np.array([label_to_int[i] for i in val_all[:, 2]])
print(y_test[:10])


# In[18]:


# saved_linear_model_classifier = open("saved_classifier/linear_model_classifier_50_0.1.pkl","rb")
# regr=pickle.load(saved_linear_model_classifier)
# saved_linear_model_classifier.close()


# saved_GaussianNB_classifier = open("saved_classifier/GaussianNB_classifier_50_0.1.pkl","rb")
# GaussianNB_clf=pickle.load(saved_GaussianNB_classifier)
# saved_GaussianNB_classifier.close()


saved_GradientBoosting_classifier = open("saved_classifier/GradientBoosting_classifier_50_0.1.pkl","rb")
GradientBoosting_clf=pickle.load(saved_GradientBoosting_classifier)
saved_GradientBoosting_classifier.close()


# In[19]:


# regr  = regr.predict(x_test)
# GaussianNB_clf = GaussianNB_clf.predict(x_test)

GradientBoosting_clf = GradientBoosting_clf.predict(x_test)


# In[20]:


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


# In[25]:


report_score([LABELS[e] for e in y_test],[LABELS[e] for e in GradientBoosting_clf])


# In[ ]:


# # beside gradient Boosting, linear regression and gaussian naive bayes can be tried
# # these three classifiers give similar accuracy

# report_score([LABELS[e] for e in y_test],[LABELS[e] for e in GaussianNB_clf])
# report_score([LABELS[e] for e in y_test],[LABELS[e] for e in regr])


# In[28]:


# read test set

test = pd.read_csv("Dataset/test_data.csv")
test["Body ID"] == 172

bodies = pd.read_csv("article_body_texts.csv")

merged_data = pd.merge(test, bodies, how='left', left_on=['Body ID'], right_on=['Body ID'])
merged_data.head()

merged_data1 = merged_data.drop('Body ID',1)

merged = merged_data1.values.tolist()


# In[38]:


# tokenise text excluding punctuation, symbols,digits and stopwords

pattern = re.compile("[^a-zA-Z0-9 ]+")  

stop_words_nltk = list(stopwords.words('english'))
stop_words_sklearn = list(stop_words.ENGLISH_STOP_WORDS)
stop_words = stop_words_nltk + stop_words_sklearn

def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text


# In[40]:


#create a corpus with all headlines and article bodies

corpus_test = np.r_[test_set[:, 1], test_set[:, 0]]  # 0 to 44973 are bodies, 44974 to 89943 are headlines


# In[41]:


#compute ducument frequency of every word in corpus

df1 = defaultdict(float)
for doc in tqdm(corpus_test):
    words = tokenise(doc)
    seen = set()
    for word in words:
        if word not in seen:
            df1[word] += 1.0
            seen.add(word)

print(list(df1.items())[:5])


# In[42]:


#compute ducument frequency of every word in corpus

num_docs_test = corpus_test.shape[0]
idf_test = defaultdict(float)
for word, val in tqdm(df1.items()):
    idf_test[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf

print(list(idf_test.items())[:5])


# In[43]:


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
    print(cosine_similarity(test_set[i]))




# In[45]:


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
    
    # Compute a simple 1-gram language model of headline and body
    lm_headline = vec_headline + eps
    lm_headline /= np.sum(lm_headline)
    lm_body = vec_body + eps
    lm_body /= np.sum(lm_body)
    
    
    return divergence(lm_headline, lm_body)

for i in range(5):
    print(kl_divergence(val_all[i]))


# In[ ]:


# Extract ngram_overlap features
# It means Returns how many times n-grams (up to 3-gram) that occur both in headline and article body.

def ngram_overlap(doc):
    tf_headline = extract_tf(doc[0], ngram=5)
    tf_body = extract_tfdoc_to_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_headline.keys():
        if words in tf_body:
            matches += tf_body[words]
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  # normalise for document length

for i in range(5):
    print(ngram_overlap(test_set[i]))


# In[ ]:


# Concatenate three features as input feature vectors

ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec

# extracted feature vectors
x_test_final = np.array([to_feature_array(doc) for doc in tqdm(test_set)])
print(x_test_final[:5])


# In[ ]:


# saved_linear_model_classifier = open("saved_classifier/linear_model_classifier_50_0.1.pkl","rb")
# regr=pickle.load(saved_linear_model_classifier)
# saved_linear_model_classifier.close()

# saved_GaussianNB_classifier = open("saved_classifier/GaussianNB_classifier_50_0.1.pkl","rb")
# GaussianNB_clf=pickle.load(saved_GaussianNB_classifier)
# saved_GaussianNB_classifier.close()

saved_GradientBoosting_classifier = open('saved_classifier/GradientBoosting_classifier_50_0.1.pkl', 'rb') 
GradientBoosting_clf=pickle.load(saved_GradientBoosting_classifier)
saved_GradientBoosting_classifier.close()


# In[ ]:


# test_prediction_regr = regr.predict(x_test_final)

# test_prediction_GaussianNB_clf = GaussianNB_clf.predict(x_test_final)

test_GradientBoosting_clf = GradientBoosting_clf.predict(x_test_final)


# In[ ]:


# prediction_GaussianNB = []
# for i in test_prediction_regr:
#     if i == 0:
#         prediction_regr.append ('agree' )  
#     elif i == 1:
#         prediction_regr.append ('disagree') 
#     elif i == 2:
#         prediction_regr.append ('discuss')
#     else:
#         prediction_regr.append ('unrelated')


# In[ ]:


# prediction_GaussianNB = []
# for i in test_prediction_GaussianNB_clf:
#     if i == 0:
#         prediction_GaussianNB.append ('agree' )  
#     elif i == 1:
#         prediction_GaussianNB.append ('disagree') 
#     elif i == 2:
#         prediction_GaussianNB.append ('discuss')
#     else:
#         prediction_GaussianNB.append ('unrelated')


# In[ ]:


prediction_GradientBoosting = []
for i in test_GradientBoosting_clf:
    if i == 0:
        prediction_GradientBoosting.append ('agree' )  
    elif i == 1:
        prediction_GradientBoosting.append ('disagree') 
    elif i == 2:
        prediction_GradientBoosting.append ('discuss')
    else:
        prediction_GradientBoosting.append ('unrelated')


# In[57]:


# np.savetxt("prediction_regr_50.csv",prediction_regr,fmt='%s')


# In[58]:


# np.savetxt("prediction_GaussianNB_50.csv",prediction_GaussianNB,fmt='%s')


# In[59]:


np.savetxt("prediction_GradientBoosting_50.csv",prediction_GradientBoosting,fmt='%s')

