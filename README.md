# Fake_News_Challenge

This paper is a project report of “Fake News Challenge Stage 1 (FNC-1): Stance Detection”. This report tries to combat fake news problem by applying Artificial Intelligence (AI) technologies, including feature extraction, natural language processing, and deep learning. 

In the challenge, two types of text are given: news headline and news article body. The stance detection task of the challenge is to classify the stance relation between a news article body and a news headline into four categories: Agree, Disagrees, Discusses and Unrelated, which means the news headline agrees, disagree, discusses the same topic, or discusses a different topic with the corresponding news article body. In the dataset of challenge, the number of headlines is a lot more than article bodies; the length of each headline is much shorter than each article body. 

This project presents three different models of classification: 1. Long Short Term Memory Network (LSTM); 2. Convolutional Neural Network (CNN); 3. Multilayer Perceptron (MLP). Word Embedding Vectors of Word2vec and GloVe, Term Frequency (TF), and Term Frequency-Inverse Document Frequency (TF-IDF) are tried on different models. MLP model with combined feature of TFand TD-IDF achieved the highest accuracy of 86.6% on Validation Set and 2390.5 weighted score on Test Set. (85.2% for n = 200, 84.4% for n = 50)
