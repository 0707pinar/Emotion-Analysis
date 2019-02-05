#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pinararslan

    Linear Support Vector Classification-based Emotion Classifier in English
    ________________________________________________________________________
    - 5 classes to be predicted: anger
                                 fear
                                 sadness
                                 joy
                                 other
    - Features used:
        word n-grams
        character n-grams
        [3] emotion lexicon (i.e., EmoLex by Saif Mohammad)
        [4] SenticNet concept-level features 
                       
    ________________________________________________________________________
    - Datasets:
      [1] "wassa2017[tra-devsets].json" 
          * Training and development sets of Wassa2017 dataset was used 
          * We used the tweets with >=0.50 emotion intensity score
          * 1808 tweets were obtained from Wassa2017 dataset.
          
      [2] "updated_instagram1000Comments[annotated].json"
          * 10 sessions were selected from Instagram dataset[2a, 2b].
          * 1000 Instagram posts were annotated with emotion, sentiment and bullying labels
          * To see the annotated Instagram data, please visit the following github link:
          ==> https://github.com/0707pinar/Post-level-bullying-emotion-polarity-annotation
    
    ________________________________________________________________________
    - References:
      [1]  WASSA-2017 Shared Task on Emotion Intensity. Saif M. Mohammad and Felipe Bravo-Marquez. In Proceedings of the EMNLP 2017 Workshop on Computational Approaches to Subjectivity, Sentiment, and Social Media (WASSA), September 2017, Copenhagen, Denmark.
      [2a] Homa Hosseinmardi, Sabrina Arredondo Mattson, Rahat Ibn Rafiq, Richard Han, Shivakant Mishra, Qin Lv, Analyzing Labeled Cyberbullying Incidents on the Instagram Social Network, accepted in 7th international Conference of Social Informatics, LNCS 9471, pp. 49–66, 2015 (SocInfo2015).
      [2b] Homa Hosseinmardi, Rahat Ibn Rafiq, Richard Han, Qin Lv and Shivakant Mishram, Prediction of Cyberbullying Incidents in a Media-based Social Network. In ASONAM 2016: Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, 2016.
      [3]  Saif M Mohammad and Peter D Turney. 2013. Crowdsourcing a word–emotion association lexicon. Computational Intelligence 29, 3 (2013), 436–465.
      [4]  Erik Cambria, Soujanya Poria, Devamanyu Hazarika, and Kenneth Kwok. 2018. SenticNet 5: discovering conceptual primitives for sentiment analysis by means of context embeddings. In Proceedings of AAAI, 2018.]
"""

import numpy as np
from scipy.sparse import hstack
from preprocessing_steps import extract_senticnet_features, preprocess_emotion_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


""" OUR PROPOSED EMOTION CLASSIFICATION MODEL """
# preprocessing the dataset (i.e., merged twitter and annotated instagram dataset) 
X, y, y_emolex = preprocess_emotion_dataset()

# for adding semantics features and polarity intensity scores (SenticNet used) 
X_senticnet, X_polarity_intensity = extract_senticnet_features()

  
# X_str is needed for character-gram model        
X_str = []
for x in X:
    X_str.append(" ".join(x))


tfidf = True

if tfidf:
    word_vec = TfidfVectorizer(preprocessor = lambda i:i,
                          tokenizer = lambda i:i,
                          stop_words='english',
                          ngram_range=(1,2))


    char_vec = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(3, 5)
        )

    emolex_vec = TfidfVectorizer(preprocessor = lambda i:i,
                          tokenizer = lambda i:i,
                          ngram_range =(1,1))
    
    
    
    senticnet_vec = TfidfVectorizer(preprocessor = lambda i:i,
                          tokenizer = lambda i:i,
                          ngram_range =(1,1))
    
else:
    vec = CountVectorizer(preprocessor = lambda i:i,
                          tokenizer = lambda i:i)



# Choose the classifier
cls = LinearSVC(random_state=0, class_weight="balanced")


# Choose the features you want to use
use_char_gram = True
use_emolex = True
use_senticnet = True
use_polarity_intensity = True


# Fit all vectorizers
words_vectorized = word_vec.fit_transform(X)
chars_vectorized = char_vec.fit_transform(X_str)
emolex_vectorized = emolex_vec.fit_transform(y_emolex)
senticnet_vectorized = senticnet_vec.fit_transform(X_senticnet)


# hstack to concatenate all selected vectors    
hstack_Xlist = [words_vectorized]


# add character gram-based features
if (use_char_gram):
    hstack_Xlist.append(chars_vectorized)


# add emolex features
if (use_emolex):
    hstack_Xlist.append(emolex_vectorized)


# add senticnet features
if use_senticnet:
    hstack_Xlist.append(senticnet_vectorized)


# get an averaged polarity intensity score per message
X_avg_polarity = np.zeros((len(X_polarity_intensity),1))  
for idx,message in enumerate(X_polarity_intensity): # a message shows polarity intensity scores for polarity-bearing words
    X_avg_polarity[idx,0] = np.sum(message)/max(len(message),1)

# use averaged polarity intensity
if use_polarity_intensity:
    hstack_Xlist.append(X_avg_polarity)

# concatenate the lists as a single matrix
hstack_finalX = hstack(hstack_Xlist)

# Fit the model for the merged dataset
Accuracy_mean_scores = cross_val_score(cls, hstack_finalX, y, cv=10, scoring="accuracy").mean()
print("The accuracy 'mean' score of the classifier (10-fold cross-validated): ",Accuracy_mean_scores)


# Predict the emotion classes
Ypredicted = cross_val_predict(cls, hstack_finalX, y, cv=10)

# Evaluate the model
from sklearn.metrics import precision_recall_fscore_support
print("macro average scores", precision_recall_fscore_support(y, Ypredicted, average = "macro")) 
print(classification_report(y, Ypredicted))  
print(confusion_matrix(y, Ypredicted)) 