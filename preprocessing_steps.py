#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pinararslan
PREPROCESSING SOCIAL MEDIA DATASETS (Twitter & Instagram Posts)
"""

import json
import re
from spell_check_enchant import spell_check
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from senticnet.senticnet import SenticNet

sn=SenticNet()
stemmer = SnowballStemmer("english")


# This opens and load json file.
def load_json_file(json_data): 
    jsonfile = open(json_data)
    jsonload = json.load(jsonfile)
    
    return jsonload
    
# This gets the input (X) and emotion labels (y) in the json data.
def extract_text_and_emotion_labels(loaded_json_data):
    X = [] # text of tweets
    y = [] # emotion labels
    for e in loaded_json_data:
        X.append(e["text"])
        y.append(e["emotion_label"])
    return X,y

# This gets the emotion, polarity, bullying annotations and Instagram posts
def get_insta_comments_annotations(json_file):
    annotations = []        # contains all annotations for emotion, polarity, bullying for each Instagram post
    instagram_comments = [] # contains 1000 instagram comments (i.e., Instagram posts)
    for j in json_file: #all sessions
        for c in j["columns"]:
            instagram_comments.append(c[0])
            annotations.append(c[1])
    return instagram_comments, annotations

# This returns annotated emotion, polarity and bullying labels for 1000 Instagram posts.
def get_emotion_polarity_bullying_labels(all_labels):
    annotated_emotion_labels = []
    annotated_polarity_labels = []
    annotated_bullying_labels = []

    all_annotations = [] #This contains all annotations (e.g. anger,negative, bullying) for 1000 comments

    for a in all_labels:
    
        annotated_emotion_labels.append(" ".join(a["emotion"]))
        annotated_polarity_labels.append(" ".join(a["polarity"]))
        annotated_bullying_labels.append(" ".join(a["bullying"]))
        
        all_annotations.append("{},{},{}".format(" ".join(a["emotion"])," ".join(a["polarity"]), " ".join(a["bullying"])))

    return annotated_emotion_labels, annotated_polarity_labels, annotated_bullying_labels

# This replaces URLS with URL tag
def tag_URLs(texts):
    texts_with_URLabels = []
    for s in texts:
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', s)
        texts_with_URLabels.append(text)
    return texts

# This removes unnecessary new line symbols in the text  
def remove_abundant_new_line_symbols(text_in_list):
    text_without_newline_symbols = [] 
    for line in text_in_list:
            line_string = "".join(line)
            rexp_string = r'\\n+'
            tags_created_rexp = re.compile(rexp_string)
            text_cleaned = tags_created_rexp.sub(" ",line_string)
            text_without_newline_symbols.append(text_cleaned)
    return text_without_newline_symbols

# This removes hashtags (e.g. #peace --> peace)
def remove_hashtag_symbols(text_in_list):
    text_without_hashtag_symbols = [] 
    for line in text_in_list:
            line = line.strip().split()
            line_string = " ".join(line)
            rexp_string = r'#'
            tags_created_rexp = re.compile(rexp_string)
            text_cleaned = tags_created_rexp.sub(" ",line_string)
            text_without_hashtag_symbols.append(text_cleaned)
    return text_without_hashtag_symbols

# This replaces usernames with username tag (this is needed for Instagram posts)
def replace_owners_with_username(text_in_list):
    X_with_username = []
    X = []
    for x in text_in_list:
        X_with_username.append(x.split())

    for x in X_with_username:
        x[0] = "USERNAME"
        X.append(x)    
    return X

# This puts a whitespace before and after a punctuation mark for Twitter post
def add_whitespace_before_and_after_punctuation_for_tweets(text_in_list):
    selected_punctuations = '!\"#$%&()*+,_-./:;<=>?[\]^`{|}~'
    text_with_space_before_punctuation = []

    for line in text_in_list:
        text_with_space_before_punctuation.append(["".join(line).translate(str.maketrans({key: " {0} ".format(key) for key in selected_punctuations}))])

    tokenized_text_with_space_before_punctuation = []
    for line in text_with_space_before_punctuation:
        tokenized_text_with_space_before_punctuation.append("".join(line).strip().split()) 
    return tokenized_text_with_space_before_punctuation

# This puts a whitespace before and after a punctuation mark for Instagram post 
def add_whitespace_before_and_after_punctuation_for_instagram(text_in_list):
    selected_punctuations = '!\"#$%&()*+,_-./:;<=>?[\]^`{|}~'
    text_with_space_before_punctuation = []

    for line in text_in_list:
        text_with_space_before_punctuation.append([" ".join(line).translate(str.maketrans({key: " {0} ".format(key) for key in selected_punctuations}))])

    tokenized_text_with_space_before_punctuation = []
    for line in text_with_space_before_punctuation:
        tokenized_text_with_space_before_punctuation.append(" ".join(line).strip().split())
    return tokenized_text_with_space_before_punctuation

# This replaces usernames with username tag (e.g. @example --> USERNAME)
def replace_usernames(text_in_list):
    # text_in_list contains tokenized words 
    text_with_replaced_usernames = [] 
    for line in text_in_list:
        line_string = " ".join(line)
        rexp_string = r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([_A-Za-z0-9-_]+[A-Za-z0-9-_]+)'
        tags_created_rexp = re.compile(rexp_string)
        text_cleaned = tags_created_rexp.sub("USERNAME",line_string)
        text_with_replaced_usernames.append(text_cleaned.split())
    return text_with_replaced_usernames

# This reads NRC Emolex (Emotion lexicon) and returns lexicon (emotion bearing words) 
# and labels (emotion and polarity labels) 
def read_english_NRC_corpus(corpus_file): 
    lexicon = []  
    labels = []  
    
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split("\t")
            emo_pol_labels = "".join(tokens[1])
            words = "".join(tokens[0])
            
            labels.append(emo_pol_labels)
            lexicon.append(words)               
    return lexicon, labels

# This returns each emotion and polarity words obtained from NRC Emolex 
def extract_emolex_emotion_polarity_words():
    
    words, emo_pol_labels = read_english_NRC_corpus("english_NRC_emolex.txt")
    
    anger_words = []
    fear_words = []
    joy_words = []
    sadness_words = []
    trust_words = []
    surprise_words = []
    anticipation_words = []
    disgust_words = []
    positive_words = []
    negative_words = []

    for i,l in enumerate(emo_pol_labels):
        if l == "anger":
            anger_words.append(words[i])
        if l == "fear":
            fear_words.append(words[i])
        if l == "sadness":
            sadness_words.append(words[i])
        if l == "joy":
            joy_words.append(words[i])
        if l == "trust":
            trust_words.append(words[i])
        if l == "surprise":
            surprise_words.append(words[i])
        if l == "anticipation":
            anticipation_words.append(words[i])
        if l == "disgust":
            disgust_words.append(words[i])
        if l == "positive":
            positive_words.append(words[i])
        if l == "negative":
            negative_words.append(words[i])
        
    return anger_words, fear_words, joy_words, sadness_words, trust_words, surprise_words, anticipation_words, disgust_words, positive_words, negative_words

# This returns NRC Emolex labels for each given post
def emolex_labels_for_high_gram_models(text_in_list):
    anger_words, fear_words, joy_words, sadness_words, trust_words, surprise_words, anticipation_words, disgust_words, positive_words, negative_words = extract_emolex_emotion_polarity_words()

    emolex_labels = []
    
    for line in text_in_list:
        
        X_sentences = []
        
        for w in line:
        
            new_sentences = []
            lowered_words = w.lower()    
        
            if lowered_words in anger_words:
                label = "_ANGER_" 
                new_sentences.append(label)
                
            if lowered_words in sadness_words:
                label = "_SADNESS_"
                new_sentences.append(label)
    
            if lowered_words in joy_words:
                label = "_JOY_"
                new_sentences.append(label)
            
            if lowered_words in fear_words:
                label = "_FEAR_"
                new_sentences.append(label)
                
            if lowered_words in trust_words:
                label = "_TRUST_"
                new_sentences.append(label)
                
            if lowered_words in anticipation_words:
                label = "_ANTICIPATION_"
                new_sentences.append(label)
                
            if lowered_words in surprise_words:
                label = "_SURPRISE_"
                new_sentences.append(label)
                
            if lowered_words in disgust_words:
                label = "_DISGUST_"
                new_sentences.append(label)
                
            if lowered_words in positive_words:
                label = "_POSITIVE_"
                new_sentences.append(label)
            
            if lowered_words in negative_words:
                label = "_NEGATIVE_"
                new_sentences.append(label)
            
            X_sentences +=  new_sentences
        emolex_labels.append(X_sentences)
    return emolex_labels

# This applies stemmer on the words and replaces adversative conjunctions, negative items and
# numbers with tags (i.e., ADVERSATIVE, NEG, NUMBER) for each given post.
def stemmed_words_with_placeholders(text_in_list):
    # This does not contain any emotion or sentiment features
    text_with_stemmed_words_with_placeholders = []
    
    adversative_conjunctions_labels = ["but", "But", "BUT",
                                  "however","However", "HOWEVER",
                                  "although","Although","ALTHOUGH",
                                  "despite","Despite", "DESPITE",
                                  "though","Though","THOUGH",
                                  "yet","Yet","YET",
                                  "nevertheless","Nevertheless", "NEVERTHELESS",
                                  "still","Still","STILL"] 

    negative_items_labels = ["no","NO","No",
                             "NOT","not","Not",
                             "don't","DON'T","Don't",
                             "doesn't","DOESN'T","Doesn't",
                             "n't","N'T","N't",
                             "NEVER","never","Never",
                             "won't","WON'T", "Won't", 
                             "hasn't", "HASN'T","Hasn't", 
                             "haven't", "HAVEN'T", "Haven't", 
                             "didn't", "DIDN'T","Didn't", 
                             "aren't", "AREN'T", "Aren't", 
                             "isn't", "ISN'T", "Isn't", 
                             "wasn't","WASN'T","Wasn't",
                             "weren't","WEREN'T","Weren't", 
                             "cannot","can't","Cannot",
                             "couldn't","COULDN'T","Couldn't", 
                             "shouldn't","SHOULDN'T","Shouldn't", 
                             "wouldn't","WOULDN'T","Wouldn't", 
                             "mightn't","MIGHTN'T","Mightn't", 
                             "amn't","AMN'T","Amn't", 
                             "hadn't","HADN'T","Hadn't"]

    for line in text_in_list:
        
        numbers = re.findall(r"([0-9]+)"," ".join(line))
        X_sentences = []
        
        for w in line:
        
            new_sentences = []
            stemmed_words = stemmer.stem(w)
    
            if stemmed_words in adversative_conjunctions_labels:
                label = "ADVERSATIVE" 
                new_sentences.append(label)
                
            if stemmed_words in negative_items_labels:
                label = "NEG"
                new_sentences.append(label)
    
            if stemmed_words in numbers:
                label = "NUMBER"
                new_sentences.append(label)
                
            if stemmed_words not in negative_items_labels:
                if stemmed_words not in adversative_conjunctions_labels:
                    if stemmed_words not in numbers:
                        new_sentences.append(stemmed_words)
                        
            X_sentences += new_sentences
        text_with_stemmed_words_with_placeholders.append(X_sentences)
    return text_with_stemmed_words_with_placeholders


# This removes a whitespace within a token if any.
def remove_whitespace_inside_one_token(text_in_list):
    text_without_space_in_one_token = [] 
    for line in text_in_list:
        X_sentence = []
        for w in line:
            rexp_string = r' '
            tags_created_rexp = re.compile(rexp_string)
            text_cleaned = tags_created_rexp.sub("",w)
            X_sentence.append(text_cleaned)
        text_without_space_in_one_token.append(X_sentence)
    return text_without_space_in_one_token

# This preprocesses Twitter dataset (i.e., Wassa 2017 dataset, training&development sets with >=0.50 emotion intensity score)
# Output: preprocessed input (X), emotion labels to be predicted (y), NRC Emolex labels for each emotion bearing word in tweets (y_wassa_emolex)
def preprocess_tweet_highgram():    
    X, y = extract_text_and_emotion_labels(load_json_file("wassa2017[tra-devsets].json"))
    X_preprocessed = replace_usernames(add_whitespace_before_and_after_punctuation_for_tweets(remove_hashtag_symbols(remove_abundant_new_line_symbols(tag_URLs(X)))))
    y_wassa_emolex = emolex_labels_for_high_gram_models(X_preprocessed)
    X_additional_features = stemmed_words_with_placeholders(X_preprocessed)
    X, y = X_additional_features, y   
    return X, y, y_wassa_emolex

# This preprocesses Annotated Instagram dataset (https://github.com/0707pinar/Post-level-bullying-emotion-polarity-annotation)
# Output: preprocessed input (Xinsta), emotion labels to be predicted (yinsta), NRC Emolex labels for each emotion bearing word in Instagram posts (y_insta_emolex)   
def preprocess_annotated_instagram_highgram():
    our_json = load_json_file("updated_instagram1000Comments[annotated].json")
    instagram_comments, labels = get_insta_comments_annotations(our_json)
    emotion_labels, polarity_labels, bullying_labels = get_emotion_polarity_bullying_labels(labels)
    
    y_emotion_labels = []
   
    for e in emotion_labels:
        if e == "no_emotion":
            emo_label = "other"
        if e == "anger":
            emo_label = "anger"
        if e == "sadness":
            emo_label = "sadness"
        if e == "fear":
            emo_label = "fear"
        if e == "joy":
            emo_label = "joy"
        if e == "other":
            emo_label = "other"
        y_emotion_labels.append(emo_label)

    X_preprocessed = remove_hashtag_symbols(remove_abundant_new_line_symbols(tag_URLs(instagram_comments)))
    X = replace_usernames(replace_owners_with_username(X_preprocessed))
    X_corrected = []
    for x in X:
        X_corrected.append(spell_check(x))

    X = remove_whitespace_inside_one_token(X_corrected)     
    X = add_whitespace_before_and_after_punctuation_for_instagram(X)
    y_insta_emolex = emolex_labels_for_high_gram_models(X)
    X = stemmed_words_with_placeholders(X)
    Xinsta, yinsta = X, y_emotion_labels
    return Xinsta, yinsta, y_insta_emolex

# This returns the merged inputs, emotion labels, NRC Emolex labels obtained from Twitter and Instagram datasets.
def merge_twitter_annotated_instagram_for_highgram_model(X_insta, y_insta, y_insta_emolex, X_tweet, y_tweet, y_tweet_emolex):
    X_insta_tweet = X_tweet + X_insta 
    y_insta_tweet = y_tweet + y_insta
    y_insta_tweet_emolex = y_tweet_emolex + y_insta_emolex
    Xinsta_tweet_merged, yinsta_tweet_merged, yinsta_tweet_emolex_merged = X_insta_tweet, y_insta_tweet, y_insta_tweet_emolex
    return Xinsta_tweet_merged, yinsta_tweet_merged, yinsta_tweet_emolex_merged

# This applies preprocessing on twitter and instagram datasets and then merges the two datasets.
# Output: preprocessed & merged input, emotion labels to be predicted, NRC Emolex labels for emotion bearing words
def preprocess_emotion_dataset():
    Xtweet, ytweet, ytweet_emolex = preprocess_tweet_highgram()
    Xinsta, yinsta, yinsta_emolex = preprocess_annotated_instagram_highgram()
    merged_X, merged_y, merged_emolex = merge_twitter_annotated_instagram_for_highgram_model(Xinsta, yinsta, yinsta_emolex, Xtweet, ytweet, ytweet_emolex)
    return merged_X, merged_y, merged_emolex

# This returns the merged inputs, emotion labels obtained from Twitter and Instagram datasets
def merge_twitter_annotated_instagram(X_insta, y_insta, X_tweet, y_tweet):
    X_insta_tweet_training_set = X_tweet + X_insta 
    y_insta_tweet_training_set = y_tweet + y_insta
    Xinsta_tweet_train, yinsta_tweet_train = X_insta_tweet_training_set, y_insta_tweet_training_set
    return Xinsta_tweet_train, yinsta_tweet_train

# This returns the merged inputs, emotion labels obtained from Twitter and Instagram datasets [no preprocessing applied]
def merge_twitter_insta_datasets_baseline_system():
    Xtweet, ytweet = extract_text_and_emotion_labels(load_json_file("wassa2017[tra-devsets].json"))    
    our_json = load_json_file("updated_instagram1000Comments[annotated].json")
    Xinsta, labels = get_insta_comments_annotations(our_json)
    emotion_labels, polarity_labels, bullying_labels = get_emotion_polarity_bullying_labels(labels)
    
    yinsta = []
   
    for e in emotion_labels:
        if e == "no_emotion":
            emo_label = "other"
        if e == "anger":
            emo_label = "anger"
        if e == "sadness":
            emo_label = "sadness"
        if e == "fear":
            emo_label = "fear"
        if e == "joy":
            emo_label = "joy"
        if e == "other":
            emo_label = "other"
        yinsta.append(emo_label)
    
    merged_X, merged_y = merge_twitter_annotated_instagram(Xinsta, yinsta, Xtweet, ytweet)
    return merged_X, merged_y

# This returns Senticnet features (i.e., polarity values, moodtags, semantics, concept words) and polarity_intensity obtained from SenticNet
def extract_senticnet_features():
    X_senticnet = []
    X_polarity_intensity = []
    Xbase_step1, ybase = merge_twitter_insta_datasets_baseline_system()
    Xbase = add_whitespace_before_and_after_punctuation_for_tweets(Xbase_step1)
    
    for x in Xbase:
        messages = []
        x_pol_intensity = []
        for w in x:
            words = []
            semantics = []
            moodtags = []
            polarity_values = []
            polarity_intensity = []
            try:
                semantics.append(sn.semantics(w))
                moodtags.append(sn.moodtags(w))
                polarity_values.append([sn.polarity_value(w)])
                polarity_intensity.append([sn.polarity_intense(w)])
                
                if len(sn.semantics(w)) != 0:
                    words.append([w])
                
            except KeyError:
                pass
            
            messages += words + semantics + moodtags + polarity_values #+ polarity_intensity
            x_pol_intensity += polarity_intensity
            
        X_senticnet.append(messages)
        X_polarity_intensity.append(x_pol_intensity)
        
    # flatten X_senticnet  
    X_flattened_senticnet = []
    for i in X_senticnet:
        adjusted_sessions = []
        for s in i:
            lowered_s = []
            for w in s:
                lowered_s.append(w.lower())
            adjusted_sessions += lowered_s
        X_flattened_senticnet.append(adjusted_sessions)
    
    # flatten X_polarity_intensity
    X_flattened_polarity_intensity = []  
    for i in X_polarity_intensity:
        adjusted_sessions = []
        for s in i:
            integer_pol_intensity = []
            for w in s:
                integer_pol_intensity.append(float("".join(w)))
            adjusted_sessions += integer_pol_intensity
        X_flattened_polarity_intensity.append(adjusted_sessions)
    
    return X_flattened_senticnet, X_flattened_polarity_intensity

# This shows the label distribution for a given list of emotions to be predicted.
def label_distribution(labels):
    label_count = defaultdict(int)
    for l in labels:
        label_count[l] += 1
    return label_count