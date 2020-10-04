from bs4 import BeautifulSoup
from constants import *
import os
import re 

import numpy as np 
import pandas as pd 

def read_data(path, set_):
    """
    Function that reads a given set of data (train or val) and returns a dataframe
    :param path: full path to the data
    :param set_: str: train or test
    :return: df
    """
    test = dict()
    fmt = '_clean' if set_ == 'training' else ''
    test['tags'] = pd.read_csv(os.path.join(path, '{}_set{}_only_tags.txt'.format(set_, fmt)), header=None)
    test['tags'].columns = ['label']

    test['text'] = pd.read_table(os.path.join(path, '{}_set{}_only_text.txt'.format(set_, fmt)), header=None)
    test['text'].columns = ['text']


    test['tags']['label_text'] = test['tags']['label'].map({0: "non-harmful",
                                                              1: "cyberbullying",
                                                              2: "hate-speech"})

    df = pd.concat([test['tags'], test['text']], axis=1)

    
    return df#, X_test, y_test

def clean_text(pattern, text, tag=" "):
    """
    Function that cleans a sentence given a pattern and a tag to be replaced with
    :param pattern: regex 
    :param text: sentence
    :tag: if None, use empty space
    :return: clean new sentence
    """
    
    rgx_list = re.findall(pattern, text)
    
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, tag, new_text)
    return new_text

class SentenceCleaner(object):
    """
    Class that contains several methods for cleaning Tweets
    """
    
    def __init__(self, sentence):
        self.sentence = sentence
        
    def html_tag(self):
        self.sentence = BeautifulSoup(self.sentence).get_text()
        
    def urls(self):
        pattern = '(\w+:\/\/\S+)'
        self.sentence = clean_text(pattern, self.sentence )
        
    def handle(self):
        pattern = "@[^\s]+"
        self.sentence = clean_text(pattern, self.sentence, "użytkownik")

    def hashtag(self):
        pattern = '#[A-Za-z0-9]+'
        self.sentence = clean_text(pattern, self.sentence )
        
    def punctuations(self):
        pattern = '[\.\,\!\?\:\;\-\=]'
        self.sentence = clean_text(pattern, self.sentence )
        
    def emojis(self):
        self.sentence = "".join((emoji_encoder.get(c, c) for c in self.sentence))
    
    def tolowercase(self):
        self.sentence = self.sentence.lower()
        
        
def clean_tweet(tweet):
    """
    Function that cleans a single tweet
    """
    entry = SentenceCleaner(tweet)
    entry.html_tag()
    entry.urls()
    entry.handle()
    entry.hashtag()
    entry.tolowercase()
    return entry.sentence


