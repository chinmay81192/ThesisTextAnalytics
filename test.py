#Importing libraries
#import nltk
import pandas as pd
import re
import time
import concurrent.futures
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem.porter import *
import spacy

#Load the english corpus and get the stop words
spacy_en = spacy.load('en_core_web_sm')
en_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#Setting up regular expression patterns to remove html tags and punctuation excluding single quotes and space
include_punc = "' "
remove_html = re.compile(r"(<.*?>)")
remove_punc = re.compile(r"[^\w"+include_punc+"]")

#Instantiating Porter Stemmer
#stemmer = PorterStemmer()

def remove_special_chars(text):
    #convert to lower case
    text = text.lower()
    # remove html markup
    text = remove_html.sub("",text)
    #remove non-ascii and digits
    text = remove_punc.sub("",text)
    #remove whitespace
    text=text.strip()
    return text
'''
def remove_stop_words(tokenized_sent):
    return list(filter(lambda word: word not in stopwords.words('english'), tokenized_sent))
'''
def process_sync(data):
    special_chars_removed = remove_special_chars(data)
    spacy_sent = spacy_en(special_chars_removed)
    stop_words_removed = [word.text for word in spacy_sent if not word.is_stop]
    stop_words_removed = " ".join(stop_words_removed)
    sent_lemmatized = [word.lemma_ for word in spacy_en(stop_words_removed)]
    return sent_lemmatized

def preprocess(user_reviews):
    processed_reviews = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_reviews = list(executor.map(process_sync, user_reviews))
    print(processed_reviews[0])

if __name__ == '__main__':
    #Read from json to pandas dataframe
    df = pd.read_json("Automotive_5.json", lines=True)
    #Get the relevent columns, in this case it would be 'overall' which provides the rating and 'reviewText' which is the actual feedback
    reviews_df = df[["overall", "reviewText"]]
    user_reviews = df["reviewText"]
    start = time.time()
    preprocess(user_reviews)
    print(time.time() - start)
