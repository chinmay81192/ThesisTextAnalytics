#Importing libraries
import nltk
import pandas as pd
import re
import time
import concurrent.futures
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from glove import Corpus, Glove

#Setting up regular expression patterns to remove html tags and punctuation excluding single quotes and space
include_punc = "' "
remove_html = re.compile(r"(<.*?>)")
remove_punc = re.compile(r"[^\w"+include_punc+"]")

#Instantiating Porter Stemmer
stemmer = PorterStemmer()

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

def remove_stop_words(tokenized_sent):
    return list(filter(lambda word: word not in stopwords.words('english'), tokenized_sent))

def process_sync(data):
    special_chars_removed = remove_special_chars(data)
    words_tokenized = special_chars_removed.split()
    stopwords_removed = remove_stop_words(words_tokenized)
    stemmed_words = [stemmer.stem(words) for words in stopwords_removed]
    return stemmed_words

def preprocess(user_reviews):
    processed_reviews = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_reviews = list(executor.map(process_sync, user_reviews))
    return processed_reviews

def getWordEmbeddings(processed_text):
    corpus = Corpus()
    corpus.fit(processed_text, window=3)
    glove = Glove(no_components = 500, learning_rate = 0.001)
    glove.fit(corpus.matrix, epochs = 300000, no_threads = 4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    print(glove.most_similar('price'))



if __name__ == '__main__':
    #Read from json to pandas dataframe
    df = pd.read_json("Automotive_5.json", lines=True)
    #Get the relevent columns, in this case it would be 'overall' which provides the rating and 'reviewText' which is the actual feedback
    reviews_df = df[["overall", "reviewText"]]
    user_reviews = df["reviewText"]
    start = time.time()
    processed_reviews = preprocess(user_reviews)
    print(time.time() - start)
    print("Now proceeding to create word embeddings")
    start = time.time()
    getWordEmbeddings(processed_reviews)
    print("Time taken to train word embeddings "+str(time.time() - start))

