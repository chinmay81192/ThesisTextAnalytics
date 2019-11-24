#Importing libraries
import nltk
import pandas as pd
import re
import time
import concurrent.futures
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import gensim
import multiprocessing
from gensim.models import Word2Vec

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
    word_2_vec = Word2Vec(window = 3, size = 500, workers = multiprocessing.cpu_count(), sg = 1, hs = 1, alpha = 0.001)
    word_2_vec.build_vocab(processed_text, progress_per = 10000)
    word_2_vec.train(processed_text, total_examples = word_2_vec.corpus_count, epochs = 50, report_delay = 1)
    word_2_vec.init_sims(replace=True)
    print(word_2_vec.most_similar('price'))


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

