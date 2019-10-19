import json
import numpy as np
import nltk
import os
import pandas as pd
import re
nltk.download('punkt')
from math import log
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask
from flask import request
from flask import render_template
from random import choice
import datetime
from flask import url_for, render_template, request, redirect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import normalize
import pickle
import pymorphy2
from math import log
import tensorflow as tf

morph = pymorphy2.MorphAnalyzer()
trained_size = 10000  # constant that defines further size of corpus for models to be trained on


def lemmatize(list_of_words):
    return list(map(lambda word: morph.parse(word)[0].normal_form, list_of_words))


def preprocess(number):
    words = nltk.word_tokenize(number)
    words = [word.lower() for word in words]
    return words


def enum_sort_tuple(arr):
    return sorted(enumerate(arr), key=lambda x: x[1], reverse=True)


def cos_sim(v1, v2):
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_data():
    questions = pd.read_csv( "quora_question_pairs_rus.csv", index_col=0).dropna()
    train_texts = questions[:trained_size]['question2'].tolist()
    preprocessed = list(map(lambda sent: " ".join(preprocess(sent)), train_texts))
    return preprocessed


def get_counts(list_of_texts):
    count_vectorizer = CountVectorizer(input='content', encoding='utf-8')
    X = count_vectorizer.fit_transform(list_of_texts)
    count_matrix = X.toarray()
    return count_matrix, count_vectorizer


class DataSet:
    def __init__(self, lemm=False):
        self.texts = get_data()
        if lemm:
            self.lemmatized_texts = list(map(lambda sent: " ".join(lemmatize(sent.split())), self.texts))
            self.count_matrix, self.count_vectorizer = get_counts(self.lemmatized_texts)
            path = "lemmatized_count_vectorizer.pickle"
        else:
            path = "raw_count_vectorizer.pickle"
            self.count_matrix, self.count_vectorizer = get_counts(self.texts)
        with open(path, 'wb') as f:
            pickle.dump(self.count_vectorizer, f)


data_lemm = DataSet(lemm=True)
data_raw = DataSet()


class SearhTfidf:

    def __init__(self, path_tfidf_matrix=""):
        self.count_vectorizer = data_lemm.count_vectorizer
        if path_tfidf_matrix:
            self.matrix = self.load(path_tfidf_matrix)
        else:
            self.matrix = self.index()

    @staticmethod
    def load(path_tfidf_matrix):
        with open(path_tfidf_matrix, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def index(path="lemmatized_normalized_tf_idf_matrix.pickle"):
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(data_lemm.count_matrix).toarray()
        matrix = normalize(tfidf_matrix)
        with open(path, 'wb') as f:
            pickle.dump(matrix, f)
        return matrix

    def search(self, text, n=10):
        if n >= trained_size:
            n = trained_size - 1
        vector = self.count_vectorizer.transform([text]).toarray()
        norm_vector = normalize(vector).reshape(-1, 1)
        cos_sim_list = np.dot(self.matrix, norm_vector)
        clean_cos_sim_list = list(map(lambda x: x[0], cos_sim_list))
        return list(map(lambda tup: tup + (data_lemm.texts[tup[0]],), enum_sort_tuple(clean_cos_sim_list)[:n]))


def tf_idf_search(x):
    tfidf = SearhTfidf("lemmatized_normalized_tf_idf_matrix.pickle")
    SearhTfidf()
    return tfidf.search(x)


## bm25
k = 2.0
trained_size = 2500


def sort_list(arr):
    return [x[0] for x in sorted(enumerate(arr), key=lambda x:x[1], reverse=True)]


questions = pd.read_csv('quora_question_pairs_rus.csv')
train_texts = questions[:trained_size]['question2'].tolist()
with open('lemmatized.json', 'w') as f:
    f.write(json.dumps(train_texts))
with open('lemmatized.json', 'r') as f:
    train_texts = json.loads(f.read())[:trained_size]
text_length = [len(text.split()) for text in train_texts]
avgdl = sum(text_length)/ trained_size
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_texts)
count_matrix = X.toarray()
tf_matrix = count_matrix / np.array(text_length).reshape((-1, 1))
vocabulary = vectorizer.get_feature_names()


def preprocess(text, lemm=False):
    if lemm:
        words = lemmatize(text)
    else:
        words = nltk.word_tokenize(text)
    query_modified = list(set(words).intersection(set(vocabulary)))
    return query_modified

in_n_docs = np.count_nonzero(count_matrix, axis=0)


def IDF_modified(word):
    word_id = vocabulary.index(word)
    n = in_n_docs[word_id]
    return log((trained_size - n + 0.5) / (n + 0.5))


def modify_tf(tf_value, doc_index, b=0.75):
    l = text_length[doc_index]
    return (tf_value * (k + 1.0))/(tf_value + k * (1.0 - b + b * (l/avgdl)))


idfs = [IDF_modified(word) for word in vocabulary]


def modify_tf_matrix(tf_matrix, b=0.75):
    enumed =  np.ndenumerate(tf_matrix)
    for i, tf_value in enumed:
        doc_index = i[0]
        tf_matrix[i] = modify_tf(tf_value, doc_index, b)
    return tf_matrix


modified_tf_matrix = modify_tf_matrix(tf_matrix)

query = ' привет '


def bm25_vector(query, lemm=False):
    vector = np.array(vectorizer.transform([' '.join(preprocess(query, lemm))]).todense())[0]
    binary_vector = np.vectorize(lambda x: 1.0 if x != 0.0 else 0.0)(vector)
    idfs_from_query = np.array(idfs)*np.array(binary_vector)
    return modified_tf_matrix.dot(idfs_from_query)


def bm25_iter(query, lemm=False):
    query_words = preprocess(query, lemm)
    relevance = []
    for i in range(trained_size):
        doc_index = i
        doc_bm25 = 0.0
        for word in set(query_words):
            word_index = vocabulary.index(word)
            tf_value = tf_matrix[(doc_index, word_index)]
            doc_bm25 += idfs[word_index] * modify_tf(tf_value, doc_index)
        relevance.append(doc_bm25)
    return relevance


def bm_25_search(query, lemm=False, n=trained_size, nonzero=False, vector=True):
    if vector:
        bms = bm25_vector(query, lemm)
    else:
        bms = bm25_iter(query, lemm)
    relevance_sorted_document_ids_top_n = sort_list(bms)[:n]
    return [(rank, index, np.array(train_texts)[index], bms[index]) for rank, index in
            enumerate(relevance_sorted_document_ids_top_n)]


response = bm_25_search(query, n=10, lemm=True)

##fasttext
model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
questions = pd.read_csv("quora_question_pairs_rus.csv", index_col=0).dropna()
train_texts = questions[:2500]['question2'].tolist()


def doc_to_vec(lemmas):
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    doc_vec = np.zeros((model.vector_size,))
    for idx, lemma in enumerate(lemmas):
        if lemma in model.vocab:  # word in vocab
            try:
                lemmas_vectors[idx] = model[lemma]
            except AttributeError as e:
                print(e)
    if lemmas_vectors.shape[0] is not 0:
        doc_vec = np.mean(lemmas_vectors, axis=0)
        return doc_vec


fasttext_doc2vec_corpus = []
for doc in train_texts:
    fasttext_doc2vec_corpus.append(doc_to_vec(lemmatize(preprocess(doc))))


def search_fasttext(query, n=10):

    query_vec = doc_to_vec(lemmatize(preprocess(query)))
    cos_sim_relevance = [cos_sim(query_vec, document) for document in fasttext_doc2vec_corpus]
    relevance_sorted_document_ids_top_n = enum_sort_tuple(cos_sim_relevance)[:n]
    return [(metric, train_texts[index]) for index, metric in
            relevance_sorted_document_ids_top_n]
