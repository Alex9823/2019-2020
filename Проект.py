import json
import numpy as np
import nltk
import os
import pandas as pd
import re
nltk.download('punkt')
from math import log
# from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from flask import Flask
from flask import request
from flask import render_template
from random import choice
import datetime
from models import tf_idf_search, bm_25_search, search_fasttext
from flask import url_for, render_template, request, redirect

import logging
from logging.handlers import RotatingFileHandler


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def foo():
    app.logger.warning('A warning occurred (%d apples)', 42)
    app.logger.error('An error occurred')
    app.logger.info('Info')
    return 'foo'


@app.route('/result')
def search():
    number = request.args['number']
    name = request.args['tabs']
    a= ''
    if name == 'cit':
        a=str(tf_idf_search(number))
    else:
        if name == 'tfidf':
            query = number
            a = bm_25_search(query, n=10, lemm=True)
        else:
            if name == 'date':
                query = number
                a = search_fasttext(query)
            else:
                a="Not Found"
    return render_template('result.html', number=number,name=name,a=a)


if __name__ == '__main__':
    handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run()