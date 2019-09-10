import csv
import pandas as pd
import numpy as np
import re
import json
import TextAnalyticsService

from flask import Response
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rasa_nlu.model import Metadata, Interpreter
from sklearn.externals import joblib


class Classifier():

    def classifySentiment(self, input):
        input_text = input

        classifier = joblib.load('/tmp/sentiment_classifier.joblib.pkl')
        vectorizer = joblib.load('/tmp/tfidfVectorizer.pkl')

        input_counts = vectorizer.transform(input_text)
        predictions = classifier.predict(input_counts)

        return predictions
    
    def clean(self, input):
        list = []
        for sentence in input:
            sentence = sentence.lower()
            sentence = re.sub('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', "", sentence)
            sentence = re.sub('[^\w\s]', "", sentence)
            sentence = re.sub('\d', "", sentence)

            list.append(sentence)

        return list

    def removeStopWords(self, input):
        list = []
        for sentence in input:
            filtered_sent = []
            tokenized_word = word_tokenize(sentence)
            for w in tokenized_word:
                if w not in stopwords.words('english'):
                    filtered_sent.append(w)
                    cleanWord = " ".join(filtered_sent)
            list.append(cleanWord)

        return list
    def analyze(self, text):

        sentiments = {0 : "Negative", 1: "Positive"}

        textToClassify = text["data"][0]

        stringifiedtext = json.dumps(classifiedText)

        userText = text["data"]
        userText = myModel.clean(userText)
        userText = myModel.removeStopWords(userText)

        sentimentResult = myModel.classifySentiment(userText)

        response = {}
        response["sentiment"] = sentiments[sentimentResult.item(0)]

        return Response(json.dumps(response), mimetype='application/json')
