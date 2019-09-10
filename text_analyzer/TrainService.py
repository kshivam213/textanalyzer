import csv
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Metadata, Interpreter
from sklearn.externals import joblib

class Train():
  def train_model(self):

    #Amazon Data
    input_file = "/home/shivam/ShivamProject/text-analytics/datasets/amazon_cells_labelled.txt"
    amazon = pd.read_csv(input_file, delimiter = '\t', header = None)
    amazon.columns = ['Sentence', 'Class']

    # Yelp Data
    input_file = "/home/shivam/ShivamProject/text-analytics/datasets/yelp_labelled.txt"
    yelp = pd.read_csv(input_file, delimiter = '\t', header = None)
    yelp.columns = ['Sentence', 'Class']

    # Imdb Data
    input_file = "/home/shivam/ShivamProject/text-analytics/datasets/imdb_labelled.txt"
    imdb = pd.read_csv(input_file, delimiter = '\t', header = None)
    imdb.columns = ['Sentence', 'Class']

    # combine all data sets
    data = pd.DataFrame()
    data = pd.concat([amazon, yelp, imdb])
    data['index'] = data.index

    # Text Preprocessing
    columns = ['index', 'Class', 'Sentence']
    df_ = pd.DataFrame(columns = columns)

    data['Sentence'] = data['Sentence'].str.lower()# remove email adress
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex = True)# remove IP address
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex = True)# remove punctaitions and special chracters
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]', '')# remove numbers
    data['Sentence'] = data['Sentence'].replace('\d', '', regex = True)

    #Remove Stopwords
    for index, row in data.iterrows():
            word_tokens = word_tokenize(row['Sentence'])
            filtered_sentence = [w for w in word_tokens if not w in stopwords.words('english')]
            df_ = df_.append({"index": row['index'], "Class":  row['Class'],"Sentence": " ".join(filtered_sentence[0:])}, ignore_index=True)
    data = df_

    X_train, X_test, y_train, y_test = train_test_split(data['Sentence'].values.astype('U'),data['Class'].values.astype('int32'), test_size=0.10, random_state=0)
        
    vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), max_features = 50000, max_df = 0.5, use_idf = True, norm = 'l2')
    counts = vectorizer.fit_transform(X_train)

    # Storing vectorizzer file in tmp folder
    vectorizerfilename = "/tmp/tfidfVectorizer.pkl"
    joblib.dump(vectorizer, vectorizerfilename)

    classifier = SGDClassifier(alpha = 1e-05, max_iter = 50, penalty = 'elasticnet')
    targets = y_train
    classifier = classifier.fit(counts, targets) 
    example_counts = vectorizer.transform(X_test)
    predictions = classifier.predict(example_counts)

  # Storing sentiment file in tmp folder  
    sentimentfilename = '/tmp/sentiment_classifier.joblib.pkl'
    joblib.dump(classifier, sentimentfilename, compress = 9)

  def train(self):
    myModel = Train()
    myModel.train_model()

    return "Successfully trained sentiments dataset"