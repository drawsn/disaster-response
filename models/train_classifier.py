import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.externals import joblib

def load_data(database_filepath):
    # create database engine
    engine = create_engine('sqlite:///' + database_filepath)
    
    # load data from database
    df = pd.read_sql_table('messages', engine)
    
    # split data values into separate lists
    X = df.message.values
    Y = df.drop(['message', 'original', 'id', 'genre'], axis=1).values
    
    # get category names
    category_names = df.drop(['message', 'original', 'id', 'genre'], axis=1).columns
    
    
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    model = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2), max_features=5000, max_df=0.75)),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    for cat in range(len(Y_pred.T)):
        class_report = classification_report(Y_test.T[cat], Y_pred.T[cat])
        print("Classification report:", category_names[cat], "\n", class_report)


def save_model(model, model_filepath):
    # Output a pickle file for the model
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()