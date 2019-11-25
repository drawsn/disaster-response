import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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
    '''
    load the relevant data from the specified database filepath and return the appropriate dataframes and category names.
    
    INPUT:
    database_filepath - string: path to the database file
            
    OUTPUT:
    X - dataframe: contains the messages data
    Y - dataframe: contains the category data
    category_names - list of strings: labels of the message categories
    '''
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
    '''
    processes text messages and creates cleaned word tokens
    
    INPUT:
    text - string: text to tokenize
            
    OUTPUT:
    tokens - list of strings: containing all words from the inputed text
    ''' 
    # create lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # load english stop words
    stop_words = stopwords.words("english")    
    
    # remove punctuation and convert everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
    return tokens


def build_model():
    '''
    build the machine learning model
    
    INPUT:
    NONE
            
    OUTPUT:
    model - gridsearch object: a machine learninng model in the form of a gridsearchCV object
    '''     
    # create pipeline object
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    # define gridsearch parameter range
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
        }

    # create gridsearch object
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model and ouput a classification report for each category
    
    INPUT:
    model - ML model: the fitted machine learning model
    X_test - dataframe: the testdata category results
    Y_test - dataframe: the testdata messages
    category_names: list of strings: labels of the message categories
            
    OUTPUT:
    print of classification reports
    '''     
    Y_pred = model.predict(X_test)
    
    for cat in range(len(Y_pred.T)):
        class_report = classification_report(Y_test.T[cat], Y_pred.T[cat])
        print("Classification report:", category_names[cat], "\n", class_report)


def save_model(model, model_filepath):
    '''
    function to save a ML model as pickle file to the inputed filepath.
    
    INPUT:
    model - ML model: the fitted machine learning model
    model_filepath - string: filepath where the model should be saved
            
    OUTPUT:
    saves the model file
    '''      
    # save the final model as a pickle file
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=17)
        
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