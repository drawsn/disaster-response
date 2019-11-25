
# Disaster Response App
The purpose of this App is to provide a categorical classification of messages to help in disastaster situations. Therefore a ETL pipeline is created that can process and prepare data that is then saved to a sqlite database. Based on the training data a ML model is created to classify the messages into the appropriate categories. The final result is a web app that shows relevant categories for every text message that gets inputed.

### Table of Contents

1. [Installation](#installation)
2. [How to start](#howto)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using the Python versions 3.*.

The python libraries used are:
- NumPy
- Pandas
- plotly
- flask
- Sklearn
- NLTK
- sqlalchemy
- re
- json


## How to start <a name="howto"></a>
Use the following script from the main directory to create the prediction model used by the app.

1. Create a database based on a dataset of messages:

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

3. Setup and train the model:

    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Start the web-app by navigating to the app/ folder and running:

    `python run.py`

6. You can now access the app at http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # contains the etl pipeline to prepare the data
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # creates a ML model based on the data from the sqlite database
|- classifier.pkl  # saved model 

- README.md
```

## Results<a name="results"></a>
The results are only party satisfying and still need more parameter tweaking. Also the underlying dataset could use some more variety, as some categories are not or only seldomly occuring.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to [Figure Eight](https://www.figure-eight.com/) for the distaster data and classifications.