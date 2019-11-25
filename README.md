
# Disaster Response App
Disaster Response pipeline using python


# How to start
Use the following script from the main directory to create the prediction model used by the app.
1. Create a database based on a dataset of messages:

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

3. Setup and train the model:

    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Start the web-app by navigating to the app/ folder and running:

    `python run.py`

6. You can now access the app at http://0.0.0.0:3001/

# File structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
