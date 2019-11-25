
# Disaster Response App
Disaster Response pipeline using python


# How to start
Use the following script from the main directory to create the prediction model used by the app.
1. Create a database based on a dataset of messages:

      python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Setup and train the model:

      python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Start the web-app by navigating to the app/ folder and running: python run.py

4. You can now access the app at http://0.0.0.0:3001/
