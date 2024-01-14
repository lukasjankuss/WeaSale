import pandas as pd
import matplotlib.pyplot as plt 
import joblib


# Load the dataset
data = pd.read_csv('dataset/weekly_forecast_updated.csv')

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(data['Time'], format='%H:%M').dt.minute


# Extract datetime components
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday

model = joblib.load('sales_prediction_model.joblib')

next_week_predictions = model.predict(next_week_data)