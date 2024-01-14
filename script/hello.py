import pandas as pd
import matplotlib.pyplot as plt 
import numpy, seaborn, sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('dataset/adjusted_mockDataset.csv')

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(data['Time'], format='%H:%M').dt.minute


# Extract datetime components
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday
 
# Drop the original 'Date' column
data = data.drop('Date', axis=1)

# Convert categorical variable 'Food' into dummy/indicator variables
data = pd.get_dummies(data, columns=['Food'])

# Drop rows with all missing values
data.dropna(how='all', inplace=True)

# Split the dataset into features and target variable
X = data.drop('Sales', axis=1)  # Features (all except Sales)
y = data['Sales']  # Target variable (Sales)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R-squared: {r2}")

# Plot
data.plot(kind='scatter', x='Temperature', y='Sales')
plt.show()

# Save the model
joblib.dump(model, 'sales_prediction_model.joblib')