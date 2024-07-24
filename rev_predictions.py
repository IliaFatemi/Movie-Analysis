from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

movie_data = pd.read_csv('movie_dataset_cleaned_with_profits.csv')

# Convert runtimeMinutes to numeric, coerce errors to handle non-numeric values
movie_data['runtimeMinutes'] = pd.to_numeric(movie_data['runtimeMinutes'], errors='coerce')

# Drop rows with missing values
movie_data = movie_data.dropna()

# Define the features and target variable
X = movie_data[['startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'budget']]
y = movie_data['revenue']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the revenue on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs. Predicted Revenue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of perfect prediction
plt.show()