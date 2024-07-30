from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np

movie_data = pd.read_csv('movie_dataset_cleaned_with_profits.csv')

# Convert runtimeMinutes to numeric, coerce errors to handle non-numeric values
movie_data['runtimeMinutes'] = pd.to_numeric(movie_data['runtimeMinutes'], errors='coerce')

# Drop Movies that have 0 revenue and runtimeMin and movies before 2000
movie_data = movie_data[(movie_data['revenue'] > 0) & (movie_data['runtimeMinutes'] > 0) & (movie_data['startYear'] > 2000)]

# Changing units of revenue to be in the millions for easier readibility 
movie_data['revenue'] = movie_data['revenue'] / 1000000

movie_data_clean = movie_data.replace('\\N', np.nan).dropna()

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
encoder = OneHotEncoder(sparse_output=False)
genres_encoded = encoder.fit_transform(movie_data_clean[['genres']])

# Add encoded genres back to the dataframe
encoded_genres_df = pd.DataFrame(genres_encoded, columns=encoder.get_feature_names_out(['genres']))
movie_data_clean = pd.concat([movie_data_clean.reset_index(drop=True), encoded_genres_df.reset_index(drop=True)], axis=1)

# Select relevant features for the model
features = ['startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'budget'] + list(encoded_genres_df.columns)
X = movie_data_clean[features]
y = movie_data_clean['revenue']

# Using 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Random Forest Regressor
model = make_pipeline(
    MinMaxScaler(),
    RandomForestRegressor(n_estimators=200, random_state=40)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Fit a linear regression model for Random Forest predictions
linear_reg_rf = LinearRegression()
linear_reg_rf.fit(y_test.values.reshape(-1, 1), y_pred)
slope = linear_reg_rf.coef_[0]
intercept = linear_reg_rf.intercept_

# show prediction scores for training and testing
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# plot the predictions and linear regression 
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(y_test, y_test * slope + intercept, '--', color='red')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.legend(['Revenue Prediction', 'Linear Fit'], loc="upper left")
plt.title('Random Forest: Actual vs Predicted Revenue')
plt.savefig('revenue_predictions.png')
plt.show()