from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import sys

movie_data = pd.read_csv(sys.argv[1])

if len(sys.argv[1]) == 0:
    exit(1)

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
print(f'Training Score: {model.score(X_train, y_train)}')
print(f'Test Score: {model.score(X_test, y_test)}')

# plot the predictions and linear regression 
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(y_test, y_test * slope + intercept, '--', color='red')
plt.xlabel('Actual Revenue (Millions $)')
plt.ylabel('Predicted Revenue (Millions $)')
plt.legend(['Revenue Prediction', 'Linear Fit'], loc="upper left")
plt.title('Random Forest: Actual vs Predicted Revenue')

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
importances = model.named_steps['randomforestregressor'].feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# Filter out the features with zero importance
non_zero_indices = indices[importances[indices] > 0]
non_zero_importances = importances[non_zero_indices]
non_zero_feature_names = feature_names[non_zero_indices]

importance_threshold = 0.01
high_importance_indices = non_zero_indices[non_zero_importances > importance_threshold]
high_importance_values = non_zero_importances[non_zero_importances > importance_threshold]
high_importance_feature_names = non_zero_feature_names[non_zero_importances > importance_threshold]
high_importance_feature_names = high_importance_feature_names.str.replace(r'genres_.*', 'genres', regex=True)

plt.subplot(1, 2, 2)
plt.title('Feature Importances')
plt.bar(range(len(high_importance_values)), high_importance_values, align='center')
plt.xticks(range(len(high_importance_values)), high_importance_feature_names, rotation=90)
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('high_importance_and_rev_prediction.png')
plt.show()