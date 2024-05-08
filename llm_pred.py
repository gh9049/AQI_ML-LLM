import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import openai

openai.api_key = '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'

response = openai.Completion.create(
  engine="gpt-3.5-turbo-instruct",
  prompt="calculate the R², MSE, K-Fold Cross-Validation Mean Score",
  temperature=0.7,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(response.choices[0].text.strip())
# Load your dataset
df = pd.read_csv('indexes.csv')

# Assuming the dataset is loaded into a DataFrame named df
X = df[['si', 'ni', 'rpi', 'spi', 'pmi']]  # Feature columns
y = df['AQI']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a polynomial regression model. Adjust the degree as necessary.
degree = 2  # Example degree
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

polyreg.fit(X_train, y_train)

# Predict on testing set
y_pred = polyreg.predict(X_test)

# Calculate R² and MSE
r2 = polyreg.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

print(f"1.R² : {r2}")
print(f"2.MSE: {mse}")


# Define the K-Fold Cross-Validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Perform K-Fold CV and calculate the mean score
cv_scores = cross_val_score(polyreg, X, y, cv=kf)

print(f"3.K-Fold Cross-Validation Mean Score: {np.mean(cv_scores)}")