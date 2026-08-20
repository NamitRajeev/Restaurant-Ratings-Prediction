import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("cleaned_dataset.csv")

df = df.drop("Switch to order menu", axis=1)

X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

categorical_columns = [
    "Country Code",
    "City",
    "Locality",
    "Cuisines",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now"
]

numerical_columns = [
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Price range",
    "Votes"
]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded = encoder.fit_transform(
    X_train[categorical_columns]
)

X_test_encoded = encoder.transform(
    X_test[categorical_columns]
)

X_train_numerical = X_train[numerical_columns].values
X_test_numerical = X_test[numerical_columns].values

X_train_final = np.hstack(
    (X_train_encoded, X_train_numerical)
)

X_test_final = np.hstack(
    (X_test_encoded, X_test_numerical)
)

model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Decision Tree Performance:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

#better than linear regression model because some relationships are non-linear and decision trees can capture those relationships better.