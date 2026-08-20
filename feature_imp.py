import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

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

X_categorical = encoder.fit_transform(
    X_train[categorical_columns]
)

X_numeric = X_train[numerical_columns].values

X_train = np.hstack([
    X_categorical,
    X_numeric
])

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

importance_values = model.feature_importances_

importance = {}

position = 0

for column in categorical_columns:
    number_of_categories = len(
        encoder.categories_[categorical_columns.index(column)]
    )

    importance[column] = sum(
        importance_values[
            position:position + number_of_categories
        ]
    )

    position += number_of_categories

for column in numerical_columns:
    importance[column] = importance_values[position]
    position += 1

importance = pd.Series(importance)
importance = importance.sort_values(ascending=False)

print("Feature Importance")
print(importance)

plt.figure(figsize=(10, 7))

plt.barh(
    importance.index[::-1],
    importance.values[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Most Important Features for Restaurant Rating Prediction")
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)

plt.show()