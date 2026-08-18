import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")

X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")

X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nCategorical columns:")

print("\nCity:")
print(df["City"].nunique())

print("\nLocality:")
print(df["Locality"].nunique())

print("\nCuisines:")
print(df["Cuisines"].nunique())

print("\nCurrency:")
print(df["Currency"].nunique())

print("\nTable booking:")
print(df["Has Table booking"].unique())

print("\nOnline delivery:")
print(df["Has Online delivery"].unique())

print("\nDelivering now:")
print(df["Is delivering now"].unique())

print("\nSwitch to order menu:")
print(df["Switch to order menu"].unique())