import pandas as pd

df = pd.read_csv("Dataset.csv")

print("Original dataset shape:", df.shape)

df = df[df["Aggregate rating"] > 0]

columns_to_remove = [
    "Restaurant ID",
    "Restaurant Name",
    "Address",
    "Locality Verbose",
    "Rating color",
    "Rating text"
]

df = df.drop(columns=columns_to_remove)

df["Cuisines"] = df["Cuisines"].fillna("Unknown")

print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Missing values:")
print(df.isnull().sum())

print("Final dataset shape:", df.shape)

df.to_csv("cleaned_dataset.csv", index=False)

print("Cleaned dataset saved successfully.")