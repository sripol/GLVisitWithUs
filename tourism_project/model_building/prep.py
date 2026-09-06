import pandas as pd
from sklearn.model_selection import train_test_split
import os 
from imblearn.over_sampling import RandomOverSampler

DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns if present
df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop(columns=["CustomerID"], inplace=True)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


ros = RandomOverSampler(
    sampling_strategy="auto",
    random_state=42
)

X_train_balanced, y_train_balanced = ros.fit_resample(x_train, y_train)

print("Before balancing:")
print(y_train.value_counts())

print("\nAfter balancing:")
print(y_train_balanced.value_counts())

print("\nBalanced percentages:")
print(y_train_balanced.value_counts(normalize=True) * 100)

X_train_balanced.to_csv("tourism_project/model_building/x_train.csv", index=False)
y_train_balanced.to_csv("tourism_project/model_building/y_train.csv", index=False)
x_test.to_csv("tourism_project/model_building/x_test.csv", index=False)
y_test.to_csv("tourism_project/model_building/y_test.csv", index=False)
print("Data preparation done")
