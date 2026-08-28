import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Automatically gets the directory where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Creates the correct paths dynamically on any computer
dataset_yolu = os.path.join(BASE_DIR, "dataset.csv")
model_yolu = os.path.join(BASE_DIR, "witcher_model.p")

print(f"Reading the dataset: {dataset_yolu}")

# 1. Read the dataset
try:
    df = pd.read_csv(dataset_yolu, header=None)
except FileNotFoundError:
    print(f"ERROR: '{dataset_yolu}' could not be found! First you have to collect data.")
    exit()

# 2. Separate into features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Total number of samples: {len(X)}")
print(f"Detected labels/signs: {np.unique(y)}")

# 3. Split into train and test sets
if len(X) > 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    dogruluk = model.score(X_test, y_test)
    print(f"Model Accuracy Rate: {dogruluk * 100:.2f}%")
else:
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)
    print("Warning: No test data was split because the number of samples is low.")

# 4. Save the model to the dynamic path
with open(model_yolu, "wb") as f:
    pickle.dump(model, f)

print(f"Model successfully saved -> {model_yolu}")