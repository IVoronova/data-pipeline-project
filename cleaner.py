import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('titanic.csv')

# Take first look
print("Shape of the dataset:", data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nMissing values in each column:")
print(data.isnull().sum())