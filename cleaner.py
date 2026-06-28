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

print("\n--- Dataset Stats --- ")
print("Average Age:", np.round(data['Age'].mean(), 1))
print("Survival Rate:", np.round(data['Survived'].mean() * 100, 1), "%")
print("Oldest Passenger Age:", data['Age'].max())
print("Youngest Passenger Age:", data['Age'].min())