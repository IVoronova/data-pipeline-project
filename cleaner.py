import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Survival count bar chart
data['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Titanic Survival Count')
plt.xticks([0,1], ["Died", "Survived"], rotation=0)
plt.ylabel('Number of Passengers')
plt.savefig('titanic_survival_chart.png')
print("\nSurvival count bar chart saved as 'titanic_survival_chart.png'!")
plt.show()

# Survival % pie chart
data['Survived'].value_counts().sort_index().plot(kind='pie', labels=['Died', 'Survived'], colors=['red', 'green'], autopct="%1.1f%%")
plt.title('Titanic Survival Percentage')
plt.ylabel('')
plt.savefig('titanic_survival_pie_chart.png')
print("\nSurvival percentage pie chart saved as 'titanic_survival_pie_chart.png'!")
plt.show()