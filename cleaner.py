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

# Clean the dataset
data["Age"] = data["Age"].fillna(data["Age"].median())
data = data.drop(columns=["Cabin"])
data = data.dropna()

print("\n--- Dataset Stats --- ")
print("Average Age:", np.round(data['Age'].mean(), 1))
print("Average Fare:", np.round(data['Fare'].mean(), 2))
print("Gender Distribution:", np.round(data['Sex'].value_counts(normalize=True).loc['male']*100, 1), "% male and", np.round(data['Sex'].value_counts(normalize=True).loc['female']*100, 1), "% female")
print("Survival Rate:", np.round(data['Survived'].mean() * 100, 1), "%")
print("Survival Rate of Women:", np.round(data['Survived'][data['Sex'] == "female"].mean() * 100, 1), "%")
print("Survival Rate of Men:", np.round(data['Survived'][data['Sex'] == "male"].mean() * 100, 1), "%")  
print("Oldest Passenger Age:", data['Age'].max())
print("Youngest Passenger Age:", data['Age'].min())
print("Average Age of Survivors:", np.round(data['Age'][data['Survived'] == 1].mean(), 1))
print("Average Age of Non-Survivors:", np.round(data['Age'][data['Survived'] == 0].mean(), 1))


# Survival count bar chart
data['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Titanic Survival Count')
plt.xticks([0,1], ["Died", "Survived"], rotation=0)
plt.ylabel('Number of Passengers')
plt.savefig('titanic_survival_chart.png')
print("\nSurvival count bar chart saved as 'titanic_survival_chart.png'!")


# Survival % pie chart
data['Survived'].value_counts().sort_index().plot(kind='pie', labels=['Died', 'Survived'], colors=['red', 'green'], autopct="%1.1f%%")
plt.title('Titanic Survival Percentage')
plt.ylabel('')
plt.savefig('titanic_survival_pie_chart.png')
print("\nSurvival percentage pie chart saved as 'titanic_survival_pie_chart.png'!")
