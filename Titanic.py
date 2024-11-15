import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
titanic_data = pd.read_csv('Titanic.csv')
print(titanic_data.head())
print(titanic_data.info())
print(titanic_data.describe())
# Data cleaning
# Handle missing values
titanic_data.fillna(method='ffill', inplace=True) 
# Convert categorical features to numerical
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Exploratory Data Analysis (EDA)
# Visualize the relationship between variables
sns.pairplot(titanic_data, vars=['Age', 'Fare', 'Pclass', 'Survived'], hue='Survived')
plt.show()
# Analyze the distribution of features
sns.histplot(titanic_data['Age'])
plt.show()
sns.boxplot(x='Pclass', y='Age', data=titanic_data)
plt.show()
# Analyze the relationship between features
sns.heatmap(titanic_data.corr(), annot=True)
plt.show()
# Identify patterns and trends
# Example: Analyze the survival rate by gender
survived_by_gender = titanic_data.groupby('Sex')['Survived'].mean()
print(survived_by_gender)
# Visualize the survival rate by gender
sns.barplot(x=survived_by_gender.index, y=survived_by_gender.values)
plt.show()
