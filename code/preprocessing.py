import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('world_tourism_economy_data.csv')

# Show first rows
print("First few rows of the dataset:")
print(df.head())

# Data cleaning
print("\nMissing values in each column:")
print(df.isnull().sum())
df.fillna(0, inplace=True)

# Remove duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Convert data types
df['tourism_receipts'] = df['tourism_receipts'].astype(float)
df['tourism_arrivals'] = df['tourism_arrivals'].astype(float)
df['gdp'] = df['gdp'].astype(float)
df['inflation'] = df['inflation'].astype(float)
df['unemployment'] = df['unemployment'].astype(float)

# Cleaned data
print("\nCleaned dataset:")
print(df.head())

# Define variables
predictor_variables = ['tourism_arrivals', 'gdp', 'inflation', 'unemployment', 'year']
target_variable = 'tourism_receipts'

print("\nPredictor Variables:")
print(predictor_variables)
print("\nTarget Variable:")
print(target_variable)

# Filter by country
aruba_data = df[df['country'] == 'Aruba']

# Group by year
yearly_tourism_receipts = df.groupby('year')['tourism_receipts'].mean().reset_index()

# Show transformed data
print("\nYearly average tourism receipts:")
print(yearly_tourism_receipts)

# Set style
sns.set(style="whitegrid")

# Plot average tourism receipts over years
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='tourism_receipts', data=yearly_tourism_receipts)
plt.title('Average Tourism Receipts Over the Years')
plt.xlabel('Year')
plt.ylabel('Tourism Receipts (in billions)')
plt.show()

# Plot Aruba tourism receipts
plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='tourism_receipts', data=aruba_data)
plt.title('Tourism Receipts for Aruba Over the Years')
plt.xlabel('Year')
plt.ylabel('Tourism Receipts (in billions)')
plt.show()

# Scatter plot: Tourism Arrivals vs GDP
plt.figure(figsize=(10, 6))
sns.scatterplot(x='gdp', y='tourism_arrivals', data=df)
plt.title('Tourism Arrivals vs. GDP')
plt.xlabel('GDP (in billions)')
plt.ylabel('Tourism Arrivals')
plt.show()

# Scatter plot: Inflation vs Unemployment
plt.figure(figsize=(10, 6))
sns.scatterplot(x='inflation', y='unemployment', data=df)
plt.title('Inflation vs. Unemployment')
plt.xlabel('Inflation')
plt.ylabel('Unemployment')
plt.show()

# Correlation heatmap
correlation_matrix = df[predictor_variables + [target_variable]].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Save cleaned data
df.to_csv('cleaned_world_tourism_economy_data.csv', index=False)

# Train-test split
from sklearn.model_selection import train_test_split
X = df[predictor_variables] 
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show train-test data
print("\nTraining Data:")
print(X_train.head())
print("\nTesting Data:")
print(X_test.head())

# Save train-test data
X_train.to_csv('training_data.csv', index=False)
X_test.to_csv('testing_data.csv', index=False)
y_train.to_csv('training_target.csv', index=False)
y_test.to_csv('testing_target.csv', index=False)