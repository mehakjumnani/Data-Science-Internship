# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Unemployment in India.csv")

# Display first 5 rows
print(df.head())

#Dataset information
print("\nDataset information:")
print(df.info())

#Stastistical Summary
print("\nStatistical Summary:")
print(df.describe())

#Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

#Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#Removing missing values
df = df.dropna()

#Remove Duplicate rows
df = df.drop_duplicates()

print("\nAfter Cleaning:")
print(df.info())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

# Average unemployment rate by state
state_unemployment = df.groupby("Region")[" Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)

print("\nAverage Unemployment Rate by State:")
print(state_unemployment)

# Bar chart of average unemployment rate by state

plt.figure(figsize=(14,8))

state_unemployment.plot(kind='bar', color='skyblue')

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()

# Convert Date column to datetime format
df[" Date"] = pd.to_datetime(df[" Date"], dayfirst=True)

# Sort data by date
df = df.sort_values(" Date")

# Monthly average unemployment rate
monthly_unemployment = df.groupby(" Date")[" Estimated Unemployment Rate (%)"].mean()

# Line chart
plt.figure(figsize=(14,6))

plt.plot(monthly_unemployment.index,
         monthly_unemployment.values,
         marker='o')

plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# Histogram of Unemployment Rate

plt.figure(figsize=(8,5))

sns.histplot(df[" Estimated Unemployment Rate (%)"], bins=20, kde=True)

plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")

plt.show()

# Box Plot of Unemployment Rate

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df[" Estimated Unemployment Rate (%)"]
)

plt.title("Box Plot of Unemployment Rate")

plt.xlabel("Unemployment Rate (%)")

plt.show()
