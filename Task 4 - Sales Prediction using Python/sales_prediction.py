# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Advertising.csv")

# Display first 5 rows
print(df.head()) 

# Display dataset information
print(df.info())

#Display Statistical Summary
print(df.describe())

# Check Missing Values
print("Missing Values")
print(df.isnull().sum())

# Check duplicate rows
print("Duplicate Rows")
print(df.duplicated().sum())

# Remove the unneccesary column
df = df.drop("Unnamed: 0" , axis=1)
 
 # Display first 5 rows
print(df.head())

# Distribution of Sales
plt.figure(figsize=(8,5))

sns.histplot(df["Sales"], bins=20, kde=True, color="skyblue")

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()

# TV Advertising vs Sales
plt.figure(figsize=(8,6))

sns.scatterplot(
    x="TV",
    y="Sales",
    data=df,
    color="blue"
)

plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")

plt.show()

# Radio Avertising vs Sales
plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Radio",
    y="Sales",
    data=df,
    color="green"
)
plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertisisng Budget")
plt.ylabel("Sales")

plt.show()

# Newspaper Advertising vs Sales
plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Newspaper",
    y="Sales",
    data=df,
    color="red"
)
plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Budget")
plt.ylabel("Sales")

plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")

plt.show()

# Features
X = df.drop("Sales", axis=1)

# Target Variable
y = df["Sales"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# Import train-test split
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

# Import Linear Regression
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Predict Sales
y_pred = model.predict(X_test)

print("Predicted Sales:")
print(y_pred[:10])

# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Plot Actual vs Predicted Sales
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, color="purple")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.show()