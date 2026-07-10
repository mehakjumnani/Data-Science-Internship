import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("car data.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display statistical summary of numerical columns
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove Duplicate Rows
df = df.drop_duplicates()

# Verify the dataset after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning")
print(df.duplicated().sum())

# Plot the distribution of Selling Price
plt.figure(figsize=(8,5))

sns.histplot(df["Selling_Price"], bins=20, kde=True, color="skyblue")

plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Number of Cars")

plt.show()

# Scatter plot: Present Price vs Selling Price
plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Present_Price",
    y="Selling_Price",
    data=df,
    color="blue"
)

plt.title("Present Prioice vs Selling Price")
plt.xlabel("Present Price(Lakhs)")
plt.ylabel("Selling Price(Lakhs)")

plt.show()

# Plot the distribution of Fuel Types
plt.figure(figsize=(6,5))

sns.countplot(
    x="Fuel_Type",
    data=df,
    hue="Fuel_Type",
    palette="Set2",
    legend=False
)
plt.title("Distribution of Fuel Types")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")

plt.show()

# Plot the distribution of Transmission Types
plt.figure(figsize=(6,5))

sns.countplot(
 x="Transmission",
 data=df,
 hue="Transmission",
 palette="pastel",
 legend=False
)
plt.title("Distribution of Transmission Types")
plt.xlabel("Transmission")
plt.ylabel("Number of Cars")

plt.show()

#Create a copy of the dataset for correaltion analysis
df_corr = df.copy()

# Convert categorical coulmns into numerical values
df_corr["Fuel_Type"] = df_corr["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df_corr["Selling_type"] = df_corr["Selling_type"].map({
    "Dealer": 0,
    "Individual": 1
})

df_corr["Transmission"] = df_corr["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})
# Plot the correlation heatmap
plt.figure(figsize=(10,8))

sns.heatmap(
    df_corr.drop("Car_Name", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# Create a copy of the dataset for Machine Learning
df_ml = df.copy()

# Convert categorical columns into numerical values
df_ml["Fuel_Type"] = df_ml["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df_ml["Selling_type"] = df_ml["Selling_type"].map({
    "Dealer": 0,
    "Individual": 1
})

df_ml["Transmission"] = df_ml["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# Features (Independent Variables)
X = df_ml.drop(["Car_Name", "Selling_Price"], axis=1)

# Target Variable
y = df_ml["Selling_Price"]

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

# Display the shape of training and testing data
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predict the selling prices using the test data
y_pred = model.predict(X_test)

# Display the first 10 predicted values
print("Predicted Selling Prices:")
print(y_pred[:10])

# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Plot Actual vs Predicted Selling Prices
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, color="green")

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")

plt.show()

