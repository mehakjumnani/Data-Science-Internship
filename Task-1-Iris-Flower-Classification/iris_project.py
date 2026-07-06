import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = pd.read_csv("iris.csv")

#Display the first 5 rows
print(iris.head())

# Display the shape of the dataset
print("\nShape of dataset:")
print(iris.shape)

# Display column names
print("\nColumn names:")
print(iris.columns)

# Display dataset information
print("\nDataset information:")
iris.info()

# Display statistical summary
print("\nStatistical Summary:")
print(iris.describe())

# Check for missing values
print("\nMissing values:")
print(iris.isnull().sum())

# Features and Target

X = iris.drop("species", axis=1)
y = iris["species"]

# Display feature values
print("\nFeatures:")
print(X.head())

# Display target values
print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Display training and testing data size
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predict species using test data

predictions = model.predict(X_test)

# Display predicted species
print("\nPredicted Species:")
print(predictions)

from sklearn.metrics import accuracy_score

# Calculate model accuracy

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display confusion matrix
print("\nConfusion Matrix:")
print(cm) 

# Visualize relationships using Pair Plot
sns.pairplot(iris, hue="species")
plt.show()

# Display histograms of all numerical features
iris.hist(figsize=(10, 8))
plt.show()