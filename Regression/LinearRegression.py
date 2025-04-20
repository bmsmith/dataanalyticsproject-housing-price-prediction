'''
Created on Mar 4, 2025

@author: Brenna Smith (n01408336)

Used this link to help create the code: https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset
file_path = r"C:\Users\Brenna\Downloads\project\cleaned_df.csv"  # Make sure this is the correct path
df = pd.read_csv(file_path)

# This is just for testing to see if the file is being loaded. It prints out the first few rows. Not important for any other reason.
print(df.head())

# List of attributes to test separately
attributes = ["Area", "LotArea", "Bedroom", "Bathroom", "PPSq"]

for attribute in attributes:
    # The process below cleans out rows with missing values for the current attribute.
    # The attribute is our independent variable, and Listed Price is our dependent variable.
    df_cleaned = df.dropna(subset=[attribute, "ListedPrice"])
    X = df_cleaned[[attribute]]
    y = df_cleaned["ListedPrice"]  

    # For machine learning, we are training our model! 80% train and 20% test is a common standard.
    # X_train, y_train -> training the model. 
    # X_test, y_test -> testing the model.  
    # test_size=0.2 -> specifies the 20% for testing.
    # random_state helps create the same train-test split. The number doesn't matter, it's only there to help if someone else uses this code.
    # Please keep the random state to prevent wonky results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=600)

    # Here we are creating and training the Linear Regression model
    # LinearRegression is a class from sklearn.linear_model. It creates the straight line: Y = mX + b
    # Y = Listed Price, X = selected attribute, m = Slope, B = intercept
    model = LinearRegression()

    # This analyzes the train data. It calculates the slope and intercept and stores the value.
    model.fit(X_train, y_train)

    # Predict on test data
    # model.predict() uses the trained model to predict the prices. 
    y_pred = model.predict(X_test)

    # Evaluate the model
    # Mean squared error measures how far the predictions are from the actual value. 
    # MSE = 1/n∑(Y actual - Y predicted)^2
    mse = mean_squared_error(y_test, y_pred)
    # Coefficient of determination
    # Measures variation in the Listed Price
    # R^2 = 1 - (sum of squared errors)/(total sum of squares)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results for each attribute
    print(f"{attribute} - Mean Squared Error (MSE): {mse:.2f}")
    print(f"{attribute} - R² Score: {r2:.2f}")

    # The graph forms here for each attribute. 
    plt.figure(f"Linear Regression - {attribute}")  # Separate pop-up for each attribute
    # The attribute is the x-axis, actual house prices are the y-axis, we made the colors blue, and a legend.
    sns.scatterplot(x=X_test[attribute], y=y_test, color="blue", label="Actual Prices")
    # The attribute is the x-axis, predicted house prices are the y-axis, we made the colors red, and a legend.
    sns.lineplot(x=X_test[attribute], y=y_pred, color="red", label="Predicted Prices")
    plt.xlabel(attribute)
    plt.ylabel("Listed Price ($)")
    plt.title(f"Linear Regression: {attribute} vs. Price (R²={r2:.2f})")
    plt.legend()

# Keep all windows open until manually closed
plt.show(block=True)  

# Calculate and print the correlation matrix
correlation_matrix = df[["Area", "LotArea", "Bedroom", "Bathroom", "PPSq"]].corr()
print("Correlation Matrix:\n", correlation_matrix)

# Visualize correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Housing Attributes")
plt.show()

