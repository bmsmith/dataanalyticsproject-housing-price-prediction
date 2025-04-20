'''
Created on Mar 4, 2025

@author: Brenna Smith (n01408336)

Used this link to help create the code:
https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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
    df_cleaned = df.dropna(subset=[attribute, "ListedPrice"]).copy()

    # Log-transform price to reduce variability
    df_cleaned["ListedPrice"] = np.log(df_cleaned["ListedPrice"])

    X = df_cleaned[[attribute]]
    y = df_cleaned["ListedPrice"]

    # For machine learning, we are training our model! 80% train and 20% test is a common standard.
    # X_train, y_train -> training the model. 
    # X_test, y_test -> testing the model.  
    # test_size=0.2 -> specifies the 20% for testing.
    # random_state helps create the same train-test split. The number doesn't matter, it's only there to help if someone else uses this code.
    # Please keep the random state to prevent wonky results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial Transformation (Degree 3)
    poly = PolynomialFeatures(degree=3)

    #Convert transformed features into a DataFrame with column names
    X_train_poly = pd.DataFrame(poly.fit_transform(X_train), columns=poly.get_feature_names_out([attribute]), index=X_train.index)
    X_test_poly = pd.DataFrame(poly.transform(X_test), columns=poly.get_feature_names_out([attribute]), index=X_test.index)

    # Train Polynomial Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # Predict on test data
    # model.predict() uses the trained model to predict the prices. 
    y_pred = poly_model.predict(X_test_poly)
    #Coefficient of determination
    # Measures variation in the Listed Price
    # R^2 = 1 - (sum of squared errors)/(total sum of squares)
    # Evaluate the model
    # Mean squared error measures how far the predictions are from the actual value. 
    # MSE = 1/n∑(Y actual - Y predicted)^2
    mse = mean_squared_error(y_test, y_pred)
    
    r2 = r2_score(y_test, y_pred)
    print(f"{attribute} - Polynomial Regression MSE: {mse:.2f}, R²: {r2:.2f}")

    #Creating curved graph
    X_range = np.linspace(X[attribute].min(), X[attribute].max(), 100).reshape(-1, 1)
    X_range_df = pd.DataFrame(X_range, columns=[attribute]) 
    X_range_poly = pd.DataFrame(poly.transform(X_range_df), columns=poly.get_feature_names_out([attribute]))

    y_poly_curve = poly_model.predict(X_range_poly)

    # Plot (Separate Pop-up for Each Attribute)
    plt.figure(f"Polynomial Regression - {attribute}")
    sns.scatterplot(x=X_test[attribute], y=y_test, color="green", label="Actual")
    sns.lineplot(x=X_range.flatten(), y=y_poly_curve, color="purple", label="Polynomial Fit")
    plt.xlabel(attribute)
    plt.ylabel("Listed Price (log-transformed)")
    plt.title(f"Polynomial Regression: {attribute} vs Price (Degree 3)")
    plt.legend()

# Keep windows open until user closes them
plt.show(block=True)  

# Check correlations between all numerical features
correlation_matrix = df[["Area", "LotArea", "Bedroom", "Bathroom", "PPSq"]].corr()

# Print the correlation matrix
print(correlation_matrix)
print("Correlation Matrix of Housing Attributes:\n", correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Housing Attributes")
plt.show()


