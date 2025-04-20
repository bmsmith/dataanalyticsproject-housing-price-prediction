'''
Created on Mar 4, 2025

@author: Brenna Smith (n01408336)


Used this link to help create the code:
https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

    #For machine learning, we are training our model! 80% train and 20% test is a common standard.
    # X_train, y_train -> training the model. 
    # X_test, y_test -> testing the model.  
    # test_size=0.2 -> specifies the 20% for testing.
    # random_state helps create the same train-test split. The number doesn't matter, it's only there to help if someone else uses this code.
    # Please keep the random state to prevent wonky results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Here we are training the Decision Tree Regression
    tree_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
    tree_model.fit(X_train, y_train)

    # Predict on test data
    # model.predict() uses the trained model to predict the prices. 
    y_pred = tree_model.predict(X_test)

    # Evaluate the model
    # Mean squared error measures how far the predictions are from the actual value. 
    # MSE = 1/n∑(Y actual - Y predicted)^2
    mse = mean_squared_error(y_test, y_pred)
    #Coefficient of determination
    # Measures variation in the Listed Price
    # R^2 = 1 - (sum of squared errors)/(total sum of squares)
    r2 = r2_score(y_test, y_pred)
    print(f"{attribute} - Decision Tree Regression MSE: {mse:.2f}, R²: {r2:.2f}")

    # Plot (Separate Pop-up for Each Attribute)
    plt.figure(f"Decision Tree Regression - {attribute}")
    sns.scatterplot(x=X_test[attribute], y=y_test, color="blue", label="Actual")
    sns.scatterplot(x=X_test[attribute], y=y_pred, color="orange", label="Predicted")
    plt.xlabel(attribute)
    plt.ylabel("Listed Price (log-transformed)")
    plt.title(f"Decision Tree Regression: {attribute} vs Price")
    plt.legend()
    
# Keep windows open until user closes them
plt.show(block=True)  