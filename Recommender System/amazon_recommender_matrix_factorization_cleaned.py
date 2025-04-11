
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load dataset
# Make sure to place the CSV file in the correct directory before running
df = pd.read_csv("ratings_Electronics.csv", names=["userId", "productId", "rating", "timestamp"])

# Drop duplicates
df.drop_duplicates(["userId", "productId"], inplace=True)

# Drop timestamp since it's not needed
df.drop("timestamp", axis=1, inplace=True)

# Filter users with more than 50 ratings
filtered_users = df.groupby("userId").filter(lambda x: len(x) >= 50)

# Create user-item matrix
user_item_matrix = filtered_users.pivot(index='userId', columns='productId', values='rating').fillna(0)

# Convert to numpy array for SVD
R = user_item_matrix.to_numpy()
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Apply Singular Value Decomposition (Matrix Factorization)
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# Reconstruct the original matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Function to recommend products
def recommend_products(user_id, num_recommendations=10):
    if user_id not in predicted_ratings_df.index:
        return "User not found."
    
    user_row = predicted_ratings_df.loc[user_id]
    user_original_data = user_item_matrix.loc[user_id]
    
    # Filter out products already rated
    unrated_products = user_original_data[user_original_data == 0]
    recommended = user_row[unrated_products.index].sort_values(ascending=False).head(num_recommendations)
    
    return recommended

# Example usage
sample_user = predicted_ratings_df.index[0]  # Replace with any userId from the dataset
recommendations = recommend_products(sample_user)
print(f"Top recommendations for User {sample_user}:
", recommendations)

# Optional: Evaluate the model (RMSE)
def calculate_rmse(pred_matrix, actual_matrix):
    pred_flattened = pred_matrix[actual_matrix.nonzero()].flatten()
    actual_flattened = actual_matrix[actual_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(pred_flattened, actual_flattened))

rmse = calculate_rmse(predicted_ratings, R)
print("RMSE of the model:", rmse)
