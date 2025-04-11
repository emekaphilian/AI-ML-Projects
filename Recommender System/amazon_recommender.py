!pip install numpy==1.23.5





!pip install --upgrade scikit-surprise


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import os
import io

from surprise import Dataset, Reader, SVD

from surprise.model_selection import train_test_split
from surprise import accuracy

# Mounting my Drive on Colab
from google.colab import drive
drive.mount('/content/drive')


#Importing Libraries

from google.colab import files

# Upload kaggle.json manually
files.upload()


!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # Set correct permissions


# Define Google Drive directory
drive_path = "/content/drive/MyDrive/Projects/Amazon_Reviews"

# Create directory if it doesn't exist
import os
os.makedirs(drive_path, exist_ok=True)

# Download dataset directly to Google Drive
!kaggle datasets download -d shivamparab/amazon-electronics-reviews -p "{drive_path}" --unzip

print(f"Dataset saved to: {drive_path}")


import pandas as pd

# Path to your dataset in Google Drive
drive_path = '/content/drive/MyDrive/Projects/Amazon_Reviews/Electronics_5.json'

# Load the JSON dataset into a pandas DataFrame, specifying lines=True
df = pd.read_json(drive_path, lines=True) # Added lines=True to handle line-delimited JSON

# Display the first few rows of the dataset
df.head()

# 1. Distribution of Ratings (Improved with Histogram)
plt.figure(figsize=(10,6))
sns.histplot(df['overall'], bins=5, kde=False, color='purple')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()




# 2. Most Reviewed Products (With product names)
top_products = df.groupby('asin')['reviewerID'].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_products.index, y=top_products.values, palette='Blues_d')
plt.title('Most Reviewed Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()





# 3. Most Active Reviewers

top_reviewers = df['reviewerID'].value_counts().head(10)  # Get top 10 most active reviewers
plt.figure(figsize=(10,6))
sns.barplot(x=top_reviewers.index, y=top_reviewers.values, palette='Reds_d')
plt.title('Most Active Reviewers')
plt.xlabel('User ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()



import pandas as pd

# Display first 5 rows
print(df.head())

# Display column names to check available features
print(df.columns)

# Display dataset information
print(f"Dataset shape: {df.shape}")
print(f"Number of unique users: {df['reviewerID'].nunique()}")
print(f"Number of unique products: {df['asin'].nunique()}")
print(f"Range of ratings: {df['overall'].min()} to {df['overall'].max()}")

# Create a copy of the relevant columns to avoid modifying the original DataFrame
df_relevant = df[['reviewerID', 'asin', 'overall']].copy()

# Rename columns for clarity
df_relevant.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'overall': 'rating'}, inplace=True)

# Check for missing values
print("Missing values:\n", df_relevant.isnull().sum())

# Drop any rows with missing values if necessary
df_relevant.dropna(inplace=True)

# Convert user IDs and product IDs to numeric values (Surprise works best with numeric IDs)
df_relevant['user_id'] = pd.factorize(df_relevant['user_id'])[0]
df_relevant['item_id'] = pd.factorize(df_relevant['item_id'])[0]

# Display the cleaned dataset
print("Processed Data Sample:")
print(df_relevant.head())

# Save the cleaned dataset
df_relevant.to_csv("/content/drive/MyDrive/Projects/Amazon_Reviews/cleaned_amazon_reviews.csv", index=False)

print("✅ Cleaned dataset saved!")


# Load the cleaned dataset (CSV file) into a DataFrame
cleaned_dataset_path = "/content/drive/MyDrive/Projects/Amazon_Reviews/cleaned_amazon_reviews.csv"
df_relevant = pd.read_csv(cleaned_dataset_path)

# Display the first few rows of the cleaned dataset
print("Cleaned Data Sample:")
print(df_relevant.head())


# Initialize the Reader object
# ⚙️ Define the format for Surprise library
reader = Reader(rating_scale=(1, 5))  # Ratings range from 1 to 5

# Prepare the data in the format Surprise expects (user_id, item_id, rating)
# 🧹 Load and prepare data for modeling
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)

# Split the data into a training set and a testing set
# 🧪 Split dataset into training and testing
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize the SVD model
svd = SVD()

# Train the model on the training set
svd.fit(trainset)

# Make predictions on the test set
predictions = svd.test(testset)

print("✅ Model trained and predictions made!")

# Evaluate using RMSE and MAE
from surprise import accuracy

# RMSE - Root Mean Squared Error
rmse = # 🧮 Evaluate model performance (RMSE)
accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# MAE - Mean Absolute Error
mae = accuracy.mae(predictions)
print(f"MAE: {mae}")


from surprise.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_factors': [50, 100, 150],
    'reg_all': [0.1, 0.2, 0.3],
    'lr_all': [0.005, 0.01, 0.02]
}

# Perform grid search
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)

# Get the best parameters and the corresponding RMSE
print("Best Parameters: ", grid_search.best_params)
print("Best RMSE: ", grid_search.best_score)


from surprise import SVD
from surprise.model_selection import cross_validate

# Best parameters obtained from the grid search
best_params = {'n_factors': 50, 'reg_all': 0.3, 'lr_all': 0.01}

# Initialize the SVD model with the best parameters
svd = SVD(n_factors=best_params['n_factors'],
          reg_all=best_params['reg_all'],
          lr_all=best_params['lr_all'])

# Perform cross-validation (let's use 5-fold cross-validation)
results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Print the cross-validation results
print("Cross-validation results:")
print(f"RMSE: {results['test_rmse']}")
print(f"MAE: {results['test_mae']}")
print(f"Mean RMSE: {results['test_rmse'].mean()}")
print(f"Mean MAE: {results['test_mae'].mean()}")


from tqdm import tqdm

def recommend_product(df, input_review, input_rating, user_id, svd_model, item_map=None, top_n=5, limit=500):
    """
    Recommends products using collaborative filtering (SVD).

    :param df: DataFrame with 'reviewerID', 'asin', 'overall'
    :param input_review: User review (for future content-based filtering)
    :param input_rating: Rating by user (1-5)
    :param user_id: ID of the user
    :param svd_model: Trained SVD model
    :param item_map: Optional dict to map asin to product titles
    :param top_n: Number of recommendations to return
    :param limit: Max number of items to test (for speed)
    """
    df_filtered = df[['reviewerID', 'asin', 'overall']]
    all_item_ids = df_filtered['asin'].unique()[:limit]  # ⚡ limit for testing

    recommended_items = []

    print("⏳ Predicting ratings...")

    for item_id in tqdm(all_item_ids):
        if item_id not in df_filtered[df_filtered['reviewerID'] == user_id]['asin'].values:
            try:
                pred = svd_model.predict(user_id, item_id)
                recommended_items.append((item_id, pred.est))
            except:
                continue  # Skip unknown user/item

    recommended_items.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommended_items[:top_n]

    print(f"\n🎯 Top {top_n} Recommended Products:")
    for item, rating in top_recommendations:
        name = item_map[item] if item_map and item in item_map else item
        print(f"🔹 {name} — Predicted Rating: {rating:.2f}")


input_review = "This GPS is great and easy to use"
input_rating = 5
user_id = "A1H8PY3QHMQQA0"

recommend_product(df, input_review, input_rating, user_id, svd_model=svd)


# ================================
# STEP 1: Define the function
# ================================
from tqdm import tqdm

def recommend_from_input(df, svd_model, item_map=None, top_n=5, limit=500):
    """
    Interactive recommender function that prompts for input and displays top N product recommendations.
    """
    print("📝 Enter your review details below:\n")

    user_id = input("👤 Your User ID: ")
    input_review = input("💬 Your Review: (optional) ")

    while True:
        try:
            input_rating = float(input("⭐ Your Rating (1-5): "))
            if 1 <= input_rating <= 5:
                break
            else:
                print("❗ Rating must be between 1 and 5.")
        except ValueError:
            print("❗ Please enter a valid number.")

    print("\n🔍 Fetching personalized recommendations...")

    df_filtered = df[['reviewerID', 'asin', 'overall']]
    all_item_ids = df_filtered['asin'].unique()[:limit]

    recommended_items = []

    for item_id in tqdm(all_item_ids):
        if item_id not in df_filtered[df_filtered['reviewerID'] == user_id]['asin'].values:
            try:
                pred = svd_model.predict(user_id, item_id)
                recommended_items.append((item_id, pred.est))
            except:
                continue

    recommended_items.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommended_items[:top_n]

    print(f"\n✅ Top {top_n} Recommended Electronics for You:\n")
    for i, (item, rating) in enumerate(top_recommendations, 1):
        name = item_map.get(item, item) if item_map else item
        print(f"{i}. {name} — Predicted Rating: {rating:.2f}")


# Use the first 'summary' for each 'asin' as its "title"
item_map = df.drop_duplicates(subset="asin")[["asin", "summary"]].set_index("asin")["summary"].to_dict()


recommend_from_input(df, svd_model=svd, item_map=item_map)


import random
import pandas as pd
from surprise import SVD, Dataset, Reader

def recommend_from_input(df, svd_model, item_map):
    """
    Recommends products based on the input review and rating using collaborative filtering.

    :param df: A pandas DataFrame containing user_id, item_id, and rating columns
    :param svd_model: The pre-trained SVD model
    :param item_map: A dictionary mapping item ids (asin) to product names
    :return: Recommended products based on collaborative filtering
    """

    # Step 1: User input for review and rating
    input_review = input("Enter your review: ")  # Get review from the user
    input_rating = int(input("Enter your rating (1-5): "))  # Get rating from the user

    if input_rating < 1 or input_rating > 5:
        print("Invalid rating. Please enter a rating between 1 and 5.")
        return

    print(f"Review: {input_review}")
    print(f"Rating: {input_rating}")

    # Step 2: Automatically select a random user ID (to simulate a user)
    user_id = random.choice(df['reviewerID'].unique())
    print(f"User ID: {user_id} (Auto-selected for recommendation)")

    # Step 3: Filter the dataset to avoid recommending the same product
    df_filtered = df[['reviewerID', 'asin', 'overall']]

    # Step 4: Create a list of all unique product ids
    all_item_ids = df_filtered['asin'].unique()

    # Step 5: Create a list to hold recommended products
    recommended_items = []

    # Step 6: Loop through all items and predict ratings for the user
    for item_id in all_item_ids:
        # Skip items that the user has already rated
        if item_id not in df_filtered[df_filtered['reviewerID'] == user_id]['asin'].values:
            try:
                # Predict rating for the current item
                predicted_rating = svd_model.predict(user_id, item_id)
                recommended_items.append((item_id, predicted_rating.est))
            except Exception as e:
                print(f"Error predicting for {item_id}: {e}")
                continue  # Skip if prediction fails

    # Step 7: Sort the recommended items by predicted rating (highest first)
    recommended_items.sort(key=lambda x: x[1], reverse=True)

    # Step 8: Get top 5 recommendations
    top_recommendations = recommended_items[:5]

    # Step 9: Output the top 5 recommended products
    print(f"\nTop Recommended Products for User {user_id} (based on collaborative filtering):")
    for item, rating in top_recommendations:
        product_name = item_map.get(item, 'Unknown Product')  # Map to product name
        print(f"Product ID: {item}, Product Name: {product_name}, Predicted Rating: {rating:.2f}")


# Example usage:
# Assuming you have your SVD model trained and item_map ready
# Example dataframe (df) and item_map are required before calling the function.
# df = ... # Your pandas DataFrame with user reviews
# svd = ... # Your trained SVD model
# item_map = ... # Your dictionary mapping item IDs to product names

# Call the function (make sure svd_model and item_map are defined)
recommend_from_input(df, svd_model=svd, item_map=item_map)


import joblib

# Save the trained model
joblib.dump(svd, '/content/drive/MyDrive/Projects/Amazon_Reviews/svd_model.pkl')
