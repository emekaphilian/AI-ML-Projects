import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV, cross_validate
from tqdm import tqdm
import joblib
import random

# =============================
# STEP 1: Load Dataset
# =============================

# Set local path to dataset
dataset_path = "data/Electronics_5.json"

# Load the JSON dataset into a pandas DataFrame
df = pd.read_json(dataset_path, lines=True)

# =============================
# STEP 2: Data Exploration
# =============================

# Plot distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['overall'], bins=5, kde=False, color='purple')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Most reviewed products
top_products = df.groupby('asin')['reviewerID'].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.index, y=top_products.values, palette='Blues_d')
plt.title('Most Reviewed Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

# Most active reviewers
top_reviewers = df['reviewerID'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_reviewers.index, y=top_reviewers.values, palette='Reds_d')
plt.title('Most Active Reviewers')
plt.xlabel('User ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

# =============================
# STEP 3: Clean and Prepare Data
# =============================

df_relevant = df[['reviewerID', 'asin', 'overall']].copy()
df_relevant.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'overall': 'rating'}, inplace=True)
df_relevant.dropna(inplace=True)
df_relevant['user_id'] = pd.factorize(df_relevant['user_id'])[0]
df_relevant['item_id'] = pd.factorize(df_relevant['item_id'])[0]

# Save cleaned dataset
os.makedirs("output", exist_ok=True)
df_relevant.to_csv("output/cleaned_amazon_reviews.csv", index=False)

# =============================
# STEP 4: Train SVD Model
# =============================

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd = SVD()
svd.fit(trainset)
predictions = svd.test(testset)

accuracy.rmse(predictions)
accuracy.mae(predictions)

# =============================
# STEP 5: Hyperparameter Tuning
# =============================

param_grid = {
    'n_factors': [50, 100, 150],
    'reg_all': [0.1, 0.2, 0.3],
    'lr_all': [0.005, 0.01, 0.02]
}
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)

print("Best Parameters:", grid_search.best_params)
print("Best RMSE:", grid_search.best_score['rmse'])

# Cross-validation with best parameters
best_params = grid_search.best_params['rmse']
svd = SVD(**best_params)
results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# =============================
# STEP 6: Product Recommendation Function
# =============================

def recommend_product(df, input_review, input_rating, user_id, svd_model, item_map=None, top_n=5, limit=500):
    df_filtered = df[['reviewerID', 'asin', 'overall']]
    all_item_ids = df_filtered['asin'].unique()[:limit]
    recommended_items = []

    print("⏳ Predicting ratings...")
    for item_id in tqdm(all_item_ids):
        if item_id not in df_filtered[df_filtered['reviewerID'] == user_id]['asin'].values:
            try:
                pred = svd_model.predict(user_id, item_id)
                recommended_items.append((item_id, pred.est))
            except:
                continue

    recommended_items.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommended_items[:top_n]

    print(f"\n🎯 Top {top_n} Recommended Products:")
    for item, rating in top_recommendations:
        name = item_map[item] if item_map and item in item_map else item
        print(f"🔹 {name} — Predicted Rating: {rating:.2f}")

# =============================
# STEP 7: Interactive Recommender
# =============================

def recommend_from_input(df, svd_model, item_map=None, top_n=5, limit=500):
    user_id = input("👤 Your User ID: ")
    input_review = input("💬 Your Review (optional): ")

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

# =============================
# STEP 8: Create Item Map & Recommend
# =============================

item_map = df.drop_duplicates(subset="asin")[["asin", "summary"]].set_index("asin")["summary"].to_dict()

# Interactive recommendation
# recommend_from_input(df, svd_model=svd, item_map=item_map)

# =============================
# STEP 9: Save Model
# =============================

model_path = "output/svd_model.pkl"
joblib.dump(svd, model_path)
print(f"✅ Model saved to {model_path}")
