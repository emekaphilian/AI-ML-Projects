
# 🔍 Amazon Electronics Product Recommender System

A collaborative filtering-based recommendation system built using matrix factorization (SVD) to suggest electronic products to users based on their past reviews and ratings.

> 📍 Built using [Amazon Electronics Reviews Dataset](https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews) with over 1.6 million reviews.

---

## 🚀 Project Highlights

- ✅ Recommender system using **Matrix Factorization (SVD)**
- ✅ Real Amazon user review data
- ✅ Streamlit-based web interface
- ✅ Hosted live in **Google Colab** using `ngrok`
- ✅ Cold-start strategy discussion for new users
- ✅ Clean and reusable Python code

---

## 📦 Dataset Overview

| Column        | Description                                |
|---------------|--------------------------------------------|
| `reviewerID`  | Unique ID of the reviewer                  |
| `asin`        | Amazon product ID                          |
| `overall`     | Rating (1 to 5)                            |
| `reviewText`  | Full review text                           |
| `summary`     | Short review summary                       |
| `reviewTime`  | Human-readable review date                 |

---

## 📊 Model Performance

- **RMSE:** ~1.10  
- **MAE:** ~0.82  

These metrics reflect the model’s ability to predict ratings close to the actual user ratings.

---

## 💻 How It Works

- User enters their **User ID**
- Model checks all unseen products
- Predicts how likely they are to rate each product highly
- Recommends **Top 5** electronics products with predicted ratings

---

## 🧪 Run in Google Colab

👉 [Open Colab Notebook](#) *(replace with your GitHub notebook link)*

Or clone the repo and upload the notebook to [Google Colab](https://colab.research.google.com/).

---

## 📱 Web App Demo (via Streamlit + ngrok)

In the Colab notebook:

```python
# Deploy the web app in Colab
!streamlit run app.py & npx localtunnel --port 8501
```
