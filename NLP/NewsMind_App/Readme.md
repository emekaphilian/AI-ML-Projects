# NewsMind - News Summarizer, Headline Predictor & Sentiment Analyzer

![NewsMind Banner](https://via.placeholder.com/1200x300.png?text=NewsMind+%7C+AI-Powered+News+Insights)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-ff69b4)](https://www.gradio.app/)
[![HuggingFace Models](https://img.shields.io/badge/Models-HuggingFace-blue)](https://huggingface.co/models)

---

## Project Overview
**NewsMind** is a Natural Language Processing (NLP) application designed to:
- 📝 Summarize lengthy news articles.
- 📰 Generate accurate, human-like headlines.
- 🎭 Analyze and detect sentiment from articles (Positive, Negative, Neutral).

Built using **open-source** transformer models and deployed with **Gradio**, NewsMind brings AI-powered journalism to your fingertips.

---

## Project Scope
- Accepts **URLs** or **raw text** of news articles.
- Extracts and cleans content using **newspaper3k**.
- Uses **facebook/bart-large-cnn** for summarization.
- Uses **t5-small** for headline generation.
- Analyzes sentiment using **TextBlob**.
- Deployed via Gradio UI and Google Colab.

---

## Project Structure
```plaintext
📦 newsmind/
 ┣ 📄 newsmind_colab.ipynb        # Google Colab version
 ┣ 📄 README.md                   # Project documentation
 ┣ 📁 assets/                     # Images, banners, and icons
 ┣ 📄 requirements.txt           # Required libraries
 ┗ 📄 app.py (optional)          # For Hugging Face Space deployment
```

---

##  Setup and Usage
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/newsmind.git
cd newsmind
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab
Open [`newsmind_colab.ipynb`](./newsmind_colab.ipynb) and run all cells.

### 4. Launch Locally (Optional)
```bash
python app.py
```

---

## Deployment (Hugging Face Spaces)
1. Upload `app.py`, `requirements.txt`, and `README.md` to your new Space.
2. Set runtime as **Gradio**.
3. Deploy and share your link!

---

## Technologies Used
- [Transformers (HuggingFace)](https://huggingface.co/models)
- [Gradio](https://gradio.app)
- [Newspaper3k](https://github.com/codelucas/newspaper)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [Google Colab](https://colab.research.google.com)

---

##  Credits
Created with ❤️ by Emeka Ogbonna

---

## 🛡 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📷 Banner
Use a [custom tool](https://www.canva.com/) or this placeholder:
![NewsMind Banner](https://via.placeholder.com/1200x300.png?text=NewsMind+%7C+AI-Powered+News+Insights)


