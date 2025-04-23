![Verfy_pay_banner](https://github.com/user-attachments/assets/e4426ca1-0ba0-4ce9-87a9-f6cc29bf3632)



### ✅ `README.md`

```markdown
# 📲 VerifyPay - Mobile-Ready Receipt Verifier

**Protect your business from fake payment receipts**  
A smart, mobile-friendly receipt verification system built with OCR, simulated blockchain verification, and optional LLM classification. Built with Gradio and ready for deployment.

---

## 📖 Project Overview

VerifyPay is an AI-powered mobile-friendly app designed to **help businesses and individuals detect fake bank payment receipts**. The tool leverages OCR (Optical Character Recognition) to extract data from uploaded receipts, verifies them using a simulated blockchain database, and optionally uses an LLM (Language Model) to classify the legitimacy of the transaction.

This MVP was built in Google Colab and designed for rapid testing and deployment. It also provides users with a downloadable HTML verification report.

---

## 🎯 Scope

- 🧾 Receipt Scanning: Extract key details like Sender, Receiver, Amount, Date, and Bank from a scanned receipt.
- 🔐 Simulated Blockchain Verification: Cross-check extracted info with a mock hashed database.
- 🤖 Optional LLM Classification: Use a language model to simulate human-like classification of the receipt's authenticity.
- 📤 HTML Report Generation: Provide a downloadable and shareable verification report.
- 📱 Gradio UI: Fully interactive and mobile-optimized interface for ease of use.

---

## 🛠️ Features

- ✅ Upload image receipts (e.g., screenshots, PDFs converted to image)
- 🧠 Intelligent text extraction with Tesseract OCR
- 🔍 Regex-based data parsing
- 🔐 Blockchain-style verification with SHA-256 hashing
- 🧾 HTML report with timestamp and parsed details
- 🧠 Optional LLM integration (OpenAI/Cohere-ready)
- 🌐 Deployable using Gradio with mobile compatibility

---

## 📁 Project Structure

```
VerifyPay/
│
├── verify_pay.ipynb             # Main notebook with the entire implementation
├── sample_receipts/             # Folder for sample receipt images (for testing)
├── html_reports/                # Optional folder to save downloaded HTML reports
├── README.md                    # You're reading this
└── requirements.txt             # List of dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/emekaogbonna/verifypay.git
cd verifypay
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> For local environments, make sure Tesseract is installed:
```bash
sudo apt-get install tesseract-ocr
```

### 3. Run the App in Colab or Jupyter

Run the notebook `verify_pay.ipynb`, scroll to the bottom cell, and click the Gradio interface button.

---

## 📸 Sample Screenshots

![image](https://github.com/user-attachments/assets/8cdd7ff6-86e5-40c8-b24e-6fb356e62cf4)

---

## 🔗 Technologies Used

- Python
- Gradio
- OpenCV
- Tesseract OCR
- Regex (re)
- hashlib
- OpenAI / Cohere API (optional LLM classification)
- HTML (report rendering)

---

## 👨‍💻 Author

**Emeka Ogbonna**  
📍 Lagos, Nigeria  
📧 ogbonnaemeka665@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/emeka-ogbonna-946828225/)  
📱 +234 816 517 6993

---

## 📄 License

MIT License © 2025 Emeka Ogbonna
```

---

### ✅ `requirements.txt`

```txt
gradio
pytesseract
opencv-python
numpy
Pillow
```

If you're planning to deploy with optional LLM support (like OpenAI or Cohere), you can also add:

```txt
openai
cohere
```


