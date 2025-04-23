# verify_pay.ipynb (Converted Python script version for GitHub push)

import gradio as gr
import pytesseract
import cv2
import numpy as np
import re
import hashlib
from datetime import datetime
from PIL import Image

# Simulated blockchain verification database
verified_hashes = {
    hashlib.sha256("Emeka Ogbonna|Blessing Philian|10000|2024-10-15|GTBank".encode()).hexdigest(): True,
    hashlib.sha256("John Doe|Jane Smith|15000|2024-10-18|UBA".encode()).hexdigest(): True,
}

def extract_text_from_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def parse_receipt(text):
    sender = re.search(r"Sender[:\s]+([\w\s]+)", text)
    receiver = re.search(r"Receiver[:\s]+([\w\s]+)", text)
    amount = re.search(r"Amount[:\s]+₦?([\d,]+)\b", text)
    date = re.search(r"Date[:\s]+([\d\-/]+)", text)
    bank = re.search(r"(GTBank|UBA|Access Bank|Zenith Bank|Fidelity|First Bank)", text, re.I)

    return {
        "sender": sender.group(1).strip() if sender else None,
        "receiver": receiver.group(1).strip() if receiver else None,
        "amount": amount.group(1).replace(",", "") if amount else None,
        "date": date.group(1) if date else None,
        "bank": bank.group(1).title() if bank else None,
    }

def verify_transaction(details):
    if None in details.values():
        return False, "Missing required fields."
    receipt_key = f"{details['sender']}|{details['receiver']}|{details['amount']}|{details['date']}|{details['bank']}"
    hashed = hashlib.sha256(receipt_key.encode()).hexdigest()
    return verified_hashes.get(hashed, False), hashed

def generate_html_report(details, status):
    html = ["<html><head><title>Receipt Verification</title></head><body>",
            f"<h2>Receipt Verification Result - {'✅ Verified' if status else '❌ Not Verified'}</h2>",
            "<ul>",
            f"<li><b>Sender:</b> {details.get('sender')}</li>",
            f"<li><b>Receiver:</b> {details.get('receiver')}</li>",
            f"<li><b>Amount:</b> ₦{details.get('amount')}</li>",
            f"<li><b>Date:</b> {details.get('date')}</li>",
            f"<li><b>Bank:</b> {details.get('bank')}</li>",
            f"<li><b>Status:</b> {'Verified ✅' if status else 'Not Verified ❌'}</li>",
            "</ul>",
            f"<p><i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>",
            "<hr><footer>🔒 Powered by VerifyPay</footer>",
            "</body></html>"]
    return "\n".join(html)

def receipt_verification(image):
    text = extract_text_from_image(image)
    details = parse_receipt(text)
    status, _ = verify_transaction(details)
    report = generate_html_report(details, status)
    return report

interface = gr.Interface(
    fn=receipt_verification,
    inputs=gr.Image(type="pil"),
    outputs=gr.HTML(),
    title="📲 VerifyPay - Receipt Checker",
    description="Upload a bank receipt screenshot to verify its authenticity."
)

interface.launch(debug=True, share=True)
