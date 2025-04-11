
# 🔍 Smart Contract Risk Correlation Analysis with Phi Coefficients

## 📌 Overview
This project investigates hidden relationships between binary smart contract risk features using the **Phi coefficient**, a statistical measure of association for binary variables.

By analyzing features like `is_closed_source`, `centralized_risk`, and `honeypot`, the project reveals how these traits correlate, helping developers, auditors, and analysts understand and mitigate vulnerabilities in decentralized finance (DeFi).

---

## 🧠 Key Insights & Interpretations

### 1. `is_closed_source` vs `hidden_owner` — Phi: 0.323
- Moderate correlation: Closed-source contracts are more likely to hide ownership.
- 🔎 Transparency risk.

### 2. `is_closed_source` vs `anti_whale_modifiable` — Phi: 0.281
- Weak correlation: Some use of modifiable protections in closed contracts.
- 🔎 Defensive but inconsistent.

### 3. `anti_whale_modifiable` vs `is_anti_whale` — Phi: 0.524
- Strong correlation: Flexibility is key in anti-whale features.
- 🔎 Common in manipulation-resistant tokens.

### 4. `is_honeypot` vs `buy_tax` — Phi: 0.376
- Moderate correlation: Honeypots often have embedded buy taxes.
- 🔎 Tax-related deception patterns.

### 5. `is_blacklisted` vs `centralized_risk_low` — Phi: 0.457
- Moderate correlation: Some blacklisted contracts appear decentralized.
- 🔎 Decentralization ≠ safety.

### 6. `reentrancy_without_eth_transfer` vs `encode_packed_collision` — Phi: 0.508
- Strong correlation: Shared vulnerability space.
- 🔎 Suggests developer errors or copy-paste flaws.

### 7. `centralized_risk_medium` vs `centralized_risk_high` — Phi: 0.233
- Weak correlation: Centralization must be assessed distinctly.
- 🔎 Don’t generalize tiers of centralization.

---

## ✅ Tools & Techniques

- **Language**: Python
- **Libraries**: `pandas`, `seaborn`, `matplotlib`
- **Methods**: Phi coefficient (for binary associations), heatmaps
- **Skills**: Data analysis, smart contract auditing, insight communication

---

## 📊 Visuals

![Correlation Heatmap](images/correlation_heatmap.png)

---

## 📁 Project Structure

```
project-root/
│
├── data/
│   └── smart_contract_risks.csv
├── notebooks/
│   └── phi_correlation_analysis.ipynb
├── images/
│   └── correlation_heatmap.png
├── README.md
└── requirements.txt
```

---

## 🧾 Conclusion

This project shows how **statistical methods** can improve smart contract security assessments. With insights from the Phi correlation matrix, we can:

- Identify hidden or suspicious behaviors in DeFi contracts.
- Improve audits and transparency.
- Strengthen design choices with proactive, data-backed decisions.

📬 Connect with me on [LinkedIn](https://www.linkedin.com/in/emeka-ogbonna-946828225/) or check my GitHub for more projects!
```


