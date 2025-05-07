# 🗳️ Political Speech Manipulation Detection

**Detecting Misinformation, Bias & Hostile Rhetoric in Political Communication Using NLP and Deep Learning**

## 📌 Overview

This project analyzes and flags manipulative language in political discourse—spanning speeches, tweets, and media articles. Using advanced **Natural Language Processing (NLP)** and **Transformer-based Deep Learning models**, it classifies political statements for truthfulness, detects manipulative or hostile rhetoric, analyzes media bias, and extracts key rhetorical themes.

> ✨ Real-time, multi-angle detection pipeline revealing *how* and *why* political speech may be misleading, emotionally charged, or biased.

---

## 🎯 Core Objectives

* 🔎 **Fact-Check Claims**: Classify political statements as true, false, or misleading.
* 💬 **Rhetoric Detection**: Identify manipulation tactics (e.g., emotional appeal, nationalism, repetition).
* 🧠 **Hostile Speech Analysis**: Detect personal attacks and hate speech in tweets.
* 📰 **Media Bias Framing**: Analyze partisan leanings and framing in political news.
* 🎙 **Theme Extraction**: Uncover core themes and ideological messaging in speeches.

---

## 🧪 Sample Output Snapshot

```
🎙 SPEECH 1 - MANIPULATION ANALYSIS REPORT

📊 Overall Sentiment: 😡 Negative (-0.46)
⚠️ Fact-Check: Likely False Statement Detected

🎭 Detected Tactics:
✅ Repetition | ✅ Emotional Appeal | ✅ Personal Attacks | ✅ Nationalism | ✅ Misinformation

🧠 Verdict: The speech contains manipulative or misleading content with emotionally charged, divisive language.
```

---

## 📂 Datasets Used

| Dataset            | Description                             | Use Case                        |
| ------------------ | --------------------------------------- | ------------------------------- |
| **LIAR**           | 12.8K fact-checked political statements | Fake news classification        |
| **Trump Speeches** | 45+ official transcripts                | Rhetorical + sentiment analysis |
| **Trump Tweets**   | 5.6K tweets labeled for hostility       | Hate speech / insult detection  |
| **All The News**   | 140K+ news articles (2016–2022)         | Media bias & framing analysis   |

---

## 🛠️ Techniques & Technologies

### 🧰 NLP & Preprocessing:

* Named Entity Recognition (NER)
* Tokenization, Lemmatization
* Sentiment & Emotion Analysis
* TF-IDF & Hate Speech Lexicons
* Entity Co-occurrence Mapping

### 🧠 Machine Learning Models:

* Logistic Regression, SVM, XGBoost, Random Forest
* **Topic Modeling:** LDA for ideological themes and topics

### 🤖 Deep Learning & Transformers:

* **BERT / BERT-Large**: Context-aware classification
* **DistilBERT, RoBERTa**: Light & interpretable models
* **BiLSTM**: Sequential pattern recognition
* **Hybrid Architectures** (e.g., BERT + BiLSTM)

---

## 🚀 Results & Highlights

| Task                      | Best Model         | Score / Insight                                 |
| ------------------------- | ------------------ | ----------------------------------------------- |
| ✅ Fake News Detection     | BERT-Large         | **87.3% Accuracy** - Strong context recognition |
| 😠 Tweet Insult Detection | BERT-BiLSTM Hybrid | **93.6% F1 Score** - High real-time capability  |
| 📰 Media Bias Prediction  | RoBERTa            | **78.9% Accuracy** - SHAP explainability        |
| 🎙 Theme Extraction       | DistilBERT + LDA   | Identifies themes: patriotism, authority, fear  |

---

## 🧠 Skills Demonstrated

> **Natural Language Understanding**, **Transformer Fine-Tuning**, **Text Classification**, **Sentiment Analysis**, **Rhetoric Detection**, **Topic Modeling**, **Explainable AI**, **Data Visualization**, and **Model Interpretation (SHAP)**.

---

## 📈 Future Enhancements

* Multi-language political discourse analysis 🌍
* Real-time Twitter/News feed monitoring 🔁
* Dashboard with dynamic visualizations 📊

---


