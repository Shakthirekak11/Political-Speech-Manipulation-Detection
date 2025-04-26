# 🗳️ Political Speech Manipulation Detection  
**Detecting Misinformation, Bias, and Hostile Rhetoric in Political Communication using NLP and Machine Learning**

---

## 📌 Overview

This project focuses on detecting and analyzing manipulative language in political content—ranging from speeches and tweets to news articles. By leveraging Natural Language Processing (NLP), machine learning, and state-of-the-art transformer models, it classifies political statements, identifies rhetorical strategies, flags hostile language, and uncovers media bias.

---

## Core Objectives

-  **Classify** political statements as *true*, *false*, or *misleading*  
-  **Detect** hostile/manipulative language in tweets  
-  **Analyze** media framing and bias in political news articles  
-  **Extract** rhetorical themes from political speeches

---

##  Datasets Used

| Dataset           | Description                                         | Use Case                                  |
|------------------|-----------------------------------------------------|-------------------------------------------|
| LIAR             | 12.8K fact-checked statements                       | Fake news classification                  |
| Trump Speeches   | 45+ official speeches                               | Rhetorical and sentiment analysis         |
| Trump Tweets     | 5.6K insult-labeled tweets                          | Hostile language detection                |
| All The News     | 140K+ news articles from major US outlets          | Media bias and framing analysis           |

---

##  Techniques & Models

- **NLP Pipelines:** Tokenization, Lemmatization, NER, Sentiment Analysis  
- **Feature Engineering:** TF-IDF vectors, hate speech lexicons, entity co-occurrence  
- **Topic Modeling:** LDA for speeches and media themes  
- **Classification Models:** Logistic Regression, SVM, XGBoost, Random Forest  
- **Deep Learning & Transformers:** BiLSTM, BERT, DistilBERT, RoBERTa

---

##  Results (Highlights)

| Task                    | Best Model           | Accuracy / F1 Score | Notes                                      |
|-------------------------|----------------------|----------------------|--------------------------------------------|
| Fake News Detection     | BERT-Large           | **87.3% Accuracy**   | High contextual understanding              |
| Tweet Insult Detection  | BERT-BiLSTM Hybrid   | **93.6% F1 Score**   | Real-time capable                          |
| Media Bias Prediction   | RoBERTa              | **78.9% Accuracy**   | Strong interpretability with SHAP          |
| Speech Theme Extraction | DistilBERT + LDA     | —                    | Identifies patriotism, law, audience tone  |



