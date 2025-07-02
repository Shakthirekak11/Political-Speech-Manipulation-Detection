
import re
from collections import Counter
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from scipy.sparse import load_npz
import joblib
import streamlit as st

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# 🔹 Emotional & Persuasive Keywords
emotional_words = {
    "fight", "great", "betrayal", "victory", "win", "weak", "strong", "disaster", 
    "hope", "success", "fear", "danger", "crime", "threat", "hero", "enemy", 
    "collapse", "freedom", "power", "unstoppable", "failure", "evil", "ruined"
}

# 🔹 Attack & Insult Keywords
attack_words = {
    "crooked", "fake news", "corrupt", "liar", "traitor", "disgrace", 
    "dishonest", "loser", "pathetic", "fraud", "criminal", "weak", "scam", 
    "cheater", "rigged", "illegal", "incompetent", "traitorous"
}

# 🔹 Nationalism & Patriotism Keywords
nationalism_words = {
    "America", "our country", "patriot", "nation", "citizens", "USA", 
    "freedom", "liberty", "American dream", "homeland", "constitution", 
    "founding fathers", "sovereignty", "national security", "border", 
    "flag", "loyalty", "true American", "make America great"
}

# 🔹 Misinformation & Conspiracy Keywords (Expanded)
misinformation_words = {
    "hoax", "deep state", "globalist", "elites", "stolen", "rigged", 
    "cover-up", "censorship", "hidden truth", "fake science", "brainwashing", 
    "mass manipulation", "plandemic", "big pharma", "government control", 
    "shadow government", "voter fraud", "stolen election", "illegals voting",
    "corrupt system", "fake ballots"
}




def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Tokenization & Lemmatization
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)



# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores['compound']  # Compound score represents overall sentiment


def extract_named_entities(text):
    doc = nlp(text)
    # Only accept named entities with proper labels and exclude all-uppercase words (often false positives)
    entities = [
        ent.text 
        for ent in doc.ents 
        if ent.label_ in ["PERSON", "ORG", "GPE"] and not ent.text.isupper()
    ]
    return list(set(entities))




# Load Pre-trained Models
liar_model = joblib.load("models/liar_logistic_regression_optimized.pkl")
trump_speech_model = joblib.load("models/trump_speeches_logistic_regression.pkl")
insult_model = joblib.load("models/trump_speeches_svm.pkl")

# Load TF-IDF Vectorizer
with open("Feature_engineered/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)


# Function to transform input speech using TF-IDF
def transform_text_tfidf(text):
    return tfidf_vectorizer.transform([text])


# Function to predict truthfulness
def predict_fact_checking(tfidf_vector):
    prediction = liar_model.predict(tfidf_vector)[0]  # Predict class
    proba = liar_model.predict_proba(tfidf_vector)[0]  # Get probability scores
    return prediction, proba


fact_check_results = {
    0: "False (Manipulative)",
    1: "True (Factual)"
}


# Function to detect rhetorical style
def predict_rhetorical_style(tfidf_vector):
    prediction = trump_speech_model.predict(tfidf_vector)[0]
    return prediction


# Function to detect insults
def predict_insult(tfidf_vector):
    prediction = insult_model.predict(tfidf_vector)[0]
    return prediction


def extract_key_phrases(speech):
    """
    Extract key persuasive, attack, nationalism, and misinformation phrases from speech.
    """
    sentences = speech.split(". ")
    
    repeated_phrases = []
    emotional_phrases = []
    attack_phrases = []
    nationalism_phrases = []
    misinformation_phrases = []
    
    # 🔹 Find most repeated words (excluding stopwords)
    words = re.findall(r'\b\w+\b', speech.lower())
    word_counts = Counter(words)
    repeated_words = {word for word, count in word_counts.items() if count > 2}  # Words repeated 3+ times
    
    for sentence in sentences:
        words_in_sentence = set(sentence.lower().split())

        # 🔹 Repetition Detection (if contains a highly repeated word)
        if repeated_words & words_in_sentence:
            repeated_phrases.append(sentence)

        # 🔹 Emotional Appeal Detection
        if emotional_words & words_in_sentence:
            emotional_phrases.append(sentence)

        # 🔹 Personal Attacks Detection
        if attack_words & words_in_sentence:
            attack_phrases.append(sentence)

        # 🔹 Nationalism Detection
        if nationalism_words & words_in_sentence:
            nationalism_phrases.append(sentence)

        # 🔹 Misinformation & Conspiracy Detection
        if misinformation_words & words_in_sentence:
            misinformation_phrases.append(sentence)

    return {
        "Repeated Phrases": repeated_phrases,
        "Emotional Phrases": emotional_phrases,
        "Attack Phrases": attack_phrases,
        "Nationalism Phrases": nationalism_phrases,
        "Misinformation Phrases": misinformation_phrases
    }





def extract_misinformation_phrases(speech):
    sentences = speech.split(". ")
    misinformation_phrases = []
    
    for sentence in sentences:
        words_in_sentence = set(sentence.lower().split())
        if misinformation_words & words_in_sentence:
            misinformation_phrases.append(sentence)
    
    return misinformation_phrases

def extract_attack_phrases(speech):
    sentences = speech.split(". ")
    attack_phrases = []
    
    for sentence in sentences:
        words_in_sentence = set(sentence.lower().split())
        if attack_words & words_in_sentence:
            attack_phrases.append(sentence)
    
    return attack_phrases

def check_media_framing(speech, media_co_occurrence):
    speech_words = set(speech.lower().split())  
    matched_phrases = []
    high_freq_pairs = media_co_occurrence[media_co_occurrence["Frequency"] > 100]

    for _, row in high_freq_pairs.iterrows():
        entity_pair = row["Entity Pair"]
        entity1, entity2 = entity_pair.split(" & ")

        if entity1.lower() == entity2.lower():
            continue

        if entity1.lower() in speech_words and entity2.lower() in speech_words:
            matched_phrases.append(f"{entity1} ↔ {entity2}")

    return matched_phrases


@st.cache_data(show_spinner="Loading data...")
def load_media_co_occurrence():
    url = "https://huggingface.co/datasets/shakthireka/political-manipulation-co-occurence-matrix/resolve/main/co_occurrence_patterns.csv"
    return pd.read_csv(url)


def run_full_analysis(speech):
    media_co_occurrence = load_media_co_occurrence()  
    preprocessed = preprocess_text(speech)
    tfidf = transform_text_tfidf(preprocessed)

    pred, proba = predict_fact_checking(tfidf)
    rhetoric = predict_rhetorical_style(tfidf)
    insult = predict_insult(tfidf)
    sentiment = analyze_sentiment(speech)
    entities = extract_named_entities(speech)
    key_phrases = extract_key_phrases(speech)
    misinformation = extract_misinformation_phrases(speech)
    framing = check_media_framing(speech, media_co_occurrence)

    return {
        "prediction": pred,
        "confidence": max(proba),
        "rhetoric": rhetoric,
        "insult": insult,
        "sentiment": sentiment,
        "entities": entities,
        "key_phrases": key_phrases,
        "misinformation": misinformation,
        "media_framing": framing
    }


if __name__ == "__main__":
    sample_speech = """My fellow Americans, our great nation is under siege. The radical left, the corrupt elites, and the globalists are 
working together to DESTROY everything we hold dear. They flood our borders with criminals, they take away your freedoms, 
and they LIE to you every single day! 

They say we should trust the mainstream media—FAKE NEWS! They tell you the economy is strong while hardworking Americans 
struggle to pay their bills. They censor the truth, cover up their crimes, and protect the swamp creatures who want 
to control your lives. But we will NOT let them!.......(or use this sample speech)"""
    results = run_full_analysis(sample_speech)
    print("Verdict:", "Manipulative" if results["prediction"] == 0 else "Factual")
    print("Sentiment Score:", results["sentiment"])
    print("Entities Detected:", results["entities"])



