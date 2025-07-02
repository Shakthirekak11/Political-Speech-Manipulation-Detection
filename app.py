import streamlit as st
from utils.text_analysis import run_full_analysis
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Political Speech Manipulation Detector", layout="wide")

# Sidebar Info
st.sidebar.title("📘 About")
st.sidebar.info("""
This app analyzes political speeches to detect:
- 📰 Fake claims
- 🚨 Misinformation
- 🔁 Repetition
- 🏛 Nationalism
- 🎯 Personal Attacks
- 📡 Media Framing
- 🔥 Emotional Appeals
""")
st.sidebar.caption("Built with ❤️ by Shakthireka")

st.title("🎙 Political Speech Manipulation Detector")

# Tab Layout
tabs = st.tabs([
    "🗣 Input Speech",
    "📊 Manipulation Insights",
    "📰 Fact-Check & Sentiment",
    "🎯 Entity & Media Framing",
    "📋 Summary Verdict",
    "ℹ️ About Project"
])

# Session state to share results across tabs
if "speech" not in st.session_state:
    st.session_state.speech = ""
if "results" not in st.session_state:
    st.session_state.results = None

# 1️⃣ Speech Input Tab
with tabs[0]:
    st.subheader("Paste the political speech below 👇")
    default_text = """My fellow Americans, the deep state is lying to you. They rigged the last election and will do it again unless we act now. The corrupt elites in Washington and the fake news media — CNN, MSNBC — are brainwashing our children with lies. This is not the America our founding fathers built.
Joe Biden and his globalist allies are opening our borders, destroying our economy, and flooding the nation with illegal criminals. They don’t care about hardworking citizens — only power, money, and control.
They tell you everything is fine. But look around: rising crime, failing schools, stolen elections, and censorship of truth. It’s a plandemic of lies — manufactured by big pharma and the shadow government to control your lives.
We will take back our country. We will stand for freedom, faith, and the American dream. Together, we will expose the frauds, defeat the traitors, and MAKE AMERICA GREAT AGAIN!

(or use this sample speech) """
    speech = st.text_area("✏️ Enter Speech", value=default_text, height=300)

    if st.button("🔍 Analyze Speech"):
        st.session_state.speech = speech
        st.session_state.results = run_full_analysis(speech)
        st.success("Analysis complete! Check all the tabs for insights.")

# Proceed if analysis is done
if st.session_state.results:

    # 2️⃣ Manipulation Insights Tab
    with tabs[1]:
        st.subheader("📊 Manipulation Techniques Detected")
        k = st.session_state.results["key_phrases"]
        m = st.session_state.results["misinformation"]
        r = st.session_state.results["rhetoric"]
        i = st.session_state.results["insult"]
        s = st.session_state.results["sentiment"]
        e = st.session_state.results["entities"]

        def get_example(lst):
            return lst[0] if lst else "None"

        tactics = {
            "🔁 Repetition": get_example(k["Repeated Phrases"]),
            "🔥 Emotional Appeal": get_example(k["Emotional Phrases"]),
            "🎯 Personal Attacks": get_example(k["Attack Phrases"]),
            "🏛 Nationalism": get_example(k["Nationalism Phrases"]),
            "🚨 Misinformation": get_example(m)
        }

        df = pd.DataFrame([
            {"Tactic": tactic, "Detected?": "✅ Yes" if phrase != "None" else "❌ No", "Example Phrase": phrase}
            for tactic, phrase in tactics.items()
        ])
        st.dataframe(df, use_container_width=True)
        st.subheader("☁️ WordCloud of Speech")
        wc = WordCloud(width=800, height=400, background_color='white').generate(st.session_state.speech)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        st.subheader("📈 Tactic Distribution Pie Chart")
        present = sum(1 for phrase in tactics.values() if phrase != "None")
        absent = len(tactics) - present
        chart_data = pd.DataFrame({'Category': ['Detected', 'Not Detected'], 'Count': [present, absent]})
        st.plotly_chart(px.pie(chart_data, names='Category', values='Count', title="Manipulation Tactics Presence"))

    # 3️⃣ Fact-Check & Sentiment
    with tabs[2]:
        st.subheader("📰 Fact-Check & Sentiment Analysis")
        verdict = "True (Factual)" if st.session_state.results["prediction"] == 1 else "False (Manipulative)"
        confidence = st.session_state.results["confidence"]
        st.metric(label="🧠 Fact-Check Result", value=verdict)
        st.progress(confidence)

        st.markdown(f"""
        **🧾 Prediction Confidence:** `{confidence:.2f}`  
        **🧠 Rhetorical Style:** `{r}`  
        **🗣 Sentiment Score (VADER):** `{s:.2f}`  
        {"✅ Neutral to Positive" if abs(s) < 0.3 else "⚠️ Emotionally Charged"}
        """)

    # 4️⃣ Entities & Media Framing
    with tabs[3]:
        st.subheader("🎯 Named Entities & Media Framing")
        st.markdown("**🏛 Named Political Entities Detected:**")
        st.write(", ".join(e) if e else "No named entities found.")

        st.markdown("**📡 Media Framing Pairs (co-occurrence patterns):**")
        frames = st.session_state.results["media_framing"]
        if frames:
            st.write("\n".join([f"🔹 {pair}" for pair in frames]))
        else:
            st.info("No strong framing patterns detected.")

    # 5️⃣ Final Verdict Tab
    with tabs[4]:
        st.subheader("📋 Final Manipulation Verdict")

        verdict_lines = []

        if st.session_state.results["prediction"] == 0:
            verdict_lines.append("❌ **Fact-check Failed**: Potentially false or manipulative.")
        if st.session_state.results["rhetoric"]:
            verdict_lines.append("🗣 **Rhetoric Detected**: Use of persuasive styles.")
        if st.session_state.results["insult"]:
            verdict_lines.append("🚨 **Insulting or Divisive Language** present.")
        if st.session_state.results["misinformation"]:
            verdict_lines.append("🚨 **Misinformation Keywords** found.")

        if not verdict_lines:
            st.success("✅ This speech appears factual and neutral.")
        else:
            for line in verdict_lines:
                st.markdown(line)
    

        st.markdown("---")
        st.markdown("🧠 *Always cross-verify political statements with credible fact-checking sources.*")
   
    # 6️⃣ About the Project
with tabs[5]:
    st.subheader(" About This Project")
    st.markdown("""
### 🗳️ *Political Speech Manipulation Detection*
**Uncovering Misinformation, Bias & Hostile Rhetoric in Political Communication**

This project analyzes political content—speeches, tweets, and media—for:
- ❌ **False or Misleading Claims**
- 🔥 **Emotional & Divisive Rhetoric**
- 🧠 **Hostility or Insults**
- 📰 **Media Framing & Bias**

> Built using advanced NLP and ML/Deep Learning models, this app reveals how political language can manipulate public opinion.

🔗 **GitHub:** [View Full Code & Docs](https://github.com/Shakthirekak11/Political-Speech-Manipulation-Detection)

---

#### 🎯 Key Features

- **Fact-Check** political statements (via LIAR dataset)
- **Detect Rhetorical Strategies** (e.g., nationalism, repetition)
- **Analyze Hostile Speech** (esp. in political tweets)
- **Visualize Media Framing** using entity co-occurrence
- **Extract Themes** and Named Entities from text

---

#### 🧠 Technologies Used

- **NLP**: spaCy, VADER, TF-IDF, NER
- **ML Models**: Logistic Regression, SVM
- **Deep Learning**: BERT, BiLSTM, RoBERTa
- **Explainability**: SHAP, keyword phrase extraction
- **Dashboard**: Streamlit with custom light theme

---

#### 📊 Sample Results (Best Models)


| Task                     | Best Model            | Score / Insight                       |
|--------------------------|-----------------------|----------------------------------------|
| ✅ Fake News Detection   | Logistic Regression   | **88.8% Accuracy**  |
| 🚨 Insult Detection      | BERT + BiLSTM         | **93.6% F1 Score**  |
| 📰 Media Bias Detection  | Co-occurrence Analysis| Based on framing patterns (heuristic)  |
| 🎙 Theme Extraction      | LDA + Sentiment       | Topics: *Witch Hunt, Fake News, Elections* |
| 💬 Deep Rhetoric Class.  | DistilBERT (Trial)    | 88.8% Accuracy  |


---

#### 🚀 Future Plans

- Support for multilingual content
- Real-time political stream analysis (tweets/news)
- Interactive dashboards with deeper visual drilldowns

---

 *This project reflects my passion for ethical NLP, fact-checking, and building explainable AI systems.*  
""")






