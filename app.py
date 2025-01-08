import streamlit as st
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
from spacy.cli import download





# Load NLP Models
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
        sentiment_pipeline = pipeline("sentiment-analysis")
        return nlp, sentiment_pipeline
    except OSError:
        download("en_core_web_sm")  # Download model if not available
        nlp = spacy.load("en_core_web_sm")
        sentiment_pipeline = pipeline("sentiment-analysis")
        return nlp, sentiment_pipeline


nlp, sentiment_pipeline = load_models()

# Risk Scoring Function
def calculate_risk_score(entities, sentiment):
    sentiment_weight = 1 if sentiment == "NEGATIVE" else 0.5
    return round(len(entities) * sentiment_weight, 2)

# Topic Modeling
@st.cache_data
def topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = [
        [terms[i] for i in topic.argsort()[:-10 - 1:-1]]
        for topic in lda.components_
    ]
    return topics

# Streamlit App
st.title("RiskDetect: Risk Identification and Analysis Tool")

uploaded_file = st.file_uploader("Upload a text file", type="txt")
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "MONEY", "LAW", "LOC"]]
    sentiment = sentiment_pipeline(text)[0]["label"]

    st.subheader("Extracted Entities")
    st.write(entities if entities else "No relevant entities found.")

    st.subheader("Sentiment Analysis")
    st.write(sentiment)

    risk_score = calculate_risk_score(entities, sentiment)
    st.subheader("Risk Score")
    st.write(risk_score)

    # Topic Modeling
    topics = topic_modeling([text])
    st.subheader("Identified Topics")
    for i, topic in enumerate(topics, start=1):
        st.write(f"Topic {i}: {', '.join(topic)}")

    # Visualization
    st.subheader("Risk Visualization")
    fig = px.bar(
        x=["Entities", "Sentiment"],
        y=[len(entities), risk_score],
        labels={'x': "Factors", 'y': "Impact"},
        title="Risk Factor Analysis"
    )
    st.plotly_chart(fig)
