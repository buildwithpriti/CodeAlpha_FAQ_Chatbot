
import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CodeAlpha FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

@st.cache_data
def load_faqs():
    with open("faqs.json", "r", encoding="utf-8") as file:
        return json.load(file)

faqs = load_faqs()

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)
    
    best_match_index = similarities.argmax()
    confidence = similarities[0][best_match_index]

    if confidence < 0.15:
        return "Sorry, I couldn't find a relevant answer. Please try asking another question."

    return answers[best_match_index]

st.title("🤖 CodeAlpha FAQ Chatbot")
st.write("Welcome! Ask me anything about CodeAlpha, AI, or internships.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    answer = get_answer(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
