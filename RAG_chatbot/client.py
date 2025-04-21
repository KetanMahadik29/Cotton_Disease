import streamlit as st 
import requests

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_response():
    try:
        response = requests.post("http://localhost:8000/api",json={"input":{"input":query}})
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

st.title("MongoDB as Vector-Store :)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]) : st.markdown(message["content"])

if query := st.chat_input("Ask your Question!!!"):
    st.session_state.messages.append({"role":"User","content":query})
    with st.chat_message("User"):st.markdown(query)
    response = get_response()
    with st.chat_message("Bot", avatar="🤖"):st.markdown(response)
    st.session_state.messages.append({"role":"Bot","content":response})

# What is the primary mechanism used in the Transformer architecture?
# How does the Transformer dispense with recurrence and convolutions?
# Can you describe the structure of the Transformer, including the encoder and decoder?