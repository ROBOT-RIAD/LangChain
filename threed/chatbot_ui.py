from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    # temperature=0,
    # max_tokens=None,
    # reasoning_format="parsed",
    # timeout=None,
    # max_retries=2,
)

st.header("Chat Bot")


user_input = st.chat_input("type your message here...")

if user_input:
    if user_input.strip().lower() == "exit":
        st.stop()

    with st.chat_message("user"):
        st.write(user_input)

    result = model.invoke(user_input)

    with st.chat_message("assistant"):
        st.write(result.content)










