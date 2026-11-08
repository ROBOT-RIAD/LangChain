from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    # temperature=0,
    # max_tokens=None,
    # reasoning_format="parsed",
    # timeout=None,
    # max_retries=2,
)

st.header("chatbot")

chat_history = []

while True:
    user_input = input("user: ")

    if user_input.lower() == "exit":
        break

    chat_history.append( user_input)
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print(f"ai : {result.content}")

print(chat_history)










