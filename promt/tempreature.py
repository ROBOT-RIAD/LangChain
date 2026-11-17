from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    # max_tokens=None,
    # reasoning_format="parsed",
    # timeout=None,
    # max_retries=2,
)
st.header("Research tool")

user_intput = st.text_input("inter your prompt")
if st.button("Summarize"):
    result = model.invoke(user_intput)
    st.write(result.content)












