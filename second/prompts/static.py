from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    # temperature=0,
    # max_tokens=None,
    # reasoning_format="parsed",
    # timeout=None,
    # max_retries=2,
)

st.header("Research Tool")
user_input = st.text_area("Enter your research question here:")

if st.button("summarize"):
    result = llm.invoke(user_input)

    st.write(result.content)






