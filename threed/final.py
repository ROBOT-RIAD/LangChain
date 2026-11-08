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

# create the chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for meg in st.session_state.chat_history:
    role = "assistant" if isinstance(meg, AIMessage) else "user"

    with st.chat_message(role):
        st.write(meg.content)

with st.sidebar:
    st.subheader("Chat History")
    if not st.session_state.chat_history:
        st.write("No chat history yet.")
    else:
        for msg in st.session_state.chat_history:
            role = "AI" if isinstance(msg, AIMessage) else "Human"
            st.write(f"{role}: {msg.content}")


user_input = st.chat_input("type your message here...")

if user_input:
    if user_input.strip().lower() == "exit":
        st.stop()

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    result = model.invoke(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=result.content))
    with st.chat_message("assistant"):
        st.write(result.content)

        














