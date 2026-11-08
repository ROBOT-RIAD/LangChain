from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

prompt1  = PromptTemplate(
    template=
    """
    Evaluate the following students answers.
    question: {question}
    students answers: {answers}

    provide a detailed evaluation including:
    - correcteness
    - strengths
    - weaknesses
    - suggestions for improvement
    """,
    input_variables=["question", "answers"],
)



prompt2  = PromptTemplate(
    template=
    """
    Convert the following detailed evaluation into a concise 5-points feedback:
    {evaluation}
    """,
    input_variables=["evaluation"],
)



# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({
    "question" : "what is machine learning?",
    "answers" : """
    Machine learning is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.
    """
})

print(result)





