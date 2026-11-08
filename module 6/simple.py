from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

prompt  = PromptTemplate(
    template=
    """
    your are a AI tutor .
    explain the following topic in simple terms: {topic}
    """,
    input_variables=["topic"],
)

# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


parser = StrOutputParser()


# prompt_value = prompt.invoke({"topic": "machine learning"})


# model_response = model.invoke(prompt_value)

# result = parser.invoke(model_response)

# print(result)

chain = prompt | model | parser

result = chain.invoke({"topic": "machine learning"})

print(result)

chain.get_graph().print_ascii()



