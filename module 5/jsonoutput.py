from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv

load_dotenv()


# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = JsonOutputParser()

prompt  = PromptTemplate(
    template=
    """
    Give me the 5 facts about {topic}.
    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)



chain = prompt | model | parser

result = chain.invoke({"topic": "quantum computing"})

print(result)