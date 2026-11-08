from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)

load_dotenv()


# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# define the schema

response_schema = [
    ResponseSchema(
        name="fact_1", 
        description="the first fact about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="the second fact about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="the third fact about the topic"
    ),
    ResponseSchema(
        name="fact_4",
        description="the fourth fact about the topic"
    ),
    ResponseSchema(
        name="fact_5",
        description="the fifth fact about the topic"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schema)

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