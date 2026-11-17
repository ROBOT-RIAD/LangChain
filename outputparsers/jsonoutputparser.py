from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
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

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
