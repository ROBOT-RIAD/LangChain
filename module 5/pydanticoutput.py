from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()


# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# define the schema

class ModelEvaluation(BaseModel):
    model_name: str = Field(description="The name of the model being evaluated")
    accuracy: float = Field(
        gr = 0,
        lt = 1 ,
        description="The accuracy of the model on the evaluation dataset"
    )
    dataset_name: str = Field(description="The name of the dataset used for evaluation")

parser = PydanticOutputParser(pydantic_object=ModelEvaluation)

prompt  = PromptTemplate(
    template=
    """
    Genarate the name , accuracy and dataset of a fictional machine learning model trained for {task}
    {format_instructions}
    """,
    input_variables=["task"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)



chain = prompt | model | parser

result = chain.invoke({"task": "image classification"})

print(result.model_name)