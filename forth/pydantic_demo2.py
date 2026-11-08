
from langchain_groq import ChatGroq
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv

load_dotenv()


# Define Pydantic Schema
class JobApplications(BaseModel):
    name: str
    experience: Optional[int] = None
    email: EmailStr

    expected_salary: int = Field(
        gt=0,
        description="Expected salary must be greater than 0"
    )


# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Create Structured Model
structured_model = model.with_structured_output(JobApplications)


# Invoke Model
result = structured_model.invoke(
    """
    Analyze the following job application and extract
    the relevant information.

    Job Application:

    Name: John Doe
    Email: john.doe@gmail.com
    Experience: 3 years
    Expected Salary: 50000 USD
    """
)

print(result)
print(result.model_dump())
