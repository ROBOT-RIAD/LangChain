
from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class JobApplications(BaseModel):
    name: str
    experience: Optional[int] = None
    email: EmailStr

    expected_salary: int = Field(
        gt=0,
        description="Expected salary must be greater than 0"
    )


new_applications = {
    "email": "abc@gmail.com",
    "experience": 3,
    "name": "John Doe",
    "expected_salary": 50000
}

application = JobApplications(**new_applications)



application_dict = application.model_dump()
