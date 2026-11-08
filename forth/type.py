from langchain_groq import ChatGroq
from typing import TypedDict , Annotated ,Literal,Optional


from dotenv import load_dotenv
load_dotenv()



model = ChatGroq(
    model="openai/gpt-oss-120b",
)

# define the schema

class ResumeAnalysis(TypedDict):
    name: Annotated[str, "The full name of the candidate"]
    skills: Annotated[list[str], "A list of the candidate's skills"]
    summary: Annotated[str, "A brief summary of the candidate's experience and background"]
    experience: Annotated[Literal["Software Engineer", "Data Scientist", "Product Manager", "Other"], "The candidate's current or most recent job title"]
    strength: Annotated[Optional[list[str]], "A list of the candidate's strengths, if available"]
    weakness: Annotated[Optional[list[str]], "A list of the candidate's weaknesses, if available"]

    


#create stractured output

structured_model = model.with_structured_output(ResumeAnalysis)

#invoke the model

result = structured_model.invoke(
    """
    Analyze the following resume and extract the relevant information:
    John Doe
    john.doe@example.com
    (123) 456-7890
    Python, JavaScript, SQL
    Software Engineer at XYZ Corp
    Bachelor's in Computer Science
    Summary: Experienced software engineer with a strong background in Python and web development.
    """
)

print(result)