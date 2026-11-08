from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Initialize Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


parser = StrOutputParser()


prompt1  = PromptTemplate(
    template=
    """
    Summarize the following text into in 3 sort lines: \n {text}
    """,
    input_variables=["text"],
)



prompt2  = PromptTemplate(
    template=
    """
    write 3 simple questions from the following text : \n {text}
    """,
    input_variables=["text"],
)


text = """
what is deep learning? Deep learning is a subset of machine learning that uses neural networks with many layers (hence "deep") to analyze various factors of data. It is particularly effective for tasks such as image and speech recognition, natural language processing, and autonomous driving. Deep learning models can automatically learn features from raw data, making them highly versatile and powerful for complex tasks.
"""

parallel_chain = RunnableParallel(
    {
        "summarization": prompt1 | model | parser,
        "question_generation": prompt2 | model | parser
    }
)


parallel_result = parallel_chain.invoke({
    "text": text})

print("Summarization Result:", parallel_result["summarization"])
print("Question Generation Result:", parallel_result["question_generation"])



