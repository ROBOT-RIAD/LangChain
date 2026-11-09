from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


documents = [
    "Bangladesh is a country in South Asia. Its capital is Dhaka. The official language is Bangla. Bangladesh became independent in 1971.",

    "The Sundarbans is the largest mangrove forest in the world. It is located in Bangladesh and India. The Royal Bengal Tiger is one of its famous animals.",

    "The Padma Bridge is an important infrastructure project in Bangladesh. It connects the southwestern region with the rest of the country and improves transportation.",

    "Cox's Bazar is a popular tourist destination in Bangladesh. It is famous for having one of the longest natural sandy sea beaches in the world.",

    "Bangladesh has a rich cultural heritage. Pohela Boishakh, Eid, and Durga Puja are widely celebrated. Rice and fish are popular traditional foods."
]

query  = "tell me bangladesh which resoine exist?"

doc_embeddings = embeddings.embed_documents(documents)

query_embeddings = embeddings.embed_query(query)

score = cosine_similarity([query_embeddings],doc_embeddings)[0]

index,score =sorted(list(enumerate(score)),key=lambda x:x[1])[-1]

print(documents[index])
print("semilarity score in score: ",score)


