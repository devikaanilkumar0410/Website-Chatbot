# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )

# db = Chroma(
#     persist_directory="./vector_db",
#     embedding_function=embeddings
# )

# retriever = db.as_retriever(search_kwargs={"k": 3})

# while True:
#     question = input("\nAsk: ")

#     if question.lower() == "exit":
#         break

#     docs = retriever.invoke(question)

#     print("\nRetrieved Documents:")
#     print("-" * 50)

#     for i, doc in enumerate(docs, start=1):
#         print(f"\nChunk {i}:")
#         print(doc.page_content[:500])


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma





from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")



embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

while True:
    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are Luminar Technolab's AI assistant.

Answer only from the provided context.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)

    print("\nAnswer:")
    print(response.text)