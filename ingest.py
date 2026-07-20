# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# loader = TextLoader(
#     "data/luminar.txt",
#     encoding="utf-8"
# )

# documents = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_documents(documents)

# print("Chunks created:", len(chunks))

from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader(
    "data/luminar.txt",
    encoding="utf-8"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./vector_db"
)

print("Vector DB Created")