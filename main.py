import os
from dotenv import load_dotenv

# load .env file
load_dotenv()

openai_api_key = os.getenv("openai_api_key")
os.environ["OPENAI_API_KEY"] = openai_api_key

from llama_index import VectordbIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()

index = VectordbIndex.from_documents(documents)

# query
response = index.query("What is the enneagram?")
print(response)
