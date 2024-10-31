import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI

# Load environment variables
if not load_dotenv():
    raise EnvironmentError("Could not find .env file")

# Get API key
openai_api_key = os.getenv("openai_api_key")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in .env file")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Check if data directory exists
data_path = Path('data')
if not data_path.exists():
    raise FileNotFoundError("Data directory not found")

try:
    # Initialize LLM
    llm = OpenAI(model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    
    # Load documents
    documents = SimpleDirectoryReader('data').load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    
    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the enneagram?")
    print(response)
  
    print("---" * 20)

    response2 = query_engine.query("What are the archetypes? please breifly explain every typology")
    print(response2)
  
  


except Exception as e:
    print(f"An error occurred: {str(e)}")
