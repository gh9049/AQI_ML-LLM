import os
import sys

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Check if a query is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <query>")
    sys.exit(1)

# Extract the query from command-line arguments
query = sys.argv[1]

# Create a CSVLoader with the specified CSV file
loader = CSVLoader('indexes.csv')
index = VectorstoreIndexCreator().from_loaders([loader])

# Create a ChatOpenAI instance
chat_model = ChatOpenAI()

# Query the index with the provided query using ChatOpenAI
response = index.query(query, llm=chat_model)

# Print the response
print(response)
