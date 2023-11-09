import os
import sys

import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = "Where am I right now?"

loader = TextLoader("myFileNotes.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI()))


