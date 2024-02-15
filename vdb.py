import openai
import tiktoken
import os
import re
import requests
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from IPython.display import Markdown
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import os
import tiktoken
import openai
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
import pickle
import pandas as pd
import numpy as np
import openai
import tiktoken
import pincone
import os
import re
import requests
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from IPython.display import Markdown

load_dotenv()

os.environ["OPENAI_API_KEY"]  ="your key here"
os.environ["YOUR-PINECONE-KEY"] = "your key here"
os.environ["YOUR-PINECONE-ENV"] ="your key here"
