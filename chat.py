import os
import glob
from dotenv import load_dotenv
import gradio as gr
# imports for langchain

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


import whisper, torch


MODEL = "gpt-4o-mini"
db_name = "vector_db"
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

folders = glob.glob("txt")

# With thanks to CG and Jon R, students on the course, for this fix needed for some users 
text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

documents = []

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)
# print (documents[0].metadata)
text_splitter = CharacterTextSplitter(separator=". ", length_function=len, chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)
print(len(chunks))
# print(chunks[0])
embeddings = OpenAIEmbeddings()
# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()


vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")
print (f"Sample Embedding [100] {sample_embedding[100]}")


# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.5, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()
print (retriever)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "who is Abdel Latif"
result = conversation_chain.invoke({"question":query})
print(result["answer"])
# set up a new conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)



def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = whisper.load_model("tiny.en").to(device)
# result = model.transcribe("https://chrt.fm/track/138C95/prfx.byspotify.com/e/play.podtrac.com/npr-344098539/traffic.megaphone.fm/NPR2458523115.mp3?d=2941&size=47058905&e=1220910109&t=podcast&p=344098539")


# with open("mp3s/transcription.txt", "w") as f:   # Opens file and casts as f 
#     f.write(result["text"] + f.name)       # Writing
#     # File closed automatically
# # print(result["text"])