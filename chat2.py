from load_text import *
from rag import *
import os, sys



# from dotenv import load_dotenv

# load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

dbpath = "vectorDB"
collection_name="NPR"
# create_db(dbpath, "NPR")
truncate = False
if len(sys.argv) > 1:
    argument = sys.argv[1]
    print("First argument:", argument)
    if argument.lower() == "true": 
      truncate = True
    else: 
      truncate = False

retriever = RAGRetriever(dbPath=dbpath)
retriever.create_collection(collection_name)
if truncate :
  retriever.truncate_collection(collection_name)


documents = get_text("txt")
for doc in documents:
  retriever.add_documents(doc)
  print(f"{len(doc)} Documents added to the collection: {collection_name} in {dbpath}")


rag_system = RAGSystem(retriever, openai_api_key=os.environ['OPENAI_API_KEY'])
# Generate a response to a query
query = "who runs Planet Money?"
response = rag_system.generate_response(query)
print(f"Response:\n{response}")
