import os
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from sentence_transformers import SentenceTransformer
# from openai import ChatCompletion
from openai import OpenAI

# Step 1: Initialize Chroma DB
class RAGRetriever:
    # def __init__(self, truncate=True,  collection_name="rag_documents", embedding_model="all-MiniLM-L6-v2", dbPath="vector_data"):
    def __init__(self, truncate=True, embedding_model="all-MiniLM-L6-v2", dbPath="vector_data"):
        """
        Initialize the ChromaDB retriever.

        :param collection_name: Name of the ChromaDB collection.
        :param embedding_model: Model for generating embeddings.
        """

        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=dbPath) 
        settings = self.client.get_settings()
        print(f"Chroma Client Persistence Directory: {settings.persist_directory}")
        

    def create_collection(self, name):
        """
        Create Collection.
        :param name: Name of the ChromaDB collection.
        """
        self.collection = self.client.get_or_create_collection(name=name)

    def truncate_collection(self, collection_name: str) -> bool:
        """
        Truncates (removes all documents from) a ChromaDB collection while preserving the collection itself.
        
        Args:
            collection_name (str): Name of the collection to truncate
            client (Optional[ChromaClient]): Existing ChromaDB client instance. If None, creates a new client.
            
        Returns:
            bool: True if truncation was successful, False if collection doesn't exist
            
        Raises:
            InvalidCollectionException: If the collection doesn't exist
            Exception: If there's an error during truncation
        """
        try:
            # Get the collection - this will raise InvalidCollectionException if it doesn't exist
            collection = self.client.get_collection(collection_name)
                
            # Get all document IDs in the collection
            all_ids = collection.get()["ids"]
            
            # If collection is not empty, delete all documents
            if all_ids:
                collection.delete(ids=all_ids)
                print(f"Successfully removed {len(all_ids)} documents from collection '{collection_name}'")
            else:
                print(f"Collection '{collection_name}' is already empty")
            
            
        except InvalidCollectionException as e:
            print(f"Collection error: {str(e)}")
            raise  # Re-raise the InvalidCollectionException
        except Exception as e:
            print(f"Error truncating collection: {str(e)}")
            raise

    def add_documents(self, docs, ids=None):
        """
        Add documents to Chroma.

        :param docs: List of documents (strings).
        :param ids: Optional list of unique IDs for the documents.
        """
        existing_ids = self.collection.get()["ids"]
        collection_count = len(existing_ids) if existing_ids else 0
        if ids is None:
            ids = [str(collection_count + i) for i in range(len(docs))]
        embeddings = self.embedding_model.encode(docs, convert_to_numpy=True).tolist()
        self.collection.add(documents=docs, ids=ids, embeddings=embeddings)
        # print(f"Documents added to the collection: {self.collection.name}")



    def retrieve(self, query, k=5):
        """
        Retrieve top-k documents relevant to the query.

        :param query: Query string.
        :param k: Number of results to retrieve.
        :return: List of top-k relevant documents.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        print (len(results))
        return results["documents"][0]  # Flatten the list of documents


# Step 2: Initialize RAG Workflow
class RAGSystem:
    def __init__(self, retriever, openai_api_key):
        """
        Initialize the RAG system.

        :param retriever: Instance of RAGRetriever.
        :param openai_api_key: OpenAI API key for generation.
        """
        self.retriever = retriever
        self.api_key = openai_api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.client = OpenAI(api_key=self.api_key)
        #  os.environ["TOKENIZERS_PARALLELISM"] 

    def generate_response(self, query, k=2, max_tokens=300):
        """
        Generate a response using retrieved documents and a language model.

        :param query: Input query string.
        :param k: Number of documents to retrieve.
        :param max_tokens: Maximum tokens for the generated response.
        :return: Generated response string.
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query, k)
        print(len(relevant_docs[0]))
        context = "\n".join(relevant_docs)
        # print (context)
        # Generate response using OpenAI
        prompt = f"Context:\n{context}\n\nQuery:\n{query}\n\nAnswer:"
        # print (prompt)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        # response = self.client.chat.completions.create(
        #     # api_key=self.api_key,
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=max_tokens,
        #     temperature=0.7
        # )
        return response.choices[0].message.content
        # return response['choices'][0]['message']['content']



