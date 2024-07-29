from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd 

df = pd.read_csv(r'D:\Internship\EnterpriseKnowledgeBased\output\sorted_data_20240718_160948_csv.csv')
questions_answers = df[['Question', 'Answer']]
# texts = list(questions_answers.itertuples(index=False, name=None))
texts = [f"{row['Question']}  {row['Answer']}" for _, row in questions_answers.iterrows()]
# Load the model
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

# Generate embeddings for the texts
embeddings = model.encode(texts, convert_to_tensor=False)

# Ensure embeddings are lists of floats
text_embeddings = [embedding.tolist() for embedding in embeddings]

from qdrant_client.models import PointStruct,VectorParams,Distance 
from qdrant_client import QdrantClient

url = 'http://localhost:6333'
collection_name  = "faq_db"

client = QdrantClient(url)
client.create_collection(
    collection_name = collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
)

points = [
    PointStruct(
        id=i,
        vector=embedding,
        payload={"text": texts[i]}
    )
    # for i, (text, embedding) in enumerate(zip(texts, embeddings))
    for i, embedding in enumerate(text_embeddings)
]

# Insert points into the collection
client.upsert(collection_name=collection_name, points=points)
print("Vector store created successfully")