from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client.models import PointStruct,VectorParams,Distance 
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json
import numpy as np


llm = CTransformers(  
    model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',  
    model_type="llama",
    config= { 'max_new_tokens': 512,'temperature':0.8 }
)  

print("LLM initialised")
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)


url = 'http://localhost:6333'

client = QdrantClient(url = url , prefer_grpc = False)
db = Qdrant(client = client, embeddings = model, collection_name = "city_bank_faq_data")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {query}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

prompt = PromptTemplate(template= prompt_template, input_variables = ['context','query'])



retriever = db.as_retriever(search_kwargs={"k":1})

# def create_retrieval_chain(search_function,k):
#     return RetrievalQA.from_chain_type(
#         llm = llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={"k": k, "search_function": search_function}),
#         chain_type_kwargs={"prompt": prompt}, 
#         # prompt_template=prompt_template,        
#         verbose=True    )
# # Similar records retrieval
# similar_retriever_chain = create_retrieval_chain(lambda x:x, k=1)


# # Dissimilar records retrieval
# def dissimilar_search(query_vector, collection):
#     # Find the farthest point by inverting the distance metric
#     distances = collection.get_distance_matrix(query_vector)
#     return np.argsort(distances)[:1]  # Adjust as needed for top-k dissimilar points

# dissimilar_retriever_chain = create_retrieval_chain(dissimilar_search, k=1)


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
#creating api
@app.get('/',response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post('/get_response')
async def get_response(query:str = Form(...)): 
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query) 
    # similar_response = similar_retriever_chain(query)
    # disimilar_response = dissimilar_retriever_chain({'context':"","question":query})
    # response = similar_response
    return jsonable_encoder(response)
    # chain_type_kwargs = {"prompt":prompt}
    # qa = RetrievalQA.from_chain_type(
    #     llm = llm, 
    #     chain_type="stuff", 
    #     retriever = retriever,
    #     chain_type_kwargs=chain_type_kwargs, verbose=True)
    



