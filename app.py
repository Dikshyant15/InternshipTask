from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

llm = CTransformers(
    model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

print("LLM initialized")

model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

url = 'http://localhost:6333'
client = QdrantClient(url=url, prefer_grpc=False)

faq_db_retriever = Qdrant(client=client, embeddings=model, collection_name="faq_db").as_retriever()
city_bank_faq_db_retriever = Qdrant(client=client, embeddings=model, collection_name="city_bank_faq_data").as_retriever()

def combined_retrieval(query, k=5):
    # Retrieve top k results from both collections
    faq_results = faq_db_retriever.retrieve(query, top_k=k)
    city_bank_results = city_bank_faq_db_retriever.retrieve(query, top_k=k)

    # Example merge strategy: simple interleaving (you can also sort by score or relevance)
    combined_results = []
    for result_pair in zip(faq_results, city_bank_results):
        combined_results.extend(result_pair)  # This interleaves results

    return combined_results[:k]  # Return the top k results combined

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {query}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'query'])

# Create the RetrievalQA chain using the custom combined retriever
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faq_db_retriever,  # Use one of the retrievers as the base retriever
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)

@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post('/get_response')
async def get_response(query: str = Form(...)):
    # Handle the retrieval chain response properly
    response = retrieval_chain({"query": query})
    # Ensure response is a dictionary
    if isinstance(response, dict):
        return jsonable_encoder(response)
    else:
        return jsonable_encoder({"error": "Invalid response format from retrieval chain"})

