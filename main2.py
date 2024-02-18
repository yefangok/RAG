from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter,SpacyTextSplitter
from text_splitter.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from text_splitter.ali_text_splitter import AliTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI

from langchain.document_loaders import UnstructuredFileLoader,JSONLoader,UnstructuredMarkdownLoader,UnstructuredHTMLLoader
from document_loaders import mypdfloader

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.milvus import Milvus

from milvus import default_server

loader = UnstructuredFileLoader('docs/test.txt')
docs = loader.load()

text_splitter = SpacyTextSplitter.from_tiktoken_encoder(encoding_name='gpt2',pipeline="zh_core_web_sm",chunk_size=10,chunk_overlap=5)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name='gpt2',chunk_size=10,chunk_overlap=5)
text_splitter = ChineseRecursiveTextSplitter.from_tiktoken_encoder(encoding_name='gpt2',chunk_size=10,chunk_overlap=5)
ali_text_splitter = AliTextSplitter.from_tiktoken_encoder(encoding_name='gpt2',chunk_size=10,chunk_overlap=5)

docs2 = ali_text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",     #BAAI/bge-m3
    model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
# em = embeddings.embed_query("hi this is harrison")
# len(em)

llm = AzureChatOpenAI(
   openai_api_key="d81b780a35aa4311878519c576777d87",openai_api_base="https://gptyyds.openai.azure.com/",
   deployment_name="gpt-35-turbo",openai_api_type = "azure",openai_api_version = "2023-10-01-preview",temperature=0)


from llama_index.core import VectorStoreIndex,SummaryIndex, StorageContext
from llama_index.core import Document
from llama_index.core import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.azure_openai import AzureOpenAI

llm = AzureOpenAI(
   api_key="d81b780a35aa4311878519c576777d87",azure_endpoint="https://gptyyds.openai.azure.com/",
   engine="gpt-35-turbo",openai_api_type = "azure",api_version = "2023-10-01-preview",temperature=0)

# creates a persistant index to disk
client = QdrantClient(path="./qdrant_data")

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "llama2_paper", client=client, enable_hybrid=True, batch_size=20
)

#m3 /home/user/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/3ab7155aa9b89ac532b2f2efcc3f136766b91025
#BAAI/bge-base-zh-v1.5:  

service_context = ServiceContext.from_defaults(
    embed_model=HuggingFaceEmbedding(model_name="/home/user/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots/b5c9d86d763d9945f7c0a73e549a4a39c423d520"),
    llm=llm,
    node_parser=SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=32)
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, service_context=service_context
)

summary_index = SummaryIndex.from_documents

for document in docs:
   document = Document.from_langchain_format(document)
   index.insert(document)

query_engine = index.as_query_engine(
    similarity_top_k=2, sparse_top_k=12, vector_store_query_mode="hybrid"
)

response = query_engine.query(
    "How was Llama2 specifically trained differently from Llama1?"
)


# with default_server:
#    vector_db = Milvus.from_documents(
#       docs2,
#       embeddings,
#       connection_args={"host": "127.0.0.1", "port": default_server.listen_port},
#    )
#    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
#    query = "What is Zilliz Cloud?"
#    qa.run(query)
