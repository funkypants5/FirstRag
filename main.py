import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

os.environ["LANGSMITH_TRACING"] = "true"
# Make sure GOOGLE_API_KEY is set in your system environment
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not set. Please run `setx GOOGLE_API_KEY \"your-key\"` in PowerShell.")

# Initialize Gemini model
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = InMemoryVectorStore(embeddings)



# Load multiple PDFs from the data folder
pdf_files = glob.glob("data/*.pdf")

docs = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())   # extend with all pages from this PDF

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

# assert len(example_messages) == 1
# print(example_messages[0].content)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "Where should I go if i want to eat the best chicken rice in singapore. Give me 1 answer only"})
print(f"Context: {result['context']}\n\n")
print(f"Answer: {result['answer']}")