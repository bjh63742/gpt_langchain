import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(page_title="chat bot", page_icon="🤖")

print("----frist-----")
print(st.session_state)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        save_ai_memory(self.message)

    # 실시간
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer questions using the following context.:\n\n{context}"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)



# 입력 저장 함수
def save_human_memroy(user_input):
    memory.chat_memory.add_message(HumanMessage(content=user_input))

# 출력 저장 함수
def save_ai_memory(ai_output):
    memory.chat_memory.add_message(AIMessage(content=ai_output))

def get_history():
    return memory.load_memory_variables({})

def load_memory(_):
    return memory.load_memory_variables({})["history"]


openai_api_key = None

if "openai_api_key" in st.session_state:
    openai_api_key = st.session_state["openai_api_key"]


print(openai_api_key)
if openai_api_key is not None:
    llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
            openai_api_key = openai_api_key
        )
    if "memory" in st.session_state:
        memory = st.session_state["memory"]
    else:
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True , llm=llm, max_token_limit =20)
        memory = st.session_state["memory"]
    st.title("chat bot")

    st.markdown(
        """
        Welcome!
                    
        Use this chatbot to ask questions to an AI about your files!

        Upload your files on the sidebar.
        """
    )
    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )

    if file:
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                    "history": load_memory,
                }
                | prompt
                | llm
            )
            save_human_memroy(message)
            with st.chat_message("ai"):
                chain.invoke(message)
                


    else:
        st.session_state["messages"] = []

else:
    st.error(
        """
        You need to set your OpenAI API key in the `openai_api_key` variable at the top of the script.
        """
    )

    with st.sidebar:
        my_key = st.text_input(
            "Please input OpenAI API"
        )
        if my_key:
            print("saving")
            print(my_key)
            st.session_state["openai_api_key"] = my_key
            st.rerun()
        print("done")