import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load data
df = pd.read_csv("vjti_faqs.csv")
df.dropna(inplace=True)
docs = [Document(page_content=row['Answer'], metadata={"question": row['Question']}) for _, row in df.iterrows()]

# Initialize LangChain
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.from_documents(docs, embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

# UI
st.set_page_config(page_title="VJTI Chatbot", layout="centered")
st.title("🤖 VJTI Helpdesk Chatbot")

st.markdown("Ask any questions about VJTI admissions, MCA, B.Tech, fees, etc.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", placeholder="Ask your question here...")

if user_input:
    response = qa.invoke(user_input)['result']
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Display chat
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)
