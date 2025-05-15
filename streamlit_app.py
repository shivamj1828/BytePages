import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from io import BytesIO

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Chat with PDFs")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load credentials
with open("credentials.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Instantiate the Authenticate class without the `preauthorized` parameter
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

# Register users, passing pre_authorized to register_user
authenticator.register_user(pre_authorized=config["preauthorized"])

name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("Incorrect username or password")

if auth_status is None:
    st.warning("Please enter your username and password")

if auth_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome {name}!")

    st.title("📄 Chat with your PDFs")
    st.header("💬 Gemini-powered Q&A from uploaded PDFs")

    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            try:
                reader = PdfReader(BytesIO(pdf.read()))
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text
            except Exception as e:
                st.error(f"PDF reading error: {e}")
        return text

    def get_text_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    def store_vectors(chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        store = FAISS.from_texts(chunks, embedding=embeddings)
        store.save_local("faiss_index")
        return store

    def get_chain():
        prompt_template = """Answer the question based on the context below.
If the answer is not available, say "You're Stupid & Answer not available in the context."

Context: {context}
Question: {question}
Answer:
"""
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def answer_question(user_qn):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_qn)
        chain = get_chain()
        response = chain({"input_documents": docs, "question": user_qn}, return_only_outputs=True)
        st.write("Reply:", response.get("output_text", "No response found."))

    user_question = st.text_input("Ask a question from the uploaded PDF(s)")
    if user_question:
        answer_question(user_question)

    with st.sidebar:
        st.subheader("📤 Upload PDFs")
        pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            if pdfs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdfs)
                    chunks = get_text_chunks(raw_text)
                    store_vectors(chunks)
                    st.success("PDFs processed!")
            else:
                st.error("Please upload at least one PDF.")
