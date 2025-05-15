import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from io import BytesIO
import shutil
import uuid

# Set up page
st.set_page_config(page_title="BytePages - Chat With PDFs", layout="wide")
load_dotenv()
genai.configure(api_key=os.getenv("Google_API_KEY"))

# Shared embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Session setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def getPdfText(pdf_docs):
    text = ""
    page_count = 0
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            page_count += len(pdf_reader.pages)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    st.session_state["all_text"] = text
    st.session_state["page_count"] = page_count
    return text

def getTextChunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.session_state["chunk_count"] = len(chunks)
    return chunks

def get_vector_store(chunks):
    vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context below. Be accurate and detailed.
    If the answer is not present, respond with "Answer not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        reply = response.get("output_text", "No output generated.")
        st.session_state.chat_history.append((user_question, reply))
        st.markdown(f"**🤖 Gemini:** {reply}")
    except Exception as e:
        st.error(f"Error: {e}")

def display_chat_history():
    if st.session_state.chat_history:
        st.subheader("🧠 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")

def download_transcript():
    if not st.session_state.chat_history:
        st.warning("No chat history to download.")
        return
    filename = f"chat_transcript_{uuid.uuid4().hex[:8]}.txt"
    content = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
    st.download_button("📄 Download Chat Transcript", content, file_name=filename)

def main():
    st.title("📚 BytePages - Chat with One or More PDFs")

    # Chat input
    if "faiss_index" in os.listdir():
        user_question = st.text_input("💬 Ask a question from the PDFs")
        if user_question:
            user_input(user_question)

    # Sidebar
    with st.sidebar:
        st.title("📂 Menu")

        pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("🚀 Submit & Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = getPdfText(pdf_docs)
                    chunks = getTextChunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ PDFs processed successfully!")

                    # Show PDF Stats
                    word_count = len(st.session_state["all_text"].split())
                    st.subheader("📊 PDF Stats")
                    st.markdown(f"- **Total Pages:** {st.session_state.get('page_count', '?')}")
                    st.markdown(f"- **Total Words:** {word_count}")
                    st.markdown(f"- **Text Chunks:** {st.session_state.get('chunk_count', '?')}")
            else:
                st.error("Please upload at least one PDF.")

        # Search inside PDFs
        if "all_text" in st.session_state:
            st.markdown("---")
            st.subheader("🔍 Search Inside PDFs")
            keyword = st.text_input("Search keyword or phrase")
            if keyword:
                matches = [line.strip() for line in st.session_state["all_text"].split("\n") if keyword.lower() in line.lower()]
                if matches:
                    st.success(f"Found {len(matches)} match(es). Showing top 10:")
                    for i, line in enumerate(matches[:10]):
                        st.markdown(f"**{i+1}.** {line}")
                else:
                    st.warning("No matches found.")

        # Download chat
        st.markdown("---")
        download_transcript()

        # Reset
        if st.button("🧹 Reset All"):
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.session_state.clear()
            st.experimental_rerun()

    # Chat history
    display_chat_history()

if __name__ == "__main__":
    main()