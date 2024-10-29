import os
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer  # Import the embedding model
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser

# Streamlit app configuration
st.set_page_config(page_title="AI Text Assistant", page_icon="🤖")

# Title and greeting message
st.title('AI Chatbot')
st.markdown("Hello! I'm your AI assistant. I can help answer questions about your uploaded PDF. You can also change the PDF file if needed and ask questions about the new content.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text was extracted from the page
                text += page_text + "\n"
        return text
    except PyPDF2.errors.PdfReadError as pdf_error:
        st.error(f"Failed to read PDF: {pdf_error}")
    except Exception as e:
        st.error(f"An error occurred while extracting text: {e}")
    return ""

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size):
    # Split text into words and create chunks
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to create embeddings
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model
    embeddings = model.encode(chunks)  # Generate embeddings for the chunks
    return embeddings

# Initialize a session state to store the PDF text, file name, and embeddings
if 'pdf_text' not in st.session_state:
    st.session_state['pdf_text'] = ""
if 'pdf_name' not in st.session_state:
    st.session_state['pdf_name'] = ""
if 'pdf_chunks' not in st.session_state:
    st.session_state['pdf_chunks'] = []
if 'pdf_embeddings' not in st.session_state:
    st.session_state['pdf_embeddings'] = []  # Add embeddings to session state

# PDF file upload section
st.subheader("Upload or Change PDF File")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the new uploaded PDF file
    st.session_state['pdf_text'] = extract_text_from_pdf(uploaded_file)
    st.session_state['pdf_name'] = uploaded_file.name
    # Chunk the extracted text
    CHUNK_SIZE = 100  # Define your chunk size
    st.session_state['pdf_chunks'] = chunk_text(st.session_state['pdf_text'], CHUNK_SIZE)

    # Create embeddings for the chunks
    st.session_state['pdf_embeddings'] = create_embeddings(st.session_state['pdf_chunks'])

    st.success(f"PDF '{uploaded_file.name}' uploaded and text extracted successfully!")

# Show current PDF name or message if no PDF is uploaded
if st.session_state['pdf_name']:
    st.write(f"Currently selected PDF: **{st.session_state['pdf_name']}**")
else:
    st.write("No PDF file selected.")

# Display the extracted PDF text
if st.session_state['pdf_text']:
    st.text_area("Extracted PDF Text:", value=st.session_state['pdf_text'], height=300)

# AI assistant ke liye prompt template create karna
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("You are a helpful AI assistant. Please respond to user queries in English, and help with the PDF content if relevant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Chat message history initialization
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Setup AI model
api_key = "AIzaSyCG5x3VXfUOh7uuD0AbmUP88GIsDrg-xdY"  # Enter your Google API key here
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create the processing chain
chain = prompt | model | StrOutputParser()

# Combine chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# User text input for query
user_input = st.text_input("Enter your question in English:", "")

if user_input:
    st.chat_message("human").write(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        config = {"configurable": {"session_id": "any"}}

        # Use user input and extracted/modified PDF chunks to get AI response
        response = chain_with_history.stream({"question": user_input, "pdf_text": st.session_state['pdf_chunks']}, config)

        for res in response:
            full_response += res or ""
            message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
else:
    st.warning("Please enter your question.")

