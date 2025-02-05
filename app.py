import streamlit as st
import os
from langchain_community.vectorstores import Pinecone as PineconeVectorStore 
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from src.helper import download_huggingface_embedding, load_data, load_data_from_uploaded_pdf, load_data_from_url,text_split
def main():

    PINECONE_INDEX_NAME = "medical"
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    embeddings = download_huggingface_embedding()

    #load environment variable
    load_dotenv()

    #configure streamlit page settings
    st.set_page_config(page_title="Medical AI Assistant",
                       page_icon=" 💊",
                       layout="centered")
    
    st.title("Medical AI Assistant 💊")
    st.markdown("### Your AI-powered Assistant for medical queries and document analysis")
    st.markdown("#### 📂 Upload a PDF or URL or Use Default data to get Started")

    # Upload file as input
    uploaded_file = st.sidebar.file_uploader("Upload a PDF File", type="pdf")

    # URL as input
    url = st.sidebar.text_input("Enter a URL (Optional)")

    # Button to use default data
    use_default = st.sidebar.checkbox("Use default data", value=True)

    # Placeholder to display user choice or data
    if uploaded_file:
        print(uploaded_file)
        st.success(f" Processing {uploaded_file.name} file... PDF Uploaded Successfully!")
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_data_from_uploaded_pdf("uploaded_file.pdf")
        doc_chucks = text_split(docs)
        docsearch = Chroma.from_documents(documents=doc_chucks,
                                           embedding=embeddings,
                                           collection_name="PDF_database",
                                           persist_directory="./chroma_db_pdf")
    elif url:
        st.success("Provided URL : {}".format(url))
        docs = load_data_from_url(url=url)
        doc_chucks = text_split(docs)
        docsearch = Chroma.from_documents (documents=doc_chucks,
                                           embedding=embeddings,
                                           collection_name="URL_database",
                                           persist_directory="./chroma_db_url")
        st.success("Index loaded Successfully!")

    elif use_default:
        st.success("Using Medical data!")
        #Loading the index
        try:
            docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
            st.success("Index loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading index: {e}")

    else:
        st.info("Please Upload a File, Enter a URL, or Select default data to Proceed.")
        st.stop()
    
    # prompt_template = """
    # Use the given information context to give appropriate answer for the user's question.
    # If you don't know the answer, just say that you know the answer, but don't make up an answer.
    # Context: {context}
    # Question: {question}
    # Only return the appropriate answer and nothing else.
    # Helpful answer:
    # """

    prompt_template = """
    Use the given context to answer the user's question. 
    If the context does not contain relevant information, reply with:
    "I can only answer medical-related questions based on my knowledge base. Please provide a relevant query."
    Do NOT generate an answer from general knowledge.
    Context: {context}
    Question: {question}
    Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs={"prompt": PROMPT}

    # llm
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="mixtral-8x7b-32768", 
        temperature=0.5,
        max_tokens=1000,
        timeout=60
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docsearch.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    input = st.chat_input("Ask Your Question")
    print(input)
    if input:
        result=qa.invoke(input)
        print("Response : ", result["result"])
        response = result["result"]
        st.session_state["chat_history"].append((input, response))

    # Display chat history
    for question, answer in st.session_state["chat_history"]:
        st.write(f"**🧑🏻 :** {question}")
        st.write(f"**👩🏻‍⚕️ :** {answer}")

if __name__=="__main__":
    main()