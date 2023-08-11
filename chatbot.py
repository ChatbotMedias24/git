import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import toml




# Sidebar contents
textcontainer = st.container()
with textcontainer:
    logo_path = "medi.png"
    logoo_path = "pageee.png"
    st.sidebar.image(logo_path,width=250)
    st.sidebar.image(logoo_path,width=100)
    
st.sidebar.subheader("Suggestions:")
questions = [
        "Donne moi un résumé du rapport ",
        "Quelles sont les conditions d'octroi des franchises et tolérances pour les effets personnels ?",
        "Comment obtenir l'admission temporaire pour mon véhicule lors de mon séjour au Maroc ?",
        "Quelles sont les formalités à remplir pour importer des pièces de rechange pour mon véhicule ?",
        "Quels sont les services douaniers disponibles pour les voyageurs au Maroc ?",
        "Quelles sont les importations strictement interdites au maroc ?",
        "Quels documents sont nécessaires pour importer une voiture au Maroc ?"
    ]    
 
load_dotenv(st.secrets["OPENAI_API_KEY"])
 
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = '𝐌𝐚𝐫𝐨𝐜𝐚𝐢𝐧𝐬.pdf'
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        


        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

         # Get the first page as an image

        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        # st.write(chunks)
 
       
        
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open("aaa.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
        selected_questions = st.sidebar.radio("****Choisir :****",questions)
    
        if selected_questions:
           query = st.text_input("Selected Question:", selected_questions)
        else :
           query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
