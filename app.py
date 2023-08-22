import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_chat import message

def conversational_chat(query, docs, chain):
    response = chain.run(input_documents=docs, question=query)
    st.session_state['history'].append((query, response))
    return response

def main():
    st.markdown("<h1 style='text-align: center;margin: 10px 0px 10px 0px;background-color:black; border-radius:10px ; color: white;'>Chat with PDF 📄</h1>", unsafe_allow_html=True)
    os.environ["OPENAI_API_KEY"] = "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"
    pdf = st.sidebar.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        reader = PdfReader(pdf)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                docsearch = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()  # Instantiate OpenAIEmbeddings
            docsearch = FAISS.from_texts(texts, embedding=embeddings)  # Pass embeddings parameter
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(docsearch, f)

        chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + pdf.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your PDF data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                docs = docsearch.similarity_search(query=user_input, k=3)
                output = conversational_chat(user_input, docs, chain)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
