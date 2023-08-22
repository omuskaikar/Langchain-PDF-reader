import os
import pickle
import atexit
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message


def conversational_chat(query, docs, chain):
    response = chain.run(input_documents=docs, question=query)
    st.session_state['history'].append((query, response))
    return response


def delete_pickle_files():
    # Iterate through all files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.pkl'):
            os.remove(filename)


def main():
    st.markdown(""
                "<h1 style='text-align: center;margin: 10px 0px 10px 0px;background-color:black;"
                "border-radius:7px ; color: white;position: relative;z-index:1000'>"
                "Chat with PDF 📄</h1>", unsafe_allow_html=True)
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

        user_input = st.chat_input("Your Query:")

        if user_input:
            # Process user input and generate bot response
            docs = docsearch.similarity_search(query=user_input, k=3)
            output = conversational_chat(user_input, docs, chain)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

        atexit.register(delete_pickle_files)


if __name__ == "__main__":
    main()
