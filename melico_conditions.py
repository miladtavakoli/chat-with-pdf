from langchain.memory import ConversationBufferMemory

import settings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import streamlit as st

import vector_db

embeddings = OpenAIEmbeddings()
st.set_page_config(
    page_title="بیمه خاورمیانه",
    page_icon="https://melico.ir/favicon.ico",
    layout="wide",
    menu_items={}
)
with open("custom-css.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


def main_open_ai():
    memory = ConversationBufferMemory(memory_key="chat_history")
    vectordb = vector_db.get_vector_db(embeddings=embeddings, create_new=False)
    chain = load_qa_chain(OpenAI(temperature=0.5, max_tokens=1000), chain_type='stuff')
    st.title('شرایط عمومی بیمه نامه')
    query = st.text_input('سوال خودت رو بپرس')
    if query:
        similarity_docs = vectordb.similarity_search(query)
        result = chain.run(input_documents=similarity_docs, question=query, memory=memory)
        st.write(result)


if __name__ == '__main__':
    main_open_ai()
