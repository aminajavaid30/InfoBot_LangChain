import streamlit as st
from infobot_langchain import get_response

st.title("InfoBot - Your AI Assistant")
st.subheader("LangChain")

query = st.text_input("Ask InfoBot a question:")

if st.button("Submit"):
    with st.spinner("Generating answer..."):
        if query:
            response = get_response(query)
            st.write(response)
        else:
            st.write("Please enter a question.")
