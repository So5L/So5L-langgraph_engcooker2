import streamlit as st

st.title("Education Agent")
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown(f"입력한 질문: {user_input}")