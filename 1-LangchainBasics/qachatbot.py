"""
Simple LangChain Streamlit App with Groq
A beginner-friendly version focusing on core concepts
"""

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

## Page config
st.set_page_config(page_title="Simple LangChain Chatbot with Groq", page_icon="🚀")

# Title
st.title("🚀 Simple LangChain Chat with Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference!")

with st.sidebar:
    st.header("Settings")

    ## APi Key
    api_key=st.text_input("GROQ API Key", type="password",help="GET Free API Key at console.groq.com")

    ## Model Selection
    model_name=st.selectbox(
        "Model",
        ["llama3-8b-8192", "gemma2-9b-it"],
         index=0
    )

    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

## Initialize LLM
@st.cache_resource
def get_chain(api_key,model_name):
    if not api_key:
        return None
    
    ## Initialize the GROQ Model
    llm=ChatGroq(groq_api_key=api_key,
             model_name=model_name,
             temperature=0.7,
             streaming=True)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])

    ## create chain
    chain=prompt| llm | StrOutputParser()

    return chain

## get chain
chain=get_chain(api_key,model_name)

if not chain:
    st.warning("👆 Please enter your Groq API key in the sidebar to start chatting!")
    st.markdown("[Get your free API key here](https://console.groq.com)")

else:
    ## Display the chat messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    
    ## chat input
    if question:= st.chat_input("Ask me anything"):
        ## Add user message to session state
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response from Groq
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

## Examples

st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")

        


    




