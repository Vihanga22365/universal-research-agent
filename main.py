import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import TaskOutput
import re



load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="JPMC POC"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"



from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import requests
from crewai import Agent, Task, Crew

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def format_docs(docs):
  format_D="\n\n".join([d.page_content for d in docs])
  return format_D

@tool("rag_inquiry_handler_tool")
def rag_inquiry_handler_tool(question: str) -> str:
    """
      Ask a question using the RAG.
      User can ask a question about our company and get the answer from the RAG.
    
      Args:
          question (str): The question to ask
          
      Returns:
          str: The answer to the question
        
           
    """
    docsearch = FAISS.load_local(os.path.join(os.path.dirname(__file__), "./rag/faiss_db/"), embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})
    
    template = """
            You are report generator.
            In the 'Question' section include the question.
            In the 'Context' section include the nessary context to generate the section of the report.
            According to the 'Context', please generate the section of the report.
            Use only the below given 'Context' to generate the section of the report.
            Make sure to provide the answer from the 'Context' only.
            Provide answer only. Nothing else.
            Context: {context}
            Question : {question}
            """
              
    prompt=ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(question)
    return response


@tool
def code_interpreter_tool(code: str) -> str:
    """
    Executes the provided Python code and returns the output.
    """
    # Define a restricted execution environment
    restricted_globals = {"__builtins__": {}}
    restricted_locals = {}

    try:
        # Execute the code within the restricted environment
        exec(code, restricted_globals, restricted_locals)
        return str(restricted_locals)
    except Exception as e:
        return f"Error: {e}"
    
    
@tool("online_web_search")
def online_web_search(search_query: str):
    """
    Perform an online web search.
    
    Parameters:
        search_query (str): The search query.
    
    Returns:
        dict: The search results in JSON format.
    """
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    data = {"api_key": "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", "query": search_query, "max_results": 5}
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()



st.set_page_config(layout="wide", page_title="Universal Research Assistant", page_icon="🤖")


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 2%;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px, rgb(209, 213, 219) 0px 0px 0px 1px inset;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>Universal Research Assistant</h1>
    """,
    unsafe_allow_html=True
)


researcher_agent = Agent(
    role="Universal Research Assistant",
    goal="To assist users in finding information on the web, in documents, and through calculations.",
    backstory="You're an experienced researcher, skilled in finding information on the web, in documents, and through calculations. Your role is to assist users in finding the information they need.",
    tools=[rag_inquiry_handler_tool, online_web_search, code_interpreter_tool],
    verbose=True,
    allow_delegation=False,
)


researcher_task = Task(
    description="""
        **User Query** : {user_query}
        
        **Tools** 
            - 'rag_inquiry_handler_tool' - Make sure only search and gather the required information about the our company 'JPMorgan Chase & Co'.
            - 'online_web_search' -  Make sure only search and gather the required information about the user given company, other than our company 'JPMorgan Chase & Co'.
            - 'code_interpreter_tool' - Execute the provided Python code and return the output.
        
        **Objective**
            - You are a Researcher Agent In Financial Company and your goal is to assist users in finding information on the web, in documents, and through calculations.
            - Clearly understand the 'User Query' and use the appropriate tool to get the required information.
            - Generate the answer based on the information gathered.    
            
        **Instructions**
            - If you want to search and gather the required information about the our company 'JPMorgan Chase & Co', use the tool 'rag_inquiry_handler_tool'.
            - If you want to search and gather the required information about the user given company, other than our company 'JPMorgan Chase & Co', use the tool 'online_web_search'.
            - If you want to execute the provided Python code and return the output, use the tool 'code_interpreter_tool'.
            - If you want to perform any calculations, don't do it manually, generate the python code and use the tool 'code_interpreter_tool' to execute the code and get the output.
            - Think step by step about the 'User Query' and use the appropriate tool to get the required information.
            - If you can't find the required information, using the available tools, mention it in final answer. 
            - If you can't find the required information, don't answer those by yourself, mention it in final answer.
            - Finally, provide the answer to the 'User Query' based on the information gathered.
            
            
            Example:
                If user ask : "What is Director's name of JPMC and Director's name of Meta Company?" (This is related to your company or your company's competitor)
                Gather the information from the 'rag_inquiry_handler_tool' about the Director's name of JPMC.
                Gather the information from the 'online_web_search' about the Director's name of Meta Company.
                
                Output :  
                    **Name of the Director of JPMC**
                        <Director's Name of JPMC>
                    
                    **Name of the Director of Meta Company**
                        <Director's Name of Meta Company>
                        
                        
                If user ask : "Director's name of Meta Company?" (User directly asked about the other company)
                Mention that you can't find the required information.
                Output :
                    I can't assist to get the information about the Director's name of Meta Company. Because, it's not related to the my capabilities. My capabilities are limited to the information about the company 'JPMorgan Chase & Co'.
                               
                        
                If user ask : "What is Director's name of JPMC and Meta Company. after tell me the revenue of JPMC in 2023?" (This is related to your company or your company's competitor)
                Gather the information from the 'rag_inquiry_handler_tool' about the Director's name of JPMC and Gather the information from 'rag_inquiry_handler_toolt' for calculating the revenue of JPMC in 2023.
                Gather the information from the 'online_web_search' about the Director's name of Meta Company.
                Generate the python code to calculate the revenue of JPMC using the tool 'code_interpreter_tool'.
                Output :  
                
                    **Name of the Director of JPMC**
                        <Director's Name of JPMC>
                        
                    **Revenue of JPMC in 2023**
                        <Revenue of JPMC in 2023>
                        
                    **Name of the Director of Meta Company**
                        <Director's Name of Meta Company>
                        
                        
                If user ask : "What is the Capital of India?" (This is not related to your company)
                Mention that you can't find the required information.
                Output :  
                    I can't assist to get the information about the capital of India. Because, it's not related to the my capabilities. My capabilities are limited to the information about the company 'JPMorgan Chase & Co'.

    """,
    expected_output="""
    
            Give the high-quality output according to the 'User Query' with step by step process like below.
            
            Example:
                If user ask : "What is Director's name of JPMC and Director's name of Meta Company?"
                Output :  
                    **Name of the Director of JPMC**
                        <Director's Name of JPMC>
                    
                    **Name of the Director of Meta Company**
                        <Director's Name of Meta Company>
                        
                
                If user ask : "What is Director's name of JPMC and Meta Company. after tell me the revenue of JPMC in 2023?"
                Output :  
                    **Name of the Director of JPMC**
                        <Director's Name of JPMC>
                        
                    **Revenue of JPMC in 2023**
                        <Revenue of JPMC in 2023>
                        
                    **Name of the Director of Meta Company**
                        <Director's Name of Meta Company>
                        
                        
                If user ask : "What is the Capital of India?"
                Output :  
                    I can't assist to get the information about the capital of India. Because, it's not related to the my capabilities. My capabilities are limited to the information about the company 'JPMorgan Chase & Co'.
                    
                    
                If user ask : "What is the revenue of JPMC in 2023 and What is the capital of India?"
                Output :  
                    **Revenue of JPMC in 2023**
                        <Revenue of JPMC in 2023>
                        
                    I can't assist to get the information about the capital of India. Because, it's not related to the my capabilities. My capabilities are limited to the information about the company 'JPMorgan Chase & Co'.

        """,
        agent=researcher_agent,

)   


crew = Crew(
    agents=[researcher_agent],
    tasks=[researcher_task],
    verbose=False,
) 


user_query = st.text_input("Enter Your Question:", autocomplete="off")

if st.button("Run") and user_query:
    with st.spinner("Executing Agents..."):
        result1 = crew.kickoff(inputs={"user_query": user_query})
        with st.container():
            text1 = result1.raw
            fixed_text1 = text1.replace("$", "\$")
            st.write(fixed_text1)