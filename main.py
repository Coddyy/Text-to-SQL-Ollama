import ollama
from db import get_schema, db
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM
from langchain.agents import AgentType
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder,AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
)
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool

llm = OllamaLLM(model="mistral")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

sql_db_query =  QuerySQLDataBaseTool(db = db)
sql_db_schema =  InfoSQLDatabaseTool(db = db)
sql_db_list_tables =  ListSQLDatabaseTool(db = db)
sql_db_query_checker = QuerySQLCheckerTool(db = db, llm = llm)

tools = [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]

prefix = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
{tools}

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

"""

suffix = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.

Thought:{agent_scratchpad}
"""


messages = [
                SystemMessagePromptTemplate.from_template(prefix),
                HumanMessagePromptTemplate.from_template("{input}"),
                AIMessagePromptTemplate.from_template(suffix),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]

prompt = ChatPromptTemplate.from_messages(messages)



# MSSQL_PROMPT = """You are an MS SQL expert. Given an input question, first create a syntactically correct MS SQL query to run, then look at the results of the query and return the answer to the input question.

# Use the following Domain Knowledge about Database: One Order can have multiple shipments & shipment containers. 

# Use the following format:

# Question: Question here
# SQLQuery: SQL Query to run
# SQLResult: Result of the SQLQuery
# Answer: Final answer here

# Only use the following tables:
# {table_info}

# Thought: I should look at the tables in the database to see what I can query. Then I should query the schema of the most relevant tables.
# {agent_scratchpad}

# Question: {input}"""


agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    prompt=prompt
)


q = "How many Ed Chase movies are there?"
res = agent_executor.invoke({
        "input": "How many Ed Chase movies are there?",
        "tool_names" : [tool.name for tool in tools],
        "tools" : [tool.name + " - " + tool.description.strip() for tool in tools],
        "agent_scratchpad": []
    })
print("SQL Query Result:")
print(res)