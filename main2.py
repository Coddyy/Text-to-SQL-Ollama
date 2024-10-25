from langchain_community.llms import Ollama
from db import get_schema, db
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM
from langchain.agents import AgentType
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.agents import AgentExecutor, create_react_agent


llm = OllamaLLM(model="llama3.2")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

embeddings = (
    OllamaEmbeddings(model = "llama3.2")
)

examples = [
    {   "input": "List all actors.", 
        "query": "SELECT * FROM Actor;"
    },
    {
        "input": "Find all movies of Ed Chase",
        "query": "SELECT film.title, concat(actor.first_name, ' ', actor.last_name) as actorname from film LEFT JOIN filmactor on film.film_id=filmactor.film_id LEFT JOIN actor on actor.actor_id=filmactor.actor_id WHERE concat(actor.first_name, ' ', actor.last_name) LIKE '%Ed Chase%'"
    },
    {
        "input": "Find all customers for the postal code 35200.",
        "query": "SELECT first_name,last_name,address_id FROM customer WHERE address_id = (SELECT address_id FROM address WHERE postal_code = '35200');",
    },
    {
        "input": "Find full address of Mary Smith.",
        "query": "SELECT address, address2, district, postal_code from address where address_id = (select address_id from customer where concat(first_name,' ', last_name) LIKE '%Mary Smith%');",
    },
    {
        "input": "How many customers are there",
        "query": 'SELECT COUNT(*) FROM customer',
    },
    {
        "input": "Find the total number of actors.",
        "query": "SELECT COUNT(DISTINT(actor_id)) FROM Actor;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT customer.customer_id AS customer_id, concat(customer.first_name, ' ', customer.last_name) as customer_name, SUM(payment.amount) AS TotalPurchase FROM payment LEFT JOIN customer on customer.customer_id=payment.customer_id GROUP BY customer.customer_id ORDER BY TotalPurchase DESC LIMIT 5;",
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=3,
    input_keys=["input"],
)

sql_db_query =  QuerySQLDataBaseTool(db = db)
sql_db_schema =  InfoSQLDatabaseTool(db = db)
sql_db_list_tables =  ListSQLDatabaseTool(db = db)
sql_db_query_checker = QuerySQLCheckerTool(db = db, llm = llm)


tools = [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]

# matched_queries = example_selector.vectorstore.search("How many actors are there?", search_type = "mmr")


# for tool in tools:
#     print(tool.name + " - " + tool.description.strip() + "\n")

system_prefix = """

You are an agent designed to interact with a SQL database. Given an input question, create a syntactically and logically correct query to run, then look at the results of the query and return the answer. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here are some examples of user inputs and their corresponding SQL queries:

"""

suffix = """
Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

dynamic_few_shot_prompt_template = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix=suffix
)


full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=dynamic_few_shot_prompt_template),
    ]
)

# prompt_val = full_prompt.invoke(
#     {
#         "input": "How many actors are there?",
#         "tool_names" : [tool.name for tool in tools],
#         "tools" : [tool.name + " - " + tool.description.strip() for tool in tools],
#         "agent_scratchpad": [],
#     }
# )
# print(prompt_val.to_string())


agent = create_react_agent(llm, tools, full_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
agent_executor.invoke({"input": "Find where Nancy Thomas lives?"})



