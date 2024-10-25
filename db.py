import sqlite3
from langchain_community.utilities import SQLDatabase
import pyparsing
# sql_query="select * from sqlite_master where type='table'"

# try:
#     sql_response = sqlite3.connect("videogames.db").cursor().execute(sql_query).fetchall()
#     #print(sql_response)
#     for table in sql_response:
#         print(table[4])
# except Exception as e:
#     print(f"SQL querying failed. Query:\n{sql_query}\n\n")
#     raise (e)



# POSTGRES
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:postgres@localhost/dvdrental")


def get_schema():
    schema = db.get_table_info()
    comment = pyparsing.nestedExpr("/*", "*/").suppress()
    schema = comment.transformString(schema)
    # f = open("schema.txt", "a")
    # f.write(schema)
    # f.close()
    return schema
