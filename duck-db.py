model='duckdb-nsql'
r = ollama.generate(
    model=model,
    system='''Here is the database schema that the SQL query will run on:
{};'''.format(get_schema()),
    prompt='get all movies where actor is Ed Chase',
)

query= r['response']