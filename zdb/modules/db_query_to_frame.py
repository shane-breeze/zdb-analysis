import pandas as pd
import sqlalchemy as sqla

def db_query_to_frame(path, queries):
    print(path)
    engine = sqla.create_engine("sqlite:///{}".format(path))
    # print(engine.execute("SELECT * FROM Events LIMIT 1"))

    dfs = []
    for label, query in queries.items():
        df = pd.read_sql(query, engine)
        df["selection"] = label
        dfs.append(df)

    return (pd.concat(dfs),)
