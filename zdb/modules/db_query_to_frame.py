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

def merge_results(results, index=None):
    df = None
    for result in results:
        dfr = result[0].set_index(index)

        if df is None:
            df = dfr
        else:
            df = (
                df.reindex_like(df+dfr).fillna(0)
                + dfr.reindex_like(df+dfr).fillna(0)
            )

    return df.reset_index()
