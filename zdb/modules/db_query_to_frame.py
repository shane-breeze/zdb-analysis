import pandas as pd
import sqlalchemy as sqla

def db_query_to_frame(path, query):
    print(path)
    engine = sqla.create_engine("sqlite:///{}".format(path))
    return (pd.read_sql(query, engine),)

def merge_results(results, index):
    df = None
    for result in results:
        dfr = result[0].set_index(index)

        if df is not None and df.duplicated().any():
            print(df.loc[df.duplicated(), :].to_string())
        if dfr is not None and dfr.duplicated().any():
            print(dfr.loc[dfr.duplicated(), :].to_string())

        if df is None:
            df = dfr
        else:
            df = (
                df.reindex_like(df+dfr).fillna(0)
                + dfr.reindex_like(df+dfr).fillna(0)
            )

    return df.reset_index()
