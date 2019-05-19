import pandas as pd
import sqlalchemy as sqla
import tqdm

def db_query_to_frame(path, query):
    #print(path)
    engine = sqla.create_engine("sqlite:///{}".format(path))
    return (pd.read_sql(query, engine),)

def merge_results(results, index, disable=False):
    df = None
    if len(results)==1:
        return results[0][0]
    for result in tqdm.tqdm(results, desc="Processing", dynamic_ncols=True, unit="files", disable=disable):
        dfr = result[0].set_index(index)
        dfr = dfr.loc[dfr.index.dropna(),:]

        if df is not None and df.index.duplicated().any():
            print("Duplicates in df")
            print(df.loc[df.index.duplicated(keep=False), :].to_string())
        if dfr is not None and dfr.index.duplicated().any():
            print("Duplicates in dfr")
            print(dfr.loc[dfr.index.duplicated(keep=False), :].to_string())

        if df is None:
            df = dfr
        else:
            df = (
                df.reindex_like(df+dfr).fillna(0)
                + dfr.reindex_like(df+dfr).fillna(0)
            )

    return df.reset_index()
