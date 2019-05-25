import numpy as np
import pandas as pd
import sqlalchemy as sqla
import tqdm

def db_query_to_frame(path, query):
    #print(path)
    engine = sqla.create_engine("sqlite:///{}".format(path))
    return (pd.read_sql(query, engine),)

def db_query_to_frame_processing(path, query_eval_groupby):
    results = []
    for query, evals, groupby in query_eval_groupby:
        df = db_query_to_frame(path, query)[0]
        for ev in evals:
            df[ev["name"]] = np.empty(df.shape[0])
            if not df.empty:
                df.loc[:, ev["name"]] = df.eval(ev["eval"])
        val_cols = [c for c in df.columns if c.startswith("sum_w")]
        tdf = df.loc[:,groupby+val_cols].groupby(groupby).sum()
        if not tdf.empty:
            results.append((tdf.reset_index(),))
    return (merge_results(results, groupby, disable=True),)

def merge_results(results, index, disable=False):
    df = None
    if len(results)==0:
        return pd.DataFrame()
    elif len(results)==1:
        return results[0][0]
    for result in tqdm.tqdm(results, desc="Processing", dynamic_ncols=True, unit="files", disable=disable):
        if result[0] is None:
            continue
        if result[0].empty:
            continue
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

    if df is None:
        return df
    return df.reset_index()
