import lz4.frame
import pickle
import numpy as np
import pandas as pd
import copy
from tqdm.auto import tqdm

def df_merge(df1, df2):
    if df1 is None or df1.empty:
        return df2
    if df2 is None or df2.empty:
        return df1

    reindex = df1.index.union(df2.index)
    return df1.reindex(reindex).fillna(0.) + df2.reindex(reindex).fillna(0.)

def df_open_merge(paths, quiet=False):
    pbar = tqdm(total=len(paths), desc="Merged", dynamic_ncols=True, disable=quiet)
    obj_out = pd.DataFrame()
    for path in paths:
        with lz4.frame.open(path, 'rb') as f:
            obj_in = pickle.load(f)
        obj_out = df_merge(obj_out, obj_in)
        pbar.update()
    pbar.close()
    return obj_out

def df_process(paths, cfg):
    out_df = pd.DataFrame()

    for path in paths:
        store = pd.HDFStore(path)
        for table_label, table_name in cfg["tables"].items():
            hist_cfg = {"table_name": table_label}

            for df in store.select(table_name, iterator=True, chunksize=100000):

                # pre-eval
                for evs in cfg["eval"]:
                    for key, val in evs.items():
                        df[key] = eval("lambda "+val)(df)

                for cutflow_name, cutflow_cfg in cfg["cutflows"].items():
                    hist_cfg.update(cutflow_cfg)

                    # apply selection
                    sdf = df.loc[df.eval(cutflow_cfg["selection"])]

                    for hist_label in cutflow_cfg["hists"]:
                        hdf = sdf.copy()

                        # hist evals
                        evals = cfg["hists"][hist_label]
                        for evs in evals:
                            for key, val in evs.items():
                                hdf[key] = eval("lambda "+val.format(**hist_cfg))(hdf)

                        # add hist
                        columns = [list(ev.keys())[0] for ev in evals]
                        out_df = df_merge(
                            out_df,
                            hdf.loc[:,columns].groupby(cfg["groupby"]).sum(),
                        )
        store.close()
    return out_df
