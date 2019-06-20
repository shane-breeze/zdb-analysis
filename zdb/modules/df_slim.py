import numpy as np
import pandas as pd

def df_slim(paths, cfg, outpath):
    for table_label, table_name in cfg["tables"].items():
        for path in paths:
            for df in pd.read_hdf(path, table_name, iterator=True, chunksize=500_000):
                tdf = pd.DataFrame()

                for evs in cfg["columns"]:
                    for key, val in evs.items():
                        print(val)
                        print(df.eval(val))
                        tdf[key] = df.eval(val, engine='python')

                tdf.to_hdf(
                    outpath, table_name, mode='a', format='table', append=True,
                    complevel=9, complib='blosc:lz4hc',
                )

    return True
