import numpy as np
import pandas as pd

def df_skim(paths, cfg, outpath):
    for table_label, table_name in cfg["tables"].items():
        for path in paths:
            for df in pd.read_hdf(path, table_name, iterator=True, chunksize=500_000):
                tdf = df.query(cfg["selection"])

                tdf.loc[:, "sample"] = tdf["sample"].astype("str")
                tdf.loc[:, "parent"] = tdf["parent"].astype("str")

                tdf.to_hdf(
                    outpath, table_name, mode='a', format='table', append=True,
                    complevel=9, complib='blosc:lz4hc',
                    min_itemsize={"sample": 50, "parent": 50},
                )

    return True
