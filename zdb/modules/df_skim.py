import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def df_skim(paths, cfg, outpath, quiet=False, chunksize=500000):
    pbar_tab = tqdm(cfg["tables"].items(), disable=quiet, unit="table")
    for table_label, table_name in pbar_tab:
        pbar_tab.set_description(table_name)

        pbar_path = tqdm(paths, disable=quiet, unit="file")
        for path in pbar_path:
            pbar_path.set_description(os.path.basename(path))

            for df in pd.read_hdf(
                path, table_name, iterator=True, chunksize=chunksize,
            ):
                tdf = df.query(cfg["selection"])

                tdf.loc[:, "sample"] = tdf["sample"].astype("str")
                tdf.loc[:, "parent"] = tdf["parent"].astype("str")

                if not tdf.empty:
                    tdf.to_hdf(
                        outpath, table_name, mode='a', format='table',
                        append=True, complevel=9, complib='zlib',
                        min_itemsize={"sample": 50, "parent": 50},
                    )

    return True
