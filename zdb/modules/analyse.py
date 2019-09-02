import numpy as np
import pandas as pd
import pysge
import yaml
import functools

from zdb.modules.df_process import df_process, df_merge, df_open_merge

def analyse(
    config, mode="multiprocesing", ncores=0, nfiles=-1, batch_opts="",
    output=None,
):
    njobs = ncores

    # setup jobs
    with open(config, 'r') as f:
        cfg = yaml.load(f)

    # group jobs
    files = cfg["files"]
    if nfiles > 0:
        files = files[:nfiles]
    if mode in ["multiprocessing"] or njobs < 0:
        njobs = len(files)

    grouped_files = [list(x) for x in np.array_split(files, njobs)]
    tasks = [
        {"task": df_process, "args": (fs, cfg["query"]), "kwargs": {}}
        for fs in grouped_files
    ]

    if mode=="multiprocessing" and ncores==0:
        results = pysge.local_submit(tasks)
        df = functools.reduce(lambda x, y: df_merge(x, y), results)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=ncores)
        df = functools.reduce(lambda x, y: df_merge(x, y), results)
    elif mode=="sge":
        results = pysge.sge_submit(
            tasks, "zdb", "_ccsp_temp/", options=batch_opts,
            sleep=5, request_resubmission_options=True,
            return_files=True,
        )
        df = df_open_merge(results)
    elif mode=="condor":
        import conpy
        results = conpy.condor_submit(
            "zdb", "_ccsp_temp/", tasks=tasks, options=batch_opts,
            sleep=5, request_resubmission_options=True,
        )
        df = df_open_merge(results)
    else:
        df = pd.DataFrame()

    #print(df)
    if output is not None:
        path, table = output.split(":")
        df.to_hdf(
            path, table, format='table', append=False, complevel=9,
            complib='blosc:lz4hc',
        )
    else:
        return df

