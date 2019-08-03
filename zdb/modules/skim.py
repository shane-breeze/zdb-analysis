import numpy as np
import pandas as pd
import oyaml as yaml
import pysge
from zdb.modules.df_skim import df_skim

def skim(
    config, mode="multiprocessing", ncores=0, nfiles=-1, batch_opts="",
    output=None,
):
    njobs = ncores

    #setup jobs
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
        {"task": df_skim, "args": (fs, cfg, output.format(idx)), "kwargs": {}}
        for idx, fs in enumerate(grouped_files)
    ]

    if mode=="multiprocessing" and ncores==0:
        results = pysge.local_submit(tasks)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=ncores)
    elif mode=="sge":
        results = pysge.sge_submit(
            "zdb", "_ccsp_temp/", tasks=tasks, options=batch_opts,
            sleep=5, request_resubmission_options=True,
        )
    elif mode=="condor":
        import conpy
        results = conpy.condor_submit(
            "zdb", "_ccsp_temp/", tasks=tasks, options=batch_opts,
            sleep=5, request_resubmission_options=True,
        )
    print("Finished!")
