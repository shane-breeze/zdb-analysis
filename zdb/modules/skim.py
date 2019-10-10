import os
import shutil
import copy
import numpy as np
import pandas as pd
import oyaml as yaml
import pysge
from zdb.modules.df_skim import df_skim

def job(filename, cfg, outname, chunksize=250000):
    switched = False
    if "TMPDIR" in os.environ:
        os.chdir(os.environ["TMPDIR"])
        shutil.copyfile(filename, "tmp.h5")
        inf = "tmp.h5"
        outf = "res.h5"
        switched = True
    else:
        inf = filename
        outf = outname

    result = df_skim(filename, cfg, outf, chunksize=chunksize)
    if switched:
        shutil.copyfile(outf, outname)
    return result

def submit_tasks(tasks, mode, ncores, batch_opts):
    if mode=="multiprocessing" and ncores==0:
        results = pysge.local_submit(tasks)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=ncores)
    elif mode=="sge":
        results = pysge.sge_submit(
            tasks, "zdb", "_ccsp_temp/", options=batch_opts,
            sleep=5, request_resubmission_options=True,
            return_files=True,
        )
    elif mode=="condor":
        import conpy
        results = conpy.condor_submit(
            "zdb", "_ccsp_temp/", tasks=tasks, options=batch_opts,
            sleep=5, request_resubmission_options=True,
        )
    return results

def skim(
    config, mode="multiprocessing", ncores=0, nfiles=-1, batch_opts="",
    output=None, chunksize=250000,
):
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    njobs = ncores

    #setup jobs
    with open(config, 'r') as f:
        cfg = yaml.full_load(f)

    # group jobs
    files = cfg["files"]
    if nfiles > 0:
        files = files[:nfiles]
    if mode in ["multiprocessing"] or njobs < 0:
        njobs = len(files)

    grouped_files = [list(x) for x in np.array_split(files, njobs)]

    tasks = [
        {"task": job, "args": (fs, cfg, output.format(idx)), "kwargs": {"chunksize": chunksize}}
        for idx, fs in enumerate(grouped_files)
    ]
    submit_tasks(tasks, mode, ncores, batch_opts)
    print("Finished!")

def resume_skim(path, batch_opts="", output=None):
    results = pysge.sge_resume("zdb", path, options=batch_opts)
    print("Finished!")

def multi_skim(
    configs, mode='multiprocessing', ncores=0, nfiles=-1, batch_opts="",
    outputs=None, chunksize=250000,
):
    all_tasks = []

    for config, output in zip(configs, outputs):
        outdir = os.path.dirname(output)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        njobs = ncores

        #setup jobs
        with open(config, 'r') as f:
            cfg = yaml.full_load(f)

        # group jobs
        files = cfg["files"]
        if nfiles > 0:
            files = files[:nfiles]
        if mode in ["multiprocessing"] or njobs < 0:
            njobs = len(files)

        grouped_files = [list(x) for x in np.array_split(files, njobs)]

        tasks = [{
            "task": job,
            "args": (fs, copy.deepcopy(cfg), output.format(idx)),
            "kwargs": {"chunksize": chunksize},
        } for idx, fs in enumerate(grouped_files)]
        all_tasks.extend(tasks)

    submit_tasks(all_tasks, mode, ncores, batch_opts)
    print("Finished!")
