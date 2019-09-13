import os
import copy
import numpy as np
import pandas as pd
import pysge
import oyaml as yaml
import functools

from zdb.modules.df_process import df_process, df_merge, df_open_merge

def submit_tasks(tasks, mode="multiprocessing", ncores=0, batch_opts=""):
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

def analyse(
    config, mode="multiprocesing", ncores=0, nfiles=-1, batch_opts="",
    output=None, chunksize=500000, merge_opts={},
):
    njobs = ncores

    # setup jobs
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
        "task": df_process,
        "args": (fs, cfg["query"]),
        "kwargs": {"chunksize": chunksize},
    } for fs in grouped_files]
    results = submit_tasks(tasks, mode=mode, ncores=ncores, batch_opts=batch_opts)
    if mode=='multiprocessing':
        df = functools.reduce(lambda x, y: df_merge(x, y), results)
    else:
        # grouped multi-merge
        merge_njobs = merge_opts.get("njobs", 100)
        grouped_merges = [list(x) for x in np.array_split(results, merge_njobs)]
        tasks = [{
            "task": df_open_merge,
            "args": (r,),
            "kwargs": {},
        } for r in grouped_merges]
        merge_mode = merge_opts.get("mode", "multiprocessing")
        merge_ncores = merge_opts.get("ncores", 0)
        if merge_mode=="multiprocessing" and ncores==0:
            semimerged_results = pysge.local_submit(tasks)
            df = functools.reduce(lambda x, y: df_merge(x, y), results)
        elif mode=="multiprocessing":
            semimerged_results = pysge.mp_submit(tasks, ncores=ncores)
            df = functools.reduce(lambda x, y: df_merge(x, y), results)
        elif mode=="sge":
            semimerged_results = pysge.sge_submit(
                tasks, "zdb-merge", "_ccsp_temp",
                options=merge_opts.get("batch_opts", "-q hep.q"),
                sleep=5, request_resubmission_options=True,
                return_files=True,
            )
            df = df_open_merge(semimerged_results)

    if output is not None:
        path, table = output.split(":")
        df.to_hdf(
            path, table, format='table', append=False, complevel=9,
            complib='zlib',
        )
    else:
        return df

def resume_analyse(path, batch_opts="", output=None):
    results = pysge.sge_resume(
        "zdb", path, options=batch_opts, sleep=5,
        request_resubmission_options=True, return_files=True,
    )
    df = df_open_merge(results)

    if output is not None:
        path, table = output.split(":")
        df.to_hdf(
            path, table, format='table', append=False, complevel=9,
            complib='zlib',
        )
    else:
        return df

def multi_analyse(
    configs, mode="multiprocesing", ncores=0, nfiles=-1, batch_opts="",
    outputs=None, chunksize=500000, merge_opts={},
):
    all_tasks, sizes = [], []
    for config in configs:
        njobs = ncores

        # setup jobs
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
            "task": df_process,
            "args": (fs, cfg["query"]),
            "kwargs": {"chunksize": chunksize},
        } for fs in grouped_files]
        all_tasks.extend(tasks)
        if len(sizes)==0:
            sizes.append(len(tasks))
        else:
            sizes.append(len(tasks)+sizes[-1])

    all_results = submit_tasks(all_tasks, mode=mode, ncores=ncores, batch_opts=batch_opts)

    merge_tasks, merge_sizes = [], []
    for start, stop in zip([0]+sizes[:-1], sizes):
        results = all_results[start:stop]

        if mode=='multiprocessing':
            df = functools.reduce(lambda x, y: df_merge(x, y), results)
        else:
            # grouped multi-merge
            merge_njobs = merge_opts.get("ncores", 100)
            grouped_merges = [list(x) for x in np.array_split(results, merge_njobs)]
            tasks = [{
                "task": df_open_merge,
                "args": (r,),
                "kwargs": {},
            } for r in grouped_merges]
            merge_tasks.extend(tasks)
            if len(merge_sizes)==0:
                merge_sizes.append(len(tasks))
            else:
                merge_sizes.append(len(tasks)+merge_sizes[-1])

    all_merge_results = submit_tasks(merge_tasks, **merge_opts)

    ret_val = []
    for output, start, stop in zip(outputs, [0]+merge_sizes[:-1], merge_sizes):
        merge_results = all_merge_results[start:stop]
        df = df_open_merge(merge_results)

        if output is not None:
            path, table = output.split(":")
            df.to_hdf(
                path, table, format='table', append=False, complevel=9,
                complib='zlib',
            )
        else:
            ret_val.append(df)
    return ret_val
