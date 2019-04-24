from .db_query_to_frame import merge_results

def multirun(*args, **kwargs):
    return (merge_results([arg[0](*arg[1]) for arg in args], kwargs["index"]),)

def multidraw(*args, **kwargs):
    return [args[0](*arg, **kwargs) for arg in args[1]]
