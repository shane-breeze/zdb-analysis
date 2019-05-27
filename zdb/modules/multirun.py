def multidraw(*args, **kwargs):
    return [args[0](*arg, **kwargs) for arg in args[1]]
