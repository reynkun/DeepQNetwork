import importlib
import inspect


def import_class(name):
    '''
    Get a class from a path string
    '''

    # make class
    if isinstance(name, str):
        mod_env_str, cl_env_str = name.rsplit('.', 1)
        mod_ag = importlib.import_module(mod_env_str)
        env_class = getattr(mod_ag, cl_env_str)
    elif inspect.isclass(name):
        env_class = name
    else:
        raise Exception('invalid env class')

    return env_class
