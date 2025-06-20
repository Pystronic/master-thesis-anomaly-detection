import importlib.util

def isColabEnvironment() -> bool:
    return importlib.util.find_spec("google") is not None