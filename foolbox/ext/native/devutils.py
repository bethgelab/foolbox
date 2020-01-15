import eagerpy as ep


def unwrap_(*args):
    """Unwraps all EagerPy tensors if they are not already unwrapped"""
    result = tuple(t.tensor if ep.istensor(t) else t for t in args)
    return result[0] if len(args) == 1 else result


def wrap_(*args):
    """Wraps all inputs as EagerPy tensors if they are not already wrapped"""
    result = tuple(ep.astensor(t) for t in args)
    return result[0] if len(args) == 1 else result


def unwrap(*args):
    """Unwraps all EagerPy tensors if they are not already unwrapped
    and returns a function restoring the original format"""
    if len(args) == 0:
        return args
    restore = wrap_ if ep.istensor(args[0]) else unwrap_
    result = unwrap_(*args)
    return (result, restore) if len(args) == 1 else (*result, restore)


def wrap(*args):
    """Wraps all inputs as EagerPy tensors if they are not already wrapped
    and returns a function restoring the original format"""
    if len(args) == 0:
        return args
    restore = wrap_ if ep.istensor(args[0]) else unwrap_
    result = wrap_(*args)
    return (result, restore) if len(args) == 1 else (*result, restore)
