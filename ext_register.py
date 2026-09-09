from base import Broomstick

# a decorator to extend Broomstick with a new method
def extends_Broomstick(func):
    setattr(Broomstick, func.__name__, func)
    return func

