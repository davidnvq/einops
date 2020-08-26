__version__ = '0.2.0'


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


__all__ = ['rearrange', 'reduce', 'parse_shape', 'asnumpy', 'EinopsError']
__all__ += ['redim', 'concat', 'transpose']
from .einops import rearrange, reduce, repeat, parse_shape, asnumpy
from .einops import redim, concat, transpose
