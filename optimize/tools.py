import numpy as np
from os.path import abspath, join, exists


class Writer(object):
    """ Utility for appending values to text files
    """
    def __init__(self, path='./output.stat'):
        self.path = abspath(path)
        try:
            os.mkdir(path)
        except:
            raise IOError

        self.__call__('step_count', 0)

    def __call__(self, filename, val):
        fullfile = join(self.path, filename)
        with open(fullfile, 'a') as f:
            f.write('%e\n' % val)


def loadnpy(filename):
    """Loads numpy binary file."""
    return np.load(filename)


def savenpy(filename, v):
    """Saves numpy binary file."""
    np.save(filename, v)
    os.rename(filename + '.npy', filename)

