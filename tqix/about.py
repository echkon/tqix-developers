"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                              x: quantum measurement, quantum metrology, 
                                 quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

__all__ = ['about']

import sys
import os
import platform
import numpy
import scipy
import inspect
import tqix.version
from   tqix.hinfo import hinfo

def about():
    """
    About box for tqix. Gives version numbers for
    QSim, NumPy, SciPy, Cython, and MatPlotLib.
    """
    print("")
    print("tqix: Quantum computor Simulation code")
    print("copyright (c) 2019 and later.")
    print("authors: Binho Le ")
    print("")
    print("tqix Version:       %s" % tqix.__version__)
    print("Numpy Version:      %s" % numpy.__version__)
    print("Scipy Version:      %s" % scipy.__version__)
    try:
        import Cython
        cython_ver = Cython.__version__
    except:
        cython_ver = 'None'
    print("Cython Version:     %s" % cython_ver)
    try:
        import matplotlib
        matplotlib_ver = matplotlib.__version__
    except:
        matplotlib_ver = 'None'
    print("Matplotlib Version: %s" % matplotlib_ver)
    print("Python Version:     %d.%d.%d" % sys.version_info[0:3])
    print("Number of CPUs:     %s" % hinfo()['cpus'])
    print("Platform Info:      %s (%s)" % (platform.system(),
                                           platform.machine()))
    print("")

if __name__ == "__main__":
    about()
