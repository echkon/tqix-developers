"""
>>> OneCircuit: One solution for all quantum circuit needs
________________________________
>>> copyright (c) 2024 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

__all__ = ['about']

import sys
import platform
import numpy
import scipy
import onecircuit._version
from   onecircuit._info import hinfo

def about():
    """
    About box for OneCircuit. Gives version numbers for
    NumPy, SciPy, and MatPlotLib.
    """
    print("")
    print("onecircuit: One solution for all quantum circuit needs")
    print("copyright (c) 2024 and later.")
    print("authors: Binho Le ")
    print("")
    print("OneCircuit Version:       %s" % onecircuit.__version__)
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
