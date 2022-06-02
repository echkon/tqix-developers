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

import os
import sys
import warnings

import tqix.version
from tqix.version import version as __version__
import sys

# ------------------------------------------
# Load modules
# ==========================================
# utility

from tqix.utility import *
from tqix.about import *

# outer loop
from tqix.qstate import *
from tqix.qx import *
from tqix.qmeas import *
from tqix.qoper import *
from tqix.qtool import *
from tqix.backend import *

from tqix.quasi_prob import *
from tqix.visualize import *

# quantum state tomography
from tqix.dsm import *

# povms
from tqix.povm import *

# pis

from tqix.pis import *

# -----------------------------------------------------------------------------
# Clean name space
#
del os, sys#, numpy, scipy, multiprocessing, distutils
