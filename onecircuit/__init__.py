"""
>>> OneCircuit: One solution for all quantum circuit needs
________________________________
>>> copyright (c) 2024 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

import os
import sys

from onecircuit._version import version as __version__
import sys

# ------------------------------------------
# Load modules
# ==========================================

from onecircuit._about import *
from onecircuit._info import *
from onecircuit._version import *

# outer loop
from onecircuit.algorithm import *
from onecircuit.ansatz import *
from onecircuit.circuit import *
from onecircuit.measurement import *
from onecircuit.metric import *
from onecircuit.noise import *
from onecircuit.optimizing import *
from onecircuit.util import *

# -----------------------------------------------------------------------------
# Clean name space
#
del os, sys#, numpy, scipy, multiprocessing, distutils
