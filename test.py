from tqix import *
import numpy as np
import time 
from tqix.pis import *


N=20
start = time.time()
qc = circuit(N)
qc.RZ(np.pi/3)
qc.RX(np.pi/3)
qc.RZ(np.pi/3)
print(time.time()-start)
