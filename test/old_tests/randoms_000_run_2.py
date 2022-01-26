from datetime import datetime
import os
#import psutil
import numpy as np
from tqix import *
from tqix.dsm.execute import *

start = datetime.now()
#Parameters
job = 'prob_2'#configuration
N = 3       # number of qubits
m = 0.0
st = 0.0
niter = 100000    #number of iteration
nsamp = 100     #number of samples
theta = 0.5     #coupling
array = np.arange(6,9,1)
name = 'randoms_000_'+job+'.dat'#file name
f = open(name,'w')
f.write("#Varied quantity: Dimension"+'\n')
f.write("#Configuration: "+job+'\n')
f.write("#This program begins running at: "+str(start)+'\n')
f.write("#Output: "+'\n')
f.write("#Dimension   #Trace_distance   #Error_of_Trace_distance   #Fidelity   #Error_of_Fidelity "+'\n'+'\n')
for N in array:
    state = random(2**N)
    model = execute(state,niter,nsamp,theta,m,st)
    result = model.job(job)
    f.write(str(N)+'  ')
    for i in range(len(result)):
        f.write(str(result[i])+'  ')
    f.write('\n')
#mem = psutil.Process(os.getpid())
end = datetime.now()
f.write('\n'+"#This program stops at: "+str(end)+'\n')
f.write("#Time elapsed: "+str(end - start)+'\n')
#f.write("#This program used "+ str(mem.memory_info().rss) + " bytes."+'\n')
f.close()
