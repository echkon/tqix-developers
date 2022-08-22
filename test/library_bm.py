from matplotlib import pyplot as plt
import numpy as np
import pickle

lb_res = pickle.load(open("./lib_benc_data.pickle","rb")) #load time data benchmarked for libraries  
ax = plt.gca() 
ax.plot(range(1,len(lb_res["tqix"])+1),np.log(lb_res["tqix"]),'b-*',label=r'$tqix$')
ax.plot(range(1,len(lb_res["qsun"])+1),np.log(lb_res["qsun"]),'g--^',label=r'$qsun$')
ax.plot(range(1,len(lb_res["qulacs"])+1),np.log(lb_res["qulacs"]),'r--o',label=r'$qulacs$')
ax.plot(range(1,len(lb_res["yao"])+1),np.log(lb_res["yao"]),'c-.',label=r'$yao$')
ax.plot(range(1,len(lb_res["qiskit"])+1),np.log(lb_res["qiskit"]),'m-+',label=r'$qiskit$')
ax.plot(range(1,len(lb_res["pennylane"])+1),np.log(lb_res["pennylane"]),'y--x',label=r'$pennylane$')
ax.plot(range(1,len(lb_res["projectQ"])+1),np.log(lb_res["projectQ"]),'k-1',label=r'$projectQ$')
ax.plot(range(1,len(lb_res["pyquil"])+1),np.log(lb_res["pyquil"]),'c--v',label=r'$pyquil$')
ax.plot(range(1,len(lb_res["cirq"])+1),np.log(lb_res["cirq"]),'g->',label=r'$cirq$')
ax.plot(range(1,len(lb_res["qsim"])+1),np.log(lb_res["qsim"]),'y-3',label=r'$qsim$')
ax.plot(range(1,len(lb_res["quest"])+1),np.log(lb_res["quest"]),'m-s',label=r'$quest$')

ax.set_xlabel("number of qubits")
ax.set_ylabel("time execution (log scale)")
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("./lib_benchmark.eps", bbox_extra_artists=(lgd,), bbox_inches='tight')