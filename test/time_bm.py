from tqix.pis import *
from tqix import *
import time 
import matplotlib.pyplot as plt

N_max=100 #max number of qubits

#run time benchmark on noiseless qubit system
# noiseless_time_qubits = []
# for N in range(N_max):
#     qc = circuit(N)

#     start = time.time()

#     qc.RX(np.pi/3)
#     qc.RY(np.pi/3)
#     qc.RZ(np.pi/3)

#     noiseless_time = time.time()-start
#     noiseless_time_qubits.append(noiseless_time)

#run time benchmark on noise qubit system with 1 process for simulating noise
# noise_time_qubits = []
# for N in range(1,N_max):
#     qc = circuit(N)
#     print(N)
#     start = time.time()

#     qc.RX(np.pi/3,noise=0.05)
#     qc.RY(np.pi/3,noise=0.05)
#     qc.RZ(np.pi/3,noise=0.05)

#     noise_time = time.time()-start
#     noise_time_qubits.append(noise_time)


#run time benchmark on noise qubit system with multi-processes for simulating noise
# mp_noise_time_qubits = []
# for N in range(N_max):
#     qc = circuit(N,num_process=25)

#     start = time.time()

#     qc.RX(np.pi/3,noise=0.05)
#     qc.RY(np.pi/3,noise=0.05)
#     qc.RZ(np.pi/3,noise=0.05)

#     mp_noise_time = time.time()-start
#     mp_noise_time_qubits.append(mp_noise_time)

# run time benchmark on noise qubit system with gpu for simulating noise
gpu_noise_time_qubits = []
# for N in range(N_max):
qc = circuit(N_max,use_gpu=True)

start = time.time()

qc.RX(np.pi/3,noise=0.05,num_processes=100)
# qc.RY(np.pi/3,noise=0.05)
# qc.RZ(np.pi/3,noise=0.05)

gpu_noise_time = time.time()-start
print(gpu_noise_time)
gpu_noise_time_qubits.append(gpu_noise_time)

# #plot data
# plt.plot(range(len(noise_time_qubits)), noise_time_qubits, label = "noise (#process = 1)")
# plt.plot(range(len(mp_noise_time_qubits)), mp_noise_time_qubits, label = "noise (#process = 25)")
# plt.plot(range(len(gpu_noise_time_qubits)), gpu_noise_time_qubits, label = "noise (GPU: NVIDIA V100)")
# plt.legend()
# plt.show()