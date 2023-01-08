Gate class and its methods:
-----------------------------

.. autoclass:: tqix.pis.gates.Gates
.. autofunction:: tqix.pis.gates.Gates.init

When adding noise to state after applying a gate operation, additional arguments need \\
to be added when calling gate methods: \\
noise (int): noise ratio \\
num_processes (int): number of processes for multiprocessing noise adding. \\
The gate methods can have these additional arguments include: \\
RX,RY,RZ,OAT,TAT,TNT,RX2,RY2,RZ2,R_plus,R_minus,GMS,RN 

.. autofunction:: tqix.pis.gates.Gates.RX
.. autofunction:: tqix.pis.gates.Gates.RY
.. autofunction:: tqix.pis.gates.Gates.RZ
.. autofunction:: tqix.pis.gates.Gates.OAT
.. autofunction:: tqix.pis.gates.Gates.TAT
.. autofunction:: tqix.pis.gates.Gates.TNT
.. autofunction:: tqix.pis.gates.Gates.RX2
.. autofunction:: tqix.pis.gates.Gates.RY2
.. autofunction:: tqix.pis.gates.Gates.RZ2
.. autofunction:: tqix.pis.gates.Gates.R_plus
.. autofunction:: tqix.pis.gates.Gates.R_minus
.. autofunction:: tqix.pis.gates.Gates.GMS
.. autofunction:: tqix.pis.gates.Gates.RN
.. autofunction:: tqix.pis.gates.Gates.check_input_param
.. autofunction:: tqix.pis.gates.Gates.get_N_d_d_dicked
.. autofunction:: tqix.pis.gates.Gates.get_N_d_d_dicked
.. autofunction:: tqix.pis.gates.Gates.get_J
.. autofunction:: tqix.pis.gates.Gates.Jx
.. autofunction:: tqix.pis.gates.Gates.Jy
.. autofunction:: tqix.pis.gates.Gates.Jz
.. autofunction:: tqix.pis.gates.Gates.J_plus
.. autofunction:: tqix.pis.gates.Gates.var
.. autofunction:: tqix.pis.gates.Gates.expval
.. autofunction:: tqix.pis.gates.Gates.gates 
.. autofunction:: tqix.pis.gates.Gates.measure  