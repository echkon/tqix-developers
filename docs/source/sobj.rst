Sobj class and its methods:
-----------------------------
.. autoclass:: tqix.pis.circuit.sobj

When initializing a circuit, we use ``tqix.pis.circuit.circuit``:

.. autofunction:: tqix.pis.circuit.circuit

It will return an instance of sobj class 

For example, to initialize a circuit with 100 qubits:

>>> from tqix.pis import *
>>> qc = circuit(N=100)

.. autofunction:: tqix.pis.circuit.sobj.init 
.. autofunction:: tqix.pis.circuit.sobj.print_state
.. autofunction:: tqix.pis.circuit.dbx
.. autofunction:: tqix.pis.circuit.dicke_ghz
