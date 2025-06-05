# tqix
 >>> tqix: a Toolbox for Quantum in X:\
 >>>    X: quantum measurement, quantum metrology, quantum tomography, and others.

# Description
 >>> tqix is an open-source software providing some convenient tools 
     for quantum measurement, quantum metrology, quantum tomography, and others.
>>>In version 2.0.1 we add a library called tqix.pis for large-scale quantum simulation.     

>>> In version 3.0.* we add a library called tqix.vqa for Variational Quantum Algorithms
>>> In 4.1 we remove tqix.vqa

# Structure of the program

    \tqix
        |--- __init__.py # an init file to mark directory
        |--- about.py # describe the program's information
        |--- hinfor.py # provide information about hardware and packages
        |--- version.py # describe the current version of the program
        |--- qobj.py # generate a quantum object, and some convenient tools operate on the object
        |--- qstate # generate various quantum states
        |--- qoper.py # provide some common quantum operators: pauli matrices, lowering, raising,..
        |--- qmeas.py # calculate quantum measurement using both analytical and simulation methods
        |--- qmetro.py # calcualte quantum metrology
        |--- backend.py # two simuation mnethods: monte carlo and comulative distribution function
        |--- qtool.py # contain auxiliary (physical) tools
        |--- quasi_prob.py # generate quasi-probability functions such as Husimi Q function, Wigner function, Husimi Spin function, and Wigner Spin function
        |--- visualize.py # contain code for Husimi and Wigner visualization in 2D, 3D, and 3D spin
        |--- utility.py # contain some common utility (mathematical) tools
        |
        \povm # generating POVM measurement sets
            |--- __init__.py 
            |--- povm.py  # to return measurement sets: Pauli, Stoke, MUB, SIC
            |--- pauli.py # code for Pauli measurement set
            |--- stoke.py # code for Stoke measurement set
            |--- mub.py   # code for Mutually Unbiased Bases (MUB) measurement set
            |--- sic.py   # code for Symmetric Informationally Complete (SIC) measurement set
            |
        \dsm # direct quantum state tomography (Direct State Measurement, DSM)
            |--- __init__.py  
            |--- dsmWeak.py    # code for DSM using Weak measurement
            |--- dsmStrong.py  # code for DSM using Strong measurement
            |--- dsmProb.py, dsmProb_Conf_1.py, dsmProb_Conf_2.py   # code for DSM using Probe-controlled-system measurement
            |--- execute.py    # execute code contained in "dsm" directory
            |--- util.py       # utility code for quantum tomography: trace distance, fidelity.. 
            |See: arXiv:2007.05294(2020), J. Phys. B: At. Mol. Opt. Phys. 53, 115501 (2020), Physics Letters A 383, 289–294 (2019).
            |
        \pis # large-scale quantum simulation library
            |---__init__.py
            |--- circuit.py #create a quantum circuit
            |--- gates.py #define quantum gates
            |--- noise.py #add noise to quantum gates
            |--- optimizers.py #define various optimizers e.g., GD, Adam, QNG,..
            |--- spin_operators.py #define spin operators
            |--- squeeze_param.py #define squeezing_parameters
            |--- util.py #utility code
            |
# License
 >>> copyright (c) 2019 and later\
 >>> authors: Le Bin Ho\
 >>> contributors: Kieu Quang Tuan, Nguyen Tan Viet

# Note for installation
>>> download source code from our website:
https://vqisinfo.wixsite.com/tqix/download

>>> or from github:
https://github.com/echkon/tqix-developers

>>> then run:\
 >>> $pip3 install .

# note for installation:
 >>> download source code and run:
 >>> ```
 >>> $ pip3 install .
 >>> ```
 >>> install from pypi, run:
 >>> ```
 >>> $ pip3 install tqix
 >>> ```
 
 >>> For some reasons, let's try this:
 >>>```
 >>> pip install -r requirements.txt
     pip install . --no-build-isolation
 >>>```

 >>> There may an error when we have new files: let try
 >>>```
 >>>pip install -v .
 >>>```
