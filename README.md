# tqix
 >>> tqix: a Toolbox for Quantum in X:
 >>>    X: quantum measurement, quantum metrology, quantum tomography, and others.

# description
 >>> tqix is an open-source software providing some convenient tools 
     for quantum measurement, quantum metrology, quantum tomography, and others.

# Structure of the program

    \tqix
        |--- __init__.py # an init file to mark directory
        |--- about.py    # describe the program's information
        |--- infor.py    # provide information about hardware and packages
        |--- version.py  # describe the current version of the program
        |--- qx.py       # generate a quantum object, and some convenient tools operate on the object
        |--- qstate      # generate various quantum states
        |--- qoper.py    # provide some common quantum operators: pauli matrices, lowering, raising,..
        |--- qmeas.py    # calculate quantum measurement using both analytical and simulation methods
        |--- backend.py  # two simuation mnethods: monte carlo and comulative distribution function
        |--- qtool.py    # contain auxiliary (physical) tools
        |--- quasi_prob.py # generate quasi-probability functions such as Husimi Q function, Wigner function, Husimi Spin function, and Wigner Spin function
        |--- visualize.py  # contain code for Husimi and Wigner visualization in 2D, 3D, and 3D spin
        |--- utility.py    # contain some common utitily (mathematical) tools
        |
        \povm # generating POVM measurement sets
           |--- __init__.py 
           |--- povm.py  # to return measurement sets: Pauli, Stoke, MUB, SIC
           |--- pauli.py # code for Pauli measurement set
           |--- stoke.py # code for Stoke measurement set
           |--- mub.py   # code for Mutually Unbiased Bases (MUB) measurement set
           |--- sic.py   # code for Symmetric Informationally Complete (SIC) measurement set
           |
        \dsm # direct quantum state tomogrpaphy (Direct State Measurement, DSM)
           |--- __init__.py  
           |--- dsmWeak.py    # code for DSM using Weak measurement
           |--- dsmStrong.py  # code for DSM using Strong measurement
           |--- dsmProb.py, dsmProb_Conf_1.py, dsmProb_Conf_2.py   # code for DSM using Probe-controlled-system measurement
           |--- execute.py    # execute code contained in "dsm" directory
           |--- util.py       # utility code for quantum tomography: trace distance, fidelity.. 
               
# license
 >>> copyright (c) 2019 and later\
 >>> authors: Le Bin Ho\
 >>> contributors: Kieu Quang Tuan
 >>> all rights reserved.

# note for installation:
>>> download source code from our website:
https://vqisinfo.wixsite.com/tqix/download

>>> or from github:
https://github.com/echkon/tqix-developers

>>> then run:
 >>> $sudo pip3 install .

# Verify the installtion and check the version
>>> from qutip import *
>>> about()
