# **OneCircuit**

`OneCircuit` is a versatile Python package designed to simplify the creation, manipulation, and visualization of quantum circuits. Built with ease of use in mind, `OneCircuit` allows researchers and developers to quickly set up quantum circuits, experiment with various quantum gates, and visualize the results.

## **Features**
- **Flexible Quantum Circuit Setup**: Easily create and manipulate quantum circuits for various quantum algorithms.
- **Customizable Circuit Properties**: Customize attributes like qubit numbers, gate parameters, and circuit layout.
- **Advanced Visualization**: Automatically generate plots for quantum circuits, allowing for intuitive understanding and debugging.
- **Noise Model Support**: Add noise models to your circuits for realistic simulations.
- **Quantum Gate Integration**: A wide variety of quantum gates supported, with the ability to add custom gates.

## **Installation**

You can install the package using pip:

```bash
pip install onecircuit
```

## **Basic Usage**

```python
from onecircuit import OneCircuit

# Initialize a quantum circuit with 3 qubits
qc = OneCircuit(N=3)

# Add a Hadamard gate to qubit 0
qc.add_gate('H', qubit=0)

# Add a CNOT gate between qubit 0 and qubit 1
qc.add_gate('CNOT', control=0, target=1)

# Visualize the quantum circuit
qc.plot_circuit()

# Run the circuit (simulating or running on a quantum processor)
qc.run()
```

## **Key Attributes**

- **`N`**: Number of qubits in the circuit.
- **`t`**: Time parameter for time-dependent gates.
- **`numt`**: Number of time steps for simulations involving time evolution.
- **`phase`**: Phase shift for quantum gates.
- **`thetas`**: List of rotation angles for parametric gates.
- **`qc`**: The quantum circuit object, storing all gates and qubit information.
- **`time`**: Time-based information for time-optimal control protocols.

## **Plotting Options**

`OneCircuit` comes with customizable plotting options:

- Frame boldness, plot boldness, legend settings.
- Adjustable figure scale for presentation-quality plots.
- Layouts like two plots in a row or two in a column.

```python
# Customize the plot
qc.plot_circuit(frame_bold=True, plot_bold=True, legend=True, scale=1.5)
```

## **Contributing**

We welcome contributions! If you would like to contribute to the development of `OneCircuit`, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Open a pull request describing your changes.

## **License**

`OneCircuit` is open-source software licensed under the MIT License.

---

Feel free to adapt the structure and content based on your project needs!

# Note for installation
>>> download source code from our website:


>>> or from github:


>>> then run:\
 >>> $pip3 install .

# note for installation:
 
 
 >>> For some reasons, let's try this:
 >>>```
 >>> pip install -r requirements.txt
     pip install . --no-build-isolation
 >>>```

