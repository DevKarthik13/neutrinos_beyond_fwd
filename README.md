# Neutrino many-body flavor evolution via the full Hamiltonian

## Paper
If using code, or code derived from it, it may be appropriate to cite the following paper:
- [Neutrino many-body flavor evolution: the full Hamiltonian](https://arxiv.org/abs/2404.16690)

## Requirements
The code in this repository requires a resonably up-to-date python3 and the package `numpy`. 

## Listing Hilbert space from a reference state 
The script `grid.py` lists all basis states that are activated by the repeated application of the full Hamiltinian under the exact pair-wise kinetic energy conservation. The momentum modes are as defined in Eq.(67) of the reference above. Two inputs to the sctipt are
- `filename`: output filename. The script will create two files: `filename`\_p.dat in which momentum modes are stored, and `filename`\_s.dat in which basis states (mod flavor degree of freedom) are stored.
- `pmax`: The `zmax` in Eq.(67) of the reference above.

The reference state is set in line 84. You will specify the momentum modes that are initially occupied. Note that each momentum modes can be occupied only up to 2 times, since the code assumes the 2-flavor case.

## Performing time evolution 
The script `kec.py` simulates time evolution of a neutrino system via the full Hamiltinian with the pair-wise kinetic energy conservation. Two inputs to the script are
- `infile`: The `filename` you specified for `grid.py` above.
- `initj`: The basis state index of the initial state at time t=0.

In addition to the inputs, once can set Hamiltonial parameters in lines 11-13 and simulation parameters in lines 18,19.

## Data for arXiv:2404.16690
To reproduce the results for the reference above, use the data files in `pub/arXiv2404_16690/`. Run, for example
```
./kec.py pub/arXiv2404_16690/Nn6 10
```
to simulate the time evolution of 10th basis state (counting from 0) for the Nn=6 case.
