A partially observable Markov model (POMM) can capture context-dependent transitions between symbols in sequences. 
For example, the syllable sequences of Bengalese finches can be described using POMMs (Jin & Kozhevnikov, PLoS Comput Biol, 2011; Lu et al., J Neurosci, 2025).
Here is the code used in Lu et al. (J Neurosci, 2025).

A POMM is a Markov transition model defined in a state space. To capture context dependence, some symbols must be represented by multiple states. 
This many-to-one mapping from states to symbols is characteristic of POMMs. 
The goal of the code is to determine the minimum number of states required for each symbol.

To run the code, first compile the C library:

  make

The state transition diagrams are plotted with GraphViz. Make sure to install it from

  https://graphviz.org

Then run 

  python ToyModels.py

  






