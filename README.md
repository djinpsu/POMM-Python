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

Toy models were used to illustrate the method in Lu et al. (J Neurosci, 2025). These should provide a helpful starting point for analyzing your own data.

Useful concepts:

    State vector: S = [0, -1, 'A', 'B', 'C', 'C', 'D', 'E']
    This specificies the symbol associated with each state. 
    0:  start state
    -1: end state
    In this example 'A', 'B', 'D', 'E' are associated with one state each. 'C' is associated with two states. 

    Transition probabilities P: transition probabilties between the states.   

The program uses mutiple cores for speeding up computation. Set 

      nProc = 10

in POMM.py if you have 10 cores on your CPU, for example. 

The inference algorithm was developed by Dezhe Jin, and should be referred to as the Jin algorithm for inferring POMM. Please cite the paper:

    Jiali Lu, Sumithra Surendralal, Kristofer E Bouchard, and Dezhe Z. Jin, 
    "Partially observable Markov models inferred using statistical tests reveal context-dependent syllable transitions in Bengalese finch songs", 
    Journal of Neuroscience, 8, e0522242024 (2025)   
        
    @article{lu2025partially,
      title={Partially observable Markov models inferred using statistical tests reveal context-dependent syllable transitions in Bengalese finch songs},
      author={Lu, Jiali and Surendralal, Sumithra and Bouchard, Kristofer E and Jin, Dezhe Z},
      journal={Journal of Neuroscience},
      year={2025},
      publisher={Society for Neuroscience}
    }         

  
