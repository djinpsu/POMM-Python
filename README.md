A partially observable Markov model (POMM) can capture context-dependent transitions between symbols in sequences. 
For example, the syllable sequences of Bengalese finches can be described using POMMs 
(Jin & Kozhevnikov, PLoS Comput Biol, 2011; Lu et al., J Neurosci, 2025).
Here is the code used in Lu et al. (J Neurosci, 2025), updated on 3/27/2026. 

A POMM is a Markov transition model defined in a state space. To capture context dependence, 
some symbols must be represented by multiple states. 
This many-to-one mapping from states to symbols is characteristic of POMMs. 
The goal of the code is to determine the minimum number of states required for each symbol.

Major update: 3/27/2026. A more robust method of inferring POMM is introduced. The new method is
    NGramPOMMSearch
    
Major update: 4/21/2026. An important bug fixed. Download all files and compile fresh if used the previous version. 
   
        Default POMM inferenece method
   
        Construct POMM using n-gram model.
        Successively build n-gram transition models, and test for Pbeta significance.
        Then merge states. 
   
        S, P, pv, PBs, PbT = NGramPOMMSearch(osIn, pValue=0.05, Pcut=0.001, stateMergeParam=[1, 0.1, 0.1], nProc=2, nSample = 10000, ngramStart = 1, fnSave=''):

        Inputs: 
        
        osIn    - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
        nProc   - number of processes used for BW
        pValue  - p-value for accepting the POMM using Pc. 
        nSample - number of samples for calculating pValue. 
        ngramStart - starting ngram, 1, MARKOV, 2, second oreder Markov, etc.
        Pcut - ignore transition probabilities below this value
        stateMergeParam - [maxV, minV, step], state merging parameter, max, min, and step stize. 
                                        Values tested from maxV to minV decreasing with stepSize. Stops for the fist pv > pValue. 
                                        The parameter ranges from 1 to 0. As it decreases, the model becomes more complex.
                                        Adjust maxV, minV, step for your sequences. 

        Return: 
        
        S - the final state
        P - the final transition prob
        pv - p-value of the observed seqeunce
        PBs - PBs sampled from the final model
        PbT - Pb of the observed sequences on the final model


To run the code, first compile the C library:

     make

The state transition diagrams are plotted with GraphViz. Make sure to install it from

     https://graphviz.org

Then run 

     python coolingBengaleseFinch.py

This code can be modified for your own data. 

Toy models were used to illustrate the method in Lu et al. (J Neurosci, 2025). To run the toy model, do

     python ToyModels.py 

Note that the POMM inference method is updated on 3/27/2026. The results are bit different from those in the paper. 

Useful concepts:

    State vector: S = [0, -1, 'A', 'B', 'C', 'C', 'D', 'E']
    This specificies the symbol associated with each state. 
    0:  start state
    -1: end state
    In this example 'A', 'B', 'D', 'E' are associated with one state each. 
    'C' is associated with two states. 

    Transition probabilities P: transition probabilties between the states.   

The program uses mutiple cores for speeding up computation. Set 

      nProc = 10

in POMM.py if you have 10 cores on your CPU, for example. 

Another example is coolingBengaleseFinch.py. This file takes sequences in the file 150mA_u_left_tl.annot_observed_sequences.txt and learns POMM. 
The user can dapat this code to use their own sequences. 

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

  
