'''
 This python library contains functions for deriving POMM from observed seqeunces
 Useful concepts
   state vector: specifies symbols associated with the states.
       The first is the start state 0, and the second is the end state -1, and the results are the other states. 
       for example, S = [0, -1, 'A', 'B', 'A'] 
   transition probability P of the state vectors. 
       This specifes the transition probabilities between the states. 

 Written by Dezhe Jin, Department of Physics, Penn State, dzj2@psu.edu, 9/9/2015, updated 4/14/2022. 
 Updates
    2025-12-10  plotTransitionDiagram(S,P,Pcut=0.01,filenameDot='temp.dot',filenamePDF='temp.pdf',removeUnreachable=False,markedStates=[],labelStates=0)
                    changd output to PDF instead of PS file format. filenamePDF='temp.pdf'
                
                getNumericalSequencesNonRepeat(seqs,syllableLabels)
                    changed return. Now returns numericSeqs, repeatNumSeqs, Syms, Syms2
                    Here Syms is a dictionary converting syms to numerics, and Syms2 converts numerics to syms. 

   
'''

place='MacAir'  # O, office, MC, mac, other group machines, G, MacAir, macbook air
#place='G'  # O, office, MC, mac, other group machines, G, MacAir, macbook air
#place=''

if place == 'MacAir':
    dirCodes = '/Users/dzj2/projects/POMMPython/'       # mac directory for the c library. 
    nProc = 2   
if place == 'MC':
    dirCodes = '/Users/dezhejin/projects/POMMPython/'       # mac directory for the c library. 
    nProc = 8
elif place == 'O':
    dirCodes = '/home/dzj2/projects/POMMPython/'            # Office machine directory for the c library. 
    nProc = 18
elif place == 'G':                                          # hodgkin, neuron 
    dirCodes = '/home/dzj2/projects/POMMPython/'    
    nProc = 40


from subprocess import call
from numpy import *
from numpy.random import rand, seed
import random
from datetime import datetime
import time
from multiprocessing import Pool
import ctypes
import copy
from scipy.stats import chisquare, ks_2samp
import matplotlib.pyplot as plt
# trigger core fonts for PDF backend
plt.rcParams["pdf.use14corefonts"] = True
# trigger core fonts for PS backend
plt.rcParams["ps.useafm"] = True
import pickle
from numpy.random import multinomial
import scipy.stats.distributions as dist
import os
import json
import threading
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd        
from scipy.sparse import csr_matrix


#import rpy2.robjects as robjects   # for Fisher exact test
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()


# load the C program library. 
lib = ctypes.CDLL(dirCodes+'libPOMM.so')

# PARAMETERS
betaTotalVariationDistance = 0.2    # the factor for modifying the sequence completeness adding the total 
                                    # variation distance, to include the effects of transition probability dependent context dependence
                                    # set this to 0, it becomes pure sequence completeness. 
pValue = 0.05                       # p-value for accepting POMM based on the distributiuon of Pb.      
BWRerun = 100                       # number of times Bohm-Welsh alogrith is ran. 
nSamples = 10000                    # number of samples for getting pv from the Pbeta distribution.     
pTolence = 1e-5                 # smallest transition probability.                      
                                    
#print('In POMM, the total variation distance is weighted with the factor betaTotalVariationDistance=',betaTotalVariationDistance)
#print('In POMM, pValue is set to ',pValue)
    
"""

   List of all functions

   NGramPOMMSearch(osIn, nRerun = 100, pValue=0.05, nProc=2,nSample = 10000):
   Construct POMM using n-gram model.
   Successively build n-gram transition models, and test for Pbeta significance.
   Then merge states. Then delete states. 
       osIn - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
       nProc - number of processes used for BW
       pValue - p-value for accepting the POMM using Pc. 
       nSample - number of samples for calculating pValue. 
       nRerun - number of time BW algorith is rerun during the state deletion process
   return: S, P, pv, PBs, PbT, Pc
       S - the final state
       P - the final transition prob
       pv - p-value of the observed seqeunce
       PBs - PBs sampled from the final model
       PbT - Pb of the observed sequences on the final model
       Pc - sequennce completenss

   TreePOMMSearch(osIn, nRerun = 100, pValue=0.05, nProc=2,nSample = 10000)
   Construct POMM using tree model as the starting point.
   Start with the tree model, then merge states. Then delete states. 
       osIn - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
       nProc - number of processes used for BW
       pValue - p-value for accepting the POMM using Pc. 
       nSample - number of samples for calculating pValue. 
       nRerun - number of time BW algorith is rerun during the state deletion process

   return: S, P, pv, PBs, PbT, Pc
       S - the final state
       P - the final transition prob
       pv - p-value of the observed seqeunce
       PBs - PBs sampled from the final model
       PbT - Pb of the observed sequences on the final model
       Pc - sequennce completenss


   MinPOMMGrid(osIn,maxNumStates=50,maxIterBW=1000,nRerunBW=50,maxNumSearch=1000,pValue=pValue,nProc=2,nSample = 10000)
   Search for the minimum POMM describing the seqeunces.
   From the start state, the algorithm searches nearby grid points, and selects the one with maximum likelihood. 
   The grid point is accepted if the POMM at the grid is not rejected by statistical test 
   that samples sets of sequences from the POMM, computes Pb distribution, and the Pb of the observed 
   seuqences is accepted with p > p_value. 
   input parameters:
       osIn - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
       maxNumStates - upper limit of number of states for each symbol. 
       maxIterBW - maximum number of iterations for BW to find P for a POMM,
       nRerunBW - number of reruns for BW to find P for a POMM. 
       maxNumSearch - maximum number of times of search 
       nProc - number of processes used for BW
       nSample - number of samples used for computing pV
       pValue - p-value for accepting the POMM on the grid. 
   return:
       S0 - the final state
       P0 - the final transition prob
       MLT - maximum likelihood on the grids searched and test sets
       pv - p-value of the observed seqeunce
       PBs - PBs sampled from the final model
       PbT - Pb of the observed sequences on the final model
   
   MinPOMMExpansionDeletion(osIn,SIn=[],maxNumStates=50,maxIterBW=1000,nRerunBW=50,maxNumSearch=1000,pValue=pValue,nProc=2,nSample = 10000)
   Search for the minimum POMM describing the sequences.
   First expand the state in the diagonal directions [1,1,1,...], [2,2,2,...],[3,3,...], then delete
   The grid point is accepted if the POMM at the grid is not rejected by statistical test 
   that samples sets of sequences from the POMM, computes Pc distribution, and the Pb of the observed 
   seuqences is accepted with p > p_value. 
   input parameters:
       osIn - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
       SIn - starting state vector, default [] for starting from the Markov Model. 
       maxIterBW - maximum number of iterations for BW to find P for a POMM,
       nRerunBW - number of reruns for BW to find P for a POMM. 
       maxNumSearch - maximum number of times of search 
       nProc - number of processes used for BW
       maxNumStates - maximum number of states for stopping the search. 
       pValue - p-value for accepting the POMM using Pc. 
       nSample - number of samples for calculating pValue. 
   return:
       S - the final state
       P - the final transition prob
       pv - p-value of the observed seqeunce
       PBs - PBs sampled from the final model
       PbT - Pb of the observed sequences on the final model
   
   MinPOMMSimpDeleteStates(S,osIn,nProc = 2, nRerun = 50, pValue=pValue, nSample=10000)
   Simplify by deleting states and making sure that the maximum likelihood remains within bound. 
   Input parameters:
       S - initial POMM
       osIn - observed sequences
       nProc - number of processes for parallel computing in BWPOMMFun
       nRerun - number of times B-W is run with different seeds. 

   Return S, P, pv, PBs, PbT, Pc
       S - state vector
       P - transition probabilities
       pv - p value of the final model
       PBs - sampled distribution of Pb
       PbT - Pb of the observed sequences
       Pc - sequence completeness


   MinPOMMSimp(S,osIn,minP = 0.001,nProc = 2, nRerun = 50, pValue=pValue, nSample=10000, factors=[1.0,0.5,0.2,0.1])
   Simplify by deleting connections and making sure that the maximum likelihood remains within bound. 
   Input parameters:
       S - initial POMM
       osIn - observed sequences
       minP - minimum P for accepting the connection as non-zero. 
       nProc - number of processes for parallel computing in BWPOMMFun
       nRerun - number of times B-W is run with different seeds. 
       pValue - final model needs to clear the pValue. 
       factors - factors for deciding the threshold of cuts. 
   Return S, P, pv, PBs, PbT
       S - state vector
       P - transition probabilities
       pv - pv achieved after cutting. 
       PBs - sampled PBs 
       PbT - of the observed sequences


   getPVSampledSeqsPOMM(S, P, osIn,nSample = 10000, nProc=2)
   get the p-value of the observed seqeunces against the Pb of the sampled sequences for a given POMM
   This method is through generating sequences from POMM
   Inputs:
       S, state vector
       P, transition probabilities
        osIn, observed sequence
        nSample, number of samples, default 10000
        nProc, number of processors used
    Returns:
        pv, p-value of the observed sequence on the model.
        PBs, modified sequence completeness sampled
        PbT, modified sequence completeness of the observed sequences
 
 
    getPVSampledSeqsPOMMnoPbDistr(S, P, osIn, nSample = 10000, nProc=2):
    get p-value of the observed sequences against the Pb of the sampled sequences for a given POMM. 
    To save memory usage, the PBs are not stored. 
    Instead, the number of times sampled Pb is larger that PbT (+1e-10 to break ties) are recorded. 
    These are used to compute pv.
    Inputs:
        S, state vector
        P, transition probabilities
        osIn, observed sequence
        nSample, number of samples, default 10000
        nProc, number of processors used
    Returns:
        pv, p-value of the observed sequence on the model.
        PbT, sequence completeness of the observed sequences. 
 
 
    getUniqueSequences(osIn)
    Input:
        osIn, list of sequences
    Returns:
        osU, unique sequences
        osK, number of times the unique sequences appear
        symU, symbols.
 
 
    BWPOMMCParallel(S,osInO,C=[],maxSteps=5000,pTol=1e-6, nRerun=100, nProc = 2)
    Parallel version of BWPOMM, calling C function BWPOMMC from libPOMM.h   
    Inputs:
        S, state vector
        osInO, observed sequences
        C, connectivity matrix, 1 or 0, those with 0 are cut.
        maxSteps, maximum number of steps for updating the transition probabilities
        pTol, tolerance for the transition probabilities
        nRerun, number of times the algorithm is run. 
        nProc, number of processors used. 
    Returns:
        P, computed transition matrix
        MLmax, maximum log likelihood
        Pc, sequence completeness of the input sequences on the model
        stdML, standard deviation of the maximum likelihood achieved for all runs. 
        MK, list of maximum likelihoods
 
 
    getUniuqeSequencesProbConfidenceIntervals(osK, alpha)
    get the confidence intervals of the probabilities of unique sequences
    Input
        osK - counts of the occurances of unique sequences
        alpha - significance level
    Output
        pL - array, lower bounds of the confidence intervals
        pU - array, upper bounds of the confidence intervals. 
 
 
    computeLogLike(S,P,osU,osK)
    Compute log likelihood of the seqeucens given the POMM. 
    inputs
        S, states
        P, transition matrix
        osU, unique sequences
        osK, counts. 
    returns
        llk, log likelihood.
 
 
    normP(P)
    normalize the transition matrix. Enforce the fact that the first row is the start state, and the second row is the end state. 
 
 
    generateSequencePOMM(S,P,nseq)
    given the state transition matrix, generate the observed seqeunces.
    Assumptionm, S[0], S[1] are the start and the end states. 
    Inputs:
        S, state vector
        P, transition probability
        nseq, number of sequences to be generated
    Output:
        gs, generated sequences
 
    printP(P)
    print the transition matrix in a nice form.
 
    
    getSequenceProbModel(S,P,osIn,osU = [])
    Given the model, compute the probabilities of unique sequences in osIn. 
    Input parameters:
        S, states
        P, transition probabilities
        osIn, observed sequences.
        osU, unique sequences in osIn. If empty, computed. 
    Returns osU, PU
        osU, unique sequences
        PU, probabilities of unique sequences.  
 
    computeSequenceCompleteness(S,P,osIn,osU = [])
    compute the sum of the probabilities of all unique sequences given the state machine.   
    Inputs
        S, states
        P, transition probabilities
        osIn, observed sequences.
        osU, unique sequences in osIn. If empty, computed. 
    Outputs
        Pc, sequence completeness
        Ps, probabilities of the sequences
 
 
    computeSequenceProb(ss, S, P)
    compute the probability of the sequence given the model
    Inputs: 
        ss, sequence
        S, state vector
        P, transition matrix
    Returns
        ps, probability of the sequence. 
 
 
    stateSeq, prob = SeqcomputeMostProbableStateSequence(S,P,seq)
 
    compute the most likely path of the state given the sequence
    Input:
        S - state vector
        P - transition probabilities
        seq - sequence
    Return:
        stateSeq - the most probable state sequence
        prob - probability of the most probable sequence
 
 
    SampleTransitionCounts(P,N)
    This function returns the number of transition sampled with transition probability P.
    The total number of sampling is N.  
    Inputs: 
        P, 1d array, transition probabilities
        N, number of transitions sampled.
    Returns"
        C, 1d array, number of times each choice is selected. 
 
 
    ConstructMarkovModel(osIn,syms,pcut = 0.0)
    This function constructs Markov model 
    Inputs:
        osIn, List of input sequences
        syms, symbols in the seuqencs       
    return 
        P, transition matrix 
        S, state vector
        C, counts of transitions
 
    CreateMarkovModelFanout(nSyms,nFanOut)
    create a Markov model with nSyms, with the fan out from each state maxed to nFanout. 
    the transition probabilities are equal for each transition. 
    Inputs
        nSyms - number of symbols
        nFanOut - maximum number of fan out. the number of fan out can be small if some unreachable states are deleted.
    returns 
        S, state vector 
        P, transition matrix
 
    CreatePOMMFanout(nSyms,nExtra,nFanOut)
    create a POMM with nSyms, with the fan out from each state maxed to nFanout. 
    the transition probabilities are equal for each transition. 
    Inpits:
        nSyms - number of symbols
        nExtra - number of extra states for each symbol
        nFanOut - maximum number of fan out. the number of fan out can be small if some unreachable states are deleted.
    Returns: 
        S, state vector 
        P, transition matrix
 
    
    removeUnreachableStates(S,P)
    remove states that are unreachiable from the start state. Keep the transitions to the end states.   
    Inputs:
        S, P
    Returns:
        S, P
 
 
    deleteTransitionSmallProb(S,P,Pcut = 0.01)
    detete connections with small transition probabilities.     
    Inputs:
        S, state vector
        P, transition matrix
        Pcut, transition probability threshold for cutting. 
 
 
    convertToNumericalSequences(seqsIn,symsIn)
    convert sequences into numerical sequences with syms from 1 - n, where n is the number of symbols.
    Inputs:
        seqsIn, input sequences, array of arrays
        symsIn, symbols in the input sequences, array
    returns:
        seqs, numerical sequences
        syms,numerical syms corresponding to symsIn, basically the numerical order of a symbol in symsIn
 
 
    getNumericalSequencesNonRepeat(seqs,syllableLabels)
    Get non-repeat sequences in numberical form read for analysis from strings. 
    Inputs
        seqs, sequences
        syllableLabels, labels of syllables in the sequences
    returns 
        osIn, numerical sequences generated
        repeatNumSeqs, repeat numbers of each syllable in the sequence. 
        Syms, dictionary converging syms to numerics
        Syms2, dictionary converging numerics to syms  
 
 
    plotTransitionDiagram(S,P,Pcut=0.01,filenameDot='temp.dot',filenamePDF='temp.pdf',removeUnreachable=False,markedStates=[],labelStates=0)
    plot the transition matrix diagram using Graphviz. 
    Inputs:
        S, symbols associated with the states. 
        P, transition matrix. 
        Pcut, do not plot if the transition probability is below Pcut. 
        filenameDot, filenamePDF, filenames for storing the dot file and the PDF file. 
 
 
    plotTwoPOMMsStateCorrespondences(S1,P1,Syms21,S2,P2,Syms22,StateCorres21,filenameDot)
    plot two POMM models in a way such that the corresponding states occupy the same positions. 
    parameteres:
        S1, P1, Syms21 - POMM 1, state vector, transition probabilities, Syms2
        S2, P2, Syms22 - POMM 2
        StateCorres21 - disctionary of state correspondence from POMM 2 to POMM 1. 
        filenameDot - filename of the dot file created. 
 
    plotSequenceCompleteness(PCs,ylimMax=-1,xlimlow=0, width=0.02, ticks = [0,0.5,1])
    plot sequence completeness in a nice way. 
 
 
    randomSelectInd(nind, ntot, excludeInd = -1)
    randomly select nind out of ntot
    excludeInd != -1, exclude this index. 
        
 
    MergeStates(S,P,mergeInds)
    merge states, keep the state vector structure but change the transition probability matrix
    merge state ii to jj. The list is given in mergeInds
    NOTE: merge is order dependent! Do not merge into empty state (1,2), (3,1) would be wrong because 1 is empty after (1,2). 
    keep the connections, recalcuate the transition proabilities. 
    returns updated transition probabilties. 
 
    getStepProbability(osT,nSym,nSteps)
    get the step probability distribution.      
    Inputs:
        osT, sequences, symbols are numerical 1 to nSym
        nSym, number of sylmols 
        nStep, number of steps for computing the probabilities. 
     Return
        PSteps, nStep x (nSym+1) matrix. PSteps[:,0] is the probability of ending at the steps. 
 
 
    MergeStatesRecalculateP(S,P,mergeInds,osT,maxIterBW=1000,nRerunBW=100,nProc=2)
    merge states, keep the state vector structure but change the transition probability matrix
    merge state ii to jj. The list is given in mergeInds
    NOTE: merge is order dependent! Do not merge into empty state (1,2), (3,1) would be wrong because 1 is empty after (1,2). 
    keep the connections, recalcuate the transition proabilities. 
    recalculate the transition probabilities with input sequencnes. 
    returns updated transition probabilties. 
    Inputs:
        S - state vector
        P - transitinn probabilities
        mergeInds - list of pair of indices (ii,jj), merging state ii to state jj. 
        maxIterBW, nRerunBW, nProc, parameters for BW algorithm. 
    Return:
        P2 - transition matrix. 
 
 
    generateSequenceSamples(S,P,N,nSample=nSample,nProc=nProc)
    generegate nSample sets of N sequences from the POMM. 
    Inputs:
        S - state vector
        P - transition matrix
        N - number of sequences in each set.
        nSample - number of sets sampled. 
        nProc - number of processes used. 
    Return:
        osSampled - sampled sets of sequences
 
    computePsStatsInSamples(osTSamples,ss,Ps0,nProc=nProc)  
    compute the Ps of subsequence ss in the sampled seqeunces, return confidence intervals. 
    Inputs:
        osTSamples - sampled sequences
        ss - subsequence
        Ps0 - Ps of the subsequence in the observed set
        nProc - number of process used
    Returns:
        pv - p-value of the observed Ps0 being larger than the smapled. 
        pL - lower bound of Ps in 95% confidence interval
        pS - upper bound of Ps in 95% confidence interval   
        pMedian - median value of the distribution
 
 
    computeNumTasksProc(nTot, nProc = 2)
    blance load on multiple process, returns arrar of number of computations each process should handle. 
    usefule wen nStask is not multiples of nProc
    Inputs:
        nTot - total number of tasks
        nProc - number of processors
    returns
        NS - list of length nProc, number of tasks assinged to each processor
 
    RemoveRareSequences(osIn, pCut = 0.001)
    remove unique sequences with probability smaller than pCut. 
    Inputs:
        osIn - list of sequences
        pCut - sequences with probability smaller that pCut are deleted
    Retuts:
        osOut - list of sequences returned. 
 
    getSequenceCompletenessSampleToSample(osRef, osIn)  
    Compute sequence completeness comparing samples
    Inputs:
        osRef - reference sequences
        osIn  - sequences to be compared
    Returns:
        Pc - sequence completeness
 
    plotSequenceLengthDistribution(seqs,fn='')
    Plot sequence length distribution and save to fn.           
 
    plotProbDistribution(Ps,ylimMax=-1,xlimlow=0, width=0.02, xticks = [0,0.2,0.4,0.6,0.8,1],yticks = [])   
    plot sequence completeness in a nice way. 
 
  AIC = computeAIC(S,P,osU,osK) 
  compute AIC score given the model. 
  parameters:
    S - state vectorr
    P - transition probabilities
    osIn - observed sequences
  return:
    AIC


    S, P, SnumVis = constructNGramPOMM(osIn, ng);

    Constructs ngram POMM model from the sequences osIn.
    Inputs:

        osIn    - observed sesequences
        ng      - length of order of the ngram Markov model. ng=1 is the Markov model, ng=2 is the 2nd order Markov model, etc
    
    Returns:
    
        S       - State vector
        P       - Transition probabilities
        SnumVis - number of times a state visited. 

"""

# compute the most likely path of the state given the sequence
# This is the Viterbi algorithm. 
# Input:
#   S - state vector
#   P - transition probabilities
#   seq - sequence
# Return:
#   stateSeq - the most probable state sequence
#   prob - probability of the most probable sequence
def computeMostProbableStateSequence(S,P,seq):
    sseq = [0]+seq+[-1]
    N = len(S)
    T = len(sseq)
    
    PS = zeros((N,T))                   # probability of the the state sequence at state i time step t. 
    SS = zeros((N,T)).astype('int')     # the state selected for time t for state i. 
    PS[0,0] = 1         # starting from the start state 
    
    # forward pass
    for t in range(1,T):
        for jj in range(N):
            if S[jj] != sseq[t]:    # the state does not have the right symbol at this time point. 
                continue    
            imax = argmax(PS[:,t-1] * P[:,jj])
            SS[jj,t] = imax
            PS[jj,t] = PS[imax,t-1] * P[imax,jj]
        
    prob = PS[1,T-1]                # this is the maximum probability. 
    stateSeq = [SS[1,T-1]]      # this is the last state before the end state
    for t in range(T-2,1,-1):   # trace back.
        imax = stateSeq[0]
        stateSeq = [SS[imax,t]]+stateSeq
    return stateSeq, prob   
    
def testComputeMostProbableStateSequence():
    S = [0,-1,1,2,3,3,4,5]
    N = len(S)
    P = zeros((N,N))
    P[0,2] = 0.5
    P[0,3] = 0.5
    P[2,4] = 1
    P[3,5] = 1
    P[4,6] = 0.6
    P[4,7] = 0.4
    P[5,7] = 0.6
    P[5,6] = 0.4
    P[6,1] = 1
    P[7,1] = 1  
    
    print('S=',S)
    printP(P)

    for itest in range(2):
        if itest == 0:
            seq = [1,3,4]
            esq = [2,4,6]
            epr = 0.5 * 0.6
        else:
            seq = [2,3,5]
            esq = [3,5,7]
            epr = 0.5 * 0.6
        stateSeq, prob = computeMostProbableStateSequence(S,P,seq)
        print(' ')
        print('seq =',seq)
        print('ssq =',stateSeq)
        print('expected state sequence = ',esq)
        print('prb =',prob,' expected=',epr)
        

# compute the probability of a single sequence using C function. 
def computeSeqProbPOMMC(N,S,P,seq,PU,ii):
    S = array(S).astype(int32)
    seq = array([0]+list(seq)+[-1]).astype(int32)
    nSeq = len(seq)
    P = array(P)
    
    lib.computeSeqProbPOMM.argtypes = [
        ctypes.c_long, 
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_long)
    ]
    
    lib.computeSeqProbPOMM.restype = ctypes.c_double
    
    prob = lib.computeSeqProbPOMM(
        ctypes.c_long(N),
        S.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), 
        P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_long(nSeq),
        seq.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    )
    
    PU[ii] = prob
    return prob
    
# test program
def testComputeSeqProbPOMMC():
    
    iModel = 1

    if iModel == 1:
        S0 = [0,-1,'A','B','C','C','A','D','E']
        N = len(S0)
        # convert to numberic symbols. 
        SymsRev = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        P = zeros((N,N))
        P[0,2] = 0.5
        P[0,3] = 0.5
        P[2,4] = 0.8
        P[2,1] = 0.2
        P[3,5] = 0.5
        P[3,6] = 0.5
        P[4,7] = 0.9
        P[4,8] = 0.1        
        P[5,7] = 0.2
        P[5,8] = 0.8
        P[6,8] = 0.5
        P[6,1] = 0.5
        P[7,1] = 1.0
        P[8,1] = 1.0    
    else:
        S0 = [0,-1,'A','B','C','C','D','E']
        N = len(S0)
        # convert to numberic symbols. 
        SymsRev = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        P = zeros((N,N))
        P[0,2] = 0.5
        P[0,3] = 0.5
        P[2,4] = 1.0
        P[3,5] = 1.0
        P[4,6] = 0.8
        P[4,7] = 0.2
        P[5,6] = 0.2
        P[5,7] = 0.8
        P[6,1] = 1.0
        P[7,1] = 1.0    
        
    print(P)
    
    S = [SymsRev[ss] for ss in S0[2:]]
    S = [0,-1] +S
    print(S)

    # generate sequences
    nTot = 90
    nSample = 10000

    seqs = generateSequencePOMM(S0,P,nTot)
    syllableLabels = list(SymsRev.keys())
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqs,syllableLabels)
    
    osU, osK, symU = getUniqueSequences(osIn)
    print(' Computing PU...')
    
    N = len(S)
    PU = zeros(N).astype(float64)

    for seq in osU:
        print('seq=',seq)
        prob = computeSeqProbPOMMC(N,S,P,seq,PU,0)
        print('C code prob=',prob)
        os2, PU2 = getSequenceProbModel(S, P, osIn, [seq])
        print('Python prob=',PU2[0])
        print(' ') 
    
    computeModifiedSequenceCompleteness(S,P,osIn)
    

# compute the modified sequence completeness
def computeModifiedSequenceCompleteness(S,P,osT):
    print(' Getting unique sequences...')
    osU, osK, symU = getUniqueSequences(osT)
    print(' Computing PU...')
    
    N = len(S)
    PU = zeros(len(osU)).astype(float64)
    
    # List to hold the thread objects
    threads = []
    # Creating and starting threads
    for i, seq in enumerate(osU):
        thread = threading.Thread(target=computeSeqProbPOMMC, args=(N,S,P,seq,PU,i))
        threads.append(thread)
        thread.start()
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    #osU, PU = getSequenceProbModel(S, P, osT, osU) 
    
    print(' Done.') 

    
    Pc =  sum(PU)       # this is the sequence completeness
    PP = osK/sum(osK)   # this is the emperical transition probabilities. 
    if Pc < 1e-5:       # extremely bad model. 
        return 0.0  
    PU = PU/Pc          # normalize the transition probabilities of the observed sequences on the model.    
    dd = 0.5 * sum(abs(PU - PP))    
    Pb = (1 - betaTotalVariationDistance) * Pc + betaTotalVariationDistance * (1 - dd) 
    return Pb
    
# get the distributions of modified sequence completeness given the model and number of sequences
# helper function for pool. 
def getModifiedSequenceCompletenessSamplingModel(params):
    S,P,N,nSample = params
    PBs = []
    for ii in range(nSample):
        osT = generateSequencePOMM(S,P,N)
        Pb = computeModifiedSequenceCompleteness(S,P,osT)
        PBs.append(Pb)
    return PBs

# using C function for creating the Pb samples.     
def getModifiedSequenceCompletenessSamplingModelC(params):
    S,P,nSeq,nSample = params
    PBs = zeros(nSample)
    S = array(S).astype(int32)
    N = len(S)
                    
    randSeed = int(rand() * 100000);

    # set parameter types   
    lib.getModifiedSequenceCompletenessSamplingModelC.argtypes = [ctypes.c_long, ctypes.c_long, \
                            ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double), \
                            ctypes.c_long, ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_long]
    lib.getModifiedSequenceCompletenessSamplingModelC.restype = None
    # call the C function.
    
    lib.getModifiedSequenceCompletenessSamplingModelC(ctypes.c_long(nSeq), ctypes.c_long(N),\
                    S.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), \
                    P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    ctypes.c_long(nSample),
                    PBs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    ctypes.c_double(betaTotalVariationDistance), ctypes.c_long(randSeed))
    return list(PBs)
    
                    
def getMaxLenSeqs(osIn):
    maxLenSeqs = 0
    for ss in osIn:
        if len(ss) > maxLenSeqs:
            maxLenSeqs = len(ss)
    return maxLenSeqs
                    
def testGetModifiedSequenceCompletenessSamplingModelC():
    # test with a simple model, the toy model
    # state vector
    iModel = 1  # 1, example POMM; 2, the conceptual POMM
    if iModel == 1:
        S0 = [0,-1,'A','B','C','C','A','D','E']
        N = len(S0)
        # convert to numberic symbols. 
        SymsRev = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        P = zeros((N,N))
        P[0,2] = 0.5
        P[0,3] = 0.5
        P[2,4] = 0.8
        P[2,1] = 0.2
        P[3,5] = 0.5
        P[3,6] = 0.5
        P[4,7] = 0.9
        P[4,8] = 0.1        
        P[5,7] = 0.2
        P[5,8] = 0.8
        P[6,8] = 0.5
        P[6,1] = 0.5
        P[7,1] = 1.0
        P[8,1] = 1.0    
    else:
        S0 = [0,-1,'A','B','C','C','D','E']
        N = len(S0)
        # convert to numberic symbols. 
        SymsRev = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        P = zeros((N,N))
        P[0,2] = 0.5
        P[0,3] = 0.5
        P[2,4] = 1.0
        P[3,5] = 1.0
        P[4,6] = 0.8
        P[4,7] = 0.2
        P[5,6] = 0.2
        P[5,7] = 0.8
        P[6,1] = 1.0
        P[7,1] = 1.0    
        
    print(P)
    
    S = [SymsRev[ss] for ss in S0[2:]]
    S = [0,-1] +S
    print(S)

    # generate sequences
    nTot = 90
    nSample = 10000

    seqs = generateSequencePOMM(S0,P,nTot)
    syllableLabels = list(SymsRev.keys())
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqs,syllableLabels)
    
    #S= [0, -1, 1, 1, 2, 3, 4, 5, 5]
    S = [0, -1, 1,  2, 3, 4, 5]
    print(S)
    P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,nProc=nProc,nRerun=100)
    printP(P)   
    
    nSeq = len(osIn)
    print('nSeq=',nSeq)
    
    t1 = time.time()
    PBs = getModifiedSequenceCompletenessSamplingModelC((S,P,nSeq,nSample))
    t2 = time.time()
    print('C code used ',t2-t1,'sec')

    if 1:   # compare to python code.
        t1 = time.time() 
        PBs2 = getModifiedSequenceCompletenessSamplingModel((S,P,nSeq,nSample))
        t2 = time.time()
        print('Python code used ',t2-t1,'sec')
        
        plt.figure()
        plt.hist(PBs,bins=50)
        plt.hist(PBs2,bins=50, alpha=0.5)
        plt.show()
                    

""" 
pv, PBs, PbT = getPVSampledSeqsPOMM((S, P, osIn)

    get the p-value of the observed seqeunces against the Pb of the sampled sequences for a given POMM
    This method is through generating sequences from POMM

    Inputs:
    S       - state vector
    P       - transition probabilities
    osIn    - observed sequence

    Returns:
        pv  - p-value of the observed sequence on the model.
        PBs - modified sequence completeness sampled
        PbT - sequence completeness of the observed sequences. 
        
"""
def getPVSampledSeqsPOMM(S, P, osIn, nSample = 10000, nProc = 2):
    
    NS = computeNumTasksProc(nSamples,nProc)
    N = len(osIn)
    Params = [[S, P, N, NS[ii]] for ii in range(nProc)]
    
    print(" Sampling...")
    
    pool = Pool(processes = nProc)
    res = pool.map(getModifiedSequenceCompletenessSamplingModelC,Params,chunksize = 1)
    #res = pool.map(getModifiedSequenceCompletenessSamplingModel,Params,chunksize = 1)
    pool.close()
    pool.join()
    
    PBs = [pb for PPBs in res for pb in PPBs]   
    PBs = sort(PBs)

    print(" getting PbT...");
    PbT = computeModifiedSequenceCompleteness(S,P,osIn)
    PbT += 1e-10    # in case all Pc's sampled are tied, we want the target Pc not swamped in numerical noise!
    for jj in range(len(PBs)):
        if PBs[jj] > PbT:
            break
    pv = 1.0 * jj/len(PBs)
    return pv, PBs, PbT 
    
    
# get the p-value of the observed seqeunces on the POMM 
# by generating sequences from the POMM. Compute the distribution of the distances. 
# Then calculate the distance of the observed sequence.
# Use this to compute the p-value.  
# Inputs:
#   S, state vector
#   P, transition probabilities
#   osIn, observed sequence
#   nSample, number of samples, default 10000
#   nProc, number of processors used
# Returns:
#   pv, p-value of the observed sequence on the model.
#   D, the distances of the sampled sequences
#   ddT, the distance of the observed sequences. 
def getPVSampledSeqsPOMMTotalVariationDistance(S, P, osIn,nSample = 10000, nProc=2):
    print('Getting p-value using total variation distance...')
    NS = computeNumTasksProc(nSample,nProc)
    N = len(osIn)
    Params = [[S, P, N, NS[ii]] for ii in range(nProc)]

    pool = Pool(processes = nProc)
    res = pool.map(getTotalVariationDistanceSamplingModel,Params,chunksize = 1)
    pool.close()
    pool.join()
    
    DDs = [dd for DDDs in res for dd in DDDs]   
    DDs = sort(DDs)[::-1]

    ddT = computeTotalVariationDistance(S,P,osIn)       
    ddT -= 1e-10    # in case all dd's sampled are tied, we want the target dd not swamped in numerical noise!
    for jj in range(len(DDs)):
        if DDs[jj] < ddT:
            break
    pv = 1.0 * jj/len(DDs)
    return pv, DDs, ddT 
    
def generateSequencePOMMFun(Params):
    S,P,N,nS = Params
    osS = []
    for isam in range(nS):  
        osT = generateSequencePOMM(S,P,N)
        osS.append(osT)
    return osS  
    
# generegate nSample sets of N sequences from the POMM. 
# Inputs:
#   S - state vector
#   P - transition matrix
#   N - number of sequences in each set.
#   nSample - number of sets sampled. 
#   nProc - number of processes used. 
def generateSequenceSamples(S,P,N,nSample=10000,nProc=2):
    NS = computeNumTasksProc(nSample,nProc)
    Params = [[S, P, N, NS[ii]] for ii in range(nProc)]
    pool = Pool(processes = nProc)
    res = pool.map(generateSequencePOMMFun,Params,chunksize = 1)
    pool.close()
    pool.join()
    
    osSampled = [ss for osS in res for ss in osS]   
    return osSampled    

# find the probability of finding a sub string in observed sequences. 
# assuming all strings are numeric, the start is 0, the end is -1
def ProbFindingSubString(osIn,Subs):
    nFound = 0
    if not isinstance(Subs[0],list):
        Subs = [Subs]

    N = len(osIn)
    for ss in osIn:
        ss = [0] + ss + [-1]
        ss = array(ss)
        
        for subs in Subs:
            flag = 0
            m = len(subs)   
            subs = array(subs)      
            for ii in range(len(ss) - m + 1):
                if sum(abs(ss[ii:ii+m] - subs)) == 0:
                    nFound += 1
                    flag = 1
                    break   # do not count multiple occurance.
            if flag == 1:
                break    
                        
    pp = nFound * 1.0/N
    return pp

def computePsStatsInSamplesFun(Params):
    osSamples,ss = Params
    PPs = []
    for osT in osSamples:
        Ps = ProbFindingSubString(osT,ss)
        PPs.append(Ps)
    return PPs
    
# compute the Ps of subsequence ss in the sampled seqeunces, return confidence intervals. 
# Inputs:
#   osTSamples - sampled sequences
#   ss - subsequence
#   Ps0 - Ps of the subsequence in the observed set
#   nProc - number of process used
# Returns:
#   pv - p-value of the observed Ps0 being larger than the smapled. 
#   pL - lower bound of Ps in 95% confidence interval
#   pS - upper bound of Ps in 95% confidence interval   
def computePsStatsInSamples(osTSamples,ss,Ps0,nProc=2):

    nSamples = len(osTSamples)
    NS = computeNumTasksProc(nSamples,nProc)
    osTS =[[] for ii in range(nProc)]
    ii = 0
    jj = 0
    for ipp in range(nProc):
        jj += NS[ipp] 
        for kk in range(ii,jj):
            osTS[ipp].append(osTSamples[kk])
        ii = jj             
    Params = [[osTS[ii], ss] for ii in range(nProc)]

    pool = Pool(processes = nProc)
    res = pool.map(computePsStatsInSamplesFun,Params,chunksize = 1)
    pool.close()
    pool.join()
    
    PP3 = []
    for PPS in res:
        PP3 += PPS
        
    PP3 = sort(PP3)
    for j3 in range(len(PP3)):
        if PP3[j3] > Ps0+1e-10:
            break
    pv = 1.0 * j3/len(PP3)
    pL = PP3[int(len(PP3)* 0.025)]
    pU = PP3[int(len(PP3)* 0.975)]
    pMedian = median(PP3)
                    
    return pv, pL, pU, pMedian


""" 
    S, P, SnumVis = constructNGramPOMM(osIn, ng);

    Constructs ngram POMM model from the sequences osIn.
    Inputs:

        osIn    - observed sesequences
        ng      - length of order of the ngram Markov model. ng=1 is the Markov model, ng=2 is the 2nd order Markov model, etc
    
    Returns:
    
        S       - State vector
        P       - Transition probabilities
        SnumVis - number of times a state visited. 
    
"""
def constructNGramPOMM(osIn, ng):   
    
    S2 = [0,-1]
    StateVecList = [[0],[-1]]
    StateNumVisits = [0, 0]
    StateIDs = []
    StateTransitionCounts = {}
    idmax = 1
    
    for seq in osIn:
        seq = [0] + seq + [-1]
        ii = 0
        for kk in range(1,len(seq)+1):
            if kk < ng:
                vec = seq[:kk]
            else:
                vec = seq[kk-ng:kk]
            if vec[-1] == -1:
                vec = [-1]
            
            if vec in StateVecList:
                jj = StateVecList.index(vec)
                StateNumVisits[jj] += 1
            else:
                StateVecList.append(vec)
                StateNumVisits.append(1)
                S2.append(vec[-1])
                idmax += 1
                jj = idmax
            if (ii,jj) in StateTransitionCounts.keys():
                StateTransitionCounts[(ii,jj)] += 1
            else:
                StateTransitionCounts[(ii,jj)] = 1
            ii = jj
                
    # construct the transition matrix. 
    P2 = zeros((len(S2),len(S2)))
    for (ii,jj) in StateTransitionCounts.keys():
        P2[ii,jj] = StateTransitionCounts[(ii,jj)]
    P2[0,0] = 0         
    P2 = normP(P2)
            
    # reorder the states. 
    P = zeros((len(S2),len(S2)))
    iids = argsort(S2[2:])
    S = [S2[kk+2] for kk in iids]
    S = [0,-1] + S
    SnumVis = [StateNumVisits[2:][kk] for kk in iids]
    SnumVis = StateNumVisits[:2]+SnumVis
    
    iids = iids + 2
    iids = [0,1] + list(iids)
    for i in range(len(S)):
        for j in range(len(S)):
            ii = iids[i]
            jj = iids[j]
            P[i,j] = P2[ii,jj]
            
    return S, P, SnumVis    
    
        
""" 
    S, P, SnumVis = constructNGramPOMMC(osIn, ng);

    Callinng C function to constructs ngram POMM model from the sequences osIn.
    Inputs:

        osIn    - observed sesequences
        ng      - length of order of the ngram Markov model. ng=1 is the Markov model, ng=2 is the 2nd order Markov model, etc
    
    Returns:
    
        S       - State vector
        P       - Transition probabilities
        SnumVis - number of times a state visited. 
    
"""

def constructNGramPOMMC(osIn, ng):
    
    #concatenate the sequences into a long int array. sequence sequence is flanked by 0...-1
    osInC =[]
    for i in range(len(osIn)):
        osInC.append(0)
        osInC += list(osIn[i])
        osInC.append(-1)
    osInC = array(osInC).astype(int32)
    nSeq = len(osInC)
        
    lib.constructNGramPOMMC.argtypes = [
        ctypes.c_long, 
        ctypes.POINTER(ctypes.c_long),
        ctypes.c_long
    ]

    class ThreeArrays(ctypes.Structure):
        _fields_ = [("N", ctypes.c_int),
                    ("S", ctypes.POINTER(ctypes.c_int)),
                    ("P", ctypes.POINTER(ctypes.c_double)),
                    ("StateNumVis", ctypes.POINTER(ctypes.c_int))
                ]
    
    lib.constructNGramPOMMC.restype = ctypes.POINTER(ThreeArrays)
    
    A = lib.constructNGramPOMMC(
        ctypes.c_long(nSeq), 
        osInC.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        ctypes.c_long(ng)
    )
    
    # transfer the data from C memory to Python memory. 
    N = A.contents.N
    S2 = [A.contents.S[i] for i in range(N)]
    P2 = [A.contents.P[i] for i in range(N * N)]
    P2 = array(P2).reshape(N,N)
    StateNumVisits = [A.contents.StateNumVis[i] for i in range(N)]

    # free memory allocated in the C code.  
    lib.freeThreeArrays.argtypes = [ctypes.POINTER(ThreeArrays)]
    lib.freeThreeArrays(A)
    
    # reorder the states. 
    P = zeros((len(S2),len(S2)))
    iids = argsort(S2[2:])
    S = [S2[kk+2] for kk in iids]
    S = [0,-1] + S
    SnumVis = [StateNumVisits[2:][kk] for kk in iids]
    SnumVis = StateNumVisits[:2]+SnumVis
    iids = iids + 2
    iids = [0,1] + list(iids)
    for i in range(len(S)):
        for j in range(len(S)):
            ii = iids[i]
            jj = iids[j]
            P[i,j] = P2[ii,jj]
        
    print(' S=',S)  
    
    return S, P, SnumVis
    
        
    
def testConstructNGramPOMM():
    
    iModel = 2
    
    if iModel == 1:
        # test with a simple model, the toy model
        # state vector
        SO = [0,-1,'A','B','C','C','A','D','E']
        N = len(SO)
        # convert to numberic symbols. 
        SymsRev = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        PO = zeros((N,N))
        PO[0,2] = 0.5
        PO[0,3] = 0.5
        PO[2,4] = 0.8
        PO[2,1] = 0.2
        PO[3,5] = 0.5
        PO[3,6] = 0.5
        PO[4,7] = 0.9
        PO[4,8] = 0.1       
        PO[5,7] = 0.2
        PO[5,8] = 0.8
        PO[6,8] = 0.5
        PO[6,1] = 0.5
        PO[7,1] = 1.0
        PO[8,1] = 1.0   
    elif iModel == 2:
        # model for sequence generation
        SO = [0,-1,'a','b','c','a','c','d','e','b','c','e','c','d']
        SymsRev = {'a':1,'b':2,'c':3,'d':4,'e':5}
        Syms2 = {1:'a',2:'b',3:'c',4:'d',5:'e'}
        N = len(SO)
        PO = zeros((N,N))
        PO[0,2] = 1
        PO[2,3] = 1
        PO[3,4] = 1
        PO[4,5] = 0.2
        PO[4,8] = 0.8
        PO[5,6] = 1
        PO[6,7] = 1
        PO[7,1] = 1
        PO[8,9] = 1
        PO[9,10] = 1
        PO[10,11] = 1
        PO[11,12] = 1
        PO[12,13] = 1
        PO[13,5] = 0.5
        PO[13,1] = 0.5
        PO = normP(PO)
        sylLabels = list(unique(SO[2:]))
        print(sylLabels)

    # generate sequences
    nTot = 10000
    seqs = generateSequencePOMM(SO,PO,nTot)
    syllableLabels = list(SymsRev.keys())
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqs,syllableLabels)

    # construct ngram model. 
    ng = 3
    
    t1 = time.time()
    S, P, SnumVis = constructNGramPOMMC(osIn, ng)
    t2 = time.time()
    print('C code used ',t2-t1,' sec')
    print('S = ',S)
    S2 = [0,-1]
    for ss in S[2:]:
        S2.append(Syms2[ss]) 
    plotTransitionDiagram(S2,P,Pcut=0.01,filenamePS='test.ConstructNGramPOMMC.ps', \
            removeUnreachable=False,markedStates=[])    
    
    t1 = time.time()    
    S, P, SnumVis = constructNGramPOMM(osIn, ng)
    t2 = time.time()
    print('Python code used ',t2-t1,' sec')
    print('S = ',S)
    S2 = [0,-1]
    for ss in S[2:]:
        S2.append(Syms2[ss]) 
    plotTransitionDiagram(S2,P,Pcut=0.01,filenamePS='test.ConstructNGramPOMM.ps', \
            removeUnreachable=False,markedStates=[])    


    
""" 
    S, P, pv, PBs, PbT, Pc = NGramPOMMSearch(osIn, nRerun = 100, pValue=0.05, nProc=2,nSample = 10000, ngramStart = 1)

    Construct POMM using n-gram model.
    Successively build n-gram transition models, and test for Pbeta significance.
    Then merge states. Then delete states. 

    Inputs: 
    
    osIn    - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
    nProc   - number of processes used for BW
    pValue  - p-value for accepting the POMM using Pc. 
    nSample - number of samples for calculating pValue. 
    nRerun  - number of time BW algorith is rerun during the state deletion process
    ngramStart - starting ngram, 1, MARKOV, 2, second oreder Markov, etc.

    Return: 
    
    S - the final state
    P - the final transition prob
    pv - p-value of the observed seqeunce
    PBs - PBs sampled from the final model
    PbT - Pb of the observed sequences on the final model
    Pc - sequennce completenss

"""

def NGramPOMMSearch(osIn, nRerun = 100, pValue=0.05, nProc=2,nSample = 10000, ngramStart = 1, fnSave=''):
    
    print('Constructing POMM with nGram transition diagram...')
    flag = 0
    maxNG = 200
    for ng in range(ngramStart,maxNG):
        print('\nTesting nGram size ng = ',ng)
        
        # construct nGram POMM
        S, P, SnumVis = constructNGramPOMMC(osIn, ng);
                    
        # test the statsicial signifance. 
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nProc=nProc)
        print(' Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))
            
        print(' S=',S)
        if pv > pValue:
            print(' Accepted pv=',round(pv,3))
            flag = 1
            break
        else:
            print(' Rejected pv=',round(pv,3))
            
        if fnSave != '':
            saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave)
                
        
    if flag == 0:
        print('WARNING: in NGramPOMMSearch: no NGram model accepted up until ng=',ng)
        Pc = 0
        return S, P, pv, PBs, PbT
        
    # find symbols that need only one state: Markov symbols. 
    print('Finding Markov symbols...')
    syms = list(unique(S[2:]))
    
    for sm in syms:
        iid = where(sm == array(S))[0]
        if len(iid) <= 2: # if there are two states no need to do this since this can be tested in the merger test. 
            continue
        print(' ')  
        print(' test merging all states for sym  = ',sm)
        
        N = len(S)
        print(' S=',S)
        ii = iid[0]
        PTest = P.copy()
        SnumVisTest = SnumVis[ii]
                
        for j in range(1,len(iid)):
            jj = iid[j]         
            for kk in range(N):
                if kk == ii or kk == jj:
                    continue
                if SnumVisTest+SnumVis[jj] == 0:
                    PTest[ii,kk] = 0
                    PTest[jj,kk] = 0
                    continue
                PTest[ii,kk] = SnumVisTest*1.0/(SnumVisTest+SnumVis[jj]) * PTest[ii,kk] + \
                               SnumVis[jj]*1.0/(SnumVisTest+SnumVis[jj]) * PTest[jj,kk]
                PTest[jj,kk] = 0
            PTest[jj,1] = 1.0
            for kk in range(N):
                if kk == ii or kk == jj:
                    continue
                PTest[kk,ii] = PTest[kk,ii] + PTest[kk,jj]
                PTest[kk,jj] = 0
            PTest[ii,jj] = 0
            SnumVisTest += SnumVis[jj]
        # test if the merge is good.        
        PTest = normP(PTest)
        
        # check of some states are not reachable. 
        C = zeros((N,N)).astype(int)
        C[where(PTest >= 0.001)] = 1
        # check if the end state can be reached from all other states. 
        ireach = checkEndStateReachability(S,C)
        if ireach == 0: # not all states have rout to the end state, indicating some kind of infinite loop!
            print(' ERROR in NGramPOMMSearch: Not all states lead to the end state. Infinite loop detected. Skip the cut. ')
            exit(1)
        
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, PTest, osIn, nSample = nSample, nProc=nProc)
        print('     Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))
        
        if pv > pValue:
            print('     merge accepted, Markov symbol, pv=',round(pv,3))
            flag = 1
            SnumVis[ii] = SnumVisTest
            P = PTest.copy()
            
            for j in range(len(iid)-1,0,-1):
                jj = iid[j]             
                S = S[:jj] + S[jj+1:]
                P = delete(P,jj,axis=0)
                P = delete(P,jj,axis=1)
                SnumVis = SnumVis[:jj] + SnumVis[jj+1:] 
        else:   
            print('     merge rejected.')
            
        if fnSave != '':
            saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave)
        
    # test merge states.
    syms = list(unique(S[2:]))
    for sm in syms:
        iid = where(sm == array(S))[0]
        if len(iid) <= 1:
            continue
        print('Test merging states for sym  = ',sm)
        iidsToDelete = []
        N = len(S)
        print(' S=',S)
        for i in range(len(iid)):
            ii = iid[i]
            if ii in iidsToDelete:
                continue
            for j in range(i+1,len(iid)):
                jj = iid[j]
                if jj in iidsToDelete:
                    continue
                print(' test merging states (',ii,jj,') sym=',sm)
                PTest = P.copy()
                for kk in range(N):
                    if kk == ii or kk == jj:
                        continue
                    if SnumVis[ii]+SnumVis[jj] == 0:
                        PTest[ii,kk] = 0
                        PTest[jj,kk] = 0
                        continue
                    PTest[ii,kk] = SnumVis[ii]*1.0/(SnumVis[ii]+SnumVis[jj]) * P[ii,kk] + \
                                   SnumVis[jj]*1.0/(SnumVis[ii]+SnumVis[jj]) * P[jj,kk]
                    PTest[jj,kk] = 0
                PTest[jj,1] = 1.0
                for kk in range(N):
                    if kk == ii or kk == jj:
                        continue
                    PTest[kk,ii] = P[kk,ii] + P[kk,jj]
                    PTest[kk,jj] = 0
                PTest[ii,jj] = 0
                PTest = normP(PTest)
                # test if the merge is good. 
                pv, PBs, PbT = getPVSampledSeqsPOMM(S, PTest, osIn, nSample = nSample, nProc=nProc)
                print('     Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))

                if pv > pValue:
                    print('     merge accepted pv=',round(pv,3))
                    flag = 1
                    SnumVis[ii] = SnumVis[ii] + SnumVis[jj] 
                    SnumVis[jj] = 0
                    P = PTest.copy()
                    iidsToDelete.append(jj) 
                else:
                    print('     merge rejected.')
                        
        # delete states
        if len(iidsToDelete) > 0:   
            iidsToDelete = list(sort(iidsToDelete))
            for jj in iidsToDelete[::-1]:
                S = S[:jj] + S[jj+1:]
                P = delete(P,jj,axis=0)
                P = delete(P,jj,axis=1)
                SnumVis = SnumVis[:jj] + SnumVis[jj+1:]
            if fnSave != '':
                saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave)
    
    # test deleting state through grids.
    print('Further simplification with state deletion method...') 
    if fnSave != '':
        fnSave += '.DeleteState.dat'
    S, P, pv, PBs, PbT, Pc = MinPOMMSimpDeleteStates(S,osIn, nProc = nProc, nRerun = nRerun, pValue=pValue, nSample=nSample, fnSave=fnSave)
                    
    # final model. 
    print('Found model S=',S)
    print('pv=',pv) 
    print('Pb=',PbT,' Pc=',Pc)
    
    return S, P, pv, PBs, PbT, Pc
    
# save NGramPOMMSearch intermediate results to file. 
def saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave):
    if fnSave[-5:] != '.json':
        fnSave += '.json'
    res = {}    
    res["S"] = S
    res["ng"] = ng
    res["pv"] = pv
    res["PbT"] = PbT
    res["P"] = P.tolist()
    res["SnumVis"] = SnumVis        
    # save results to file. 
    print('Saving NGramPOMMSearchRes results to ',fnSave)
    json_object = json.dumps(res, indent=4)
    with open(fnSave, "w") as outfile:
        outfile.write(json_object)
        
# read the results.
def readNGramPOMMSearchRes(fnSave):
    with open(fnSave,"r") as fpt:
        res = json.load(fpt)
    
    S = res["S"]
    ng = res["ng"]
    pv = res["pv"]
    PbT = res["PbT"]
    P = array(res["P"],dtype="float64")
    SnumVis = res["SnumVis"]        
    
    return S, P, ng, pv, PbT, SnumVis   
    
    
"""
 compute AIC score given the model. 
 parameters:
    S - state vectorr
    P - transition probabilities
    osIn - observed sequences
 return:
    AIC
"""
def computeAIC(S,P,osU,osK):
    
    N = len(S)
    K = (N-2)*(N-2) + 2 * (N-2) - N     # number of non-zero elements in the transition matrix. Take into account the transition prorbabilities normalize.   

    # compute the log maximum likelihood of all sequences. 
    ml = computeLogLike(S,P,osU,osK)
    
    # AIC
    AIC = 2*K - 2*ml
    return AIC

# Construct POMM using tree model as the starting point.
# Start with the tree model, then merge states. Then delete states. 
#   osIn - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
#   nProc - number of processes used for BW
#   pValue - p-value for accepting the POMM using Pc. 
#   nSample - number of samples for calculating pValue. 
#   nRerun - number of time BW algorith is rerun during the state deletion process
#
# return: S, P, pv, PBs, PbT, Pc
#   S - the final state
#   P - the final transition prob
#   pv - p-value of the observed seqeunce
#   PBs - PBs sampled from the final model
#   PbT - Pb of the observed sequences on the final model
#   Pc - sequennce completenss
def TreePOMMSearch(osIn, nRerun = 100, pValue=0.05, nProc=2,nSample = 10000):

    print('Constructing POMM with tree as the starting point...')

    StateNumVisits = [0, 0]
    StateIDs = [0,1]
    StateNextJJ = [[],[]]   # state IDs 
    S2 = [0, -1]            # state vector.     
    StateTransitionCounts = {}
    idmax = 1
    
    for seq in osIn:
        seq = [0] + seq + [-1]
        ii = 0
        StateNumVisits[0] += 1
        
        for kk in range(1,len(seq)):            
            sym = seq[kk]
            flag = 0
            for jj in StateNextJJ[ii]:
                if S2[jj] == sym:
                    flag = 1
                    StateNumVisits[jj] += 1
                    break
            if flag == 0: # new branch, add new state
                if sym == -1:   # this is the end state
                    jj = 1
                    StateNumVisits[1] += 1
                else:   
                    idmax += 1
                    jj = idmax
                    S2.append(sym)
                    StateNextJJ.append([])
                    StateNumVisits.append(1)                    
                StateNextJJ[ii].append(jj)

            if (ii,jj) in StateTransitionCounts.keys():
                StateTransitionCounts[(ii,jj)] += 1
            else:
                StateTransitionCounts[(ii,jj)] = 1
            ii = jj
                    
    # construct the transition matrix. 
    P2 = zeros((len(S2),len(S2)))
    for (ii,jj) in StateTransitionCounts.keys():
        P2[ii,jj] = StateTransitionCounts[(ii,jj)]
    P2[0,0] = 0         
    P2 = normP(P2)
            
    # reorder the states. 
    P = zeros((len(S2),len(S2)))
    iids = argsort(S2[2:])
    S = [S2[kk+2] for kk in iids]
    S = [0,-1] + S
    SnumVis = [StateNumVisits[2:][kk] for kk in iids]
    SnumVis = StateNumVisits[:2]+SnumVis
    
    iids = iids + 2
    iids = [0,1] + list(iids)
    for i in range(len(S)):
        for j in range(len(S)):
            ii = iids[i]
            jj = iids[j]
            P[i,j] = P2[ii,jj]
            
    print('Tree S=',S)      
        
    # find symbols that need only one state: Markov symbols. 
    print('Finding Markov symbols...')
    syms = list(unique(S[2:]))
    for sm in syms:
        iid = where(sm == array(S))[0]
        if len(iid) <= 2: # if there are two states no need to do this since this can be tested in the merger test. 
            continue
        print(' test merging all states for sym  = ',sm)
        
        N = len(S)
        print(' S=',S)
        ii = iid[0]
        PTest = P.copy()
        SnumVisTest = SnumVis[ii]
                
        for j in range(1,len(iid)):
            jj = iid[j]         
            for kk in range(N):
                if kk == ii or kk == jj:
                    continue
                if SnumVisTest+SnumVis[jj] == 0:
                    PTest[ii,kk] = 0
                    PTest[jj,kk] = 0
                    continue
                PTest[ii,kk] = SnumVisTest*1.0/(SnumVisTest+SnumVis[jj]) * PTest[ii,kk] + \
                               SnumVis[jj]*1.0/(SnumVisTest+SnumVis[jj]) * PTest[jj,kk]
                PTest[jj,kk] = 0
            PTest[jj,1] = 1.0
            for kk in range(N):
                if kk == ii or kk == jj:
                    continue
                PTest[kk,ii] = PTest[kk,ii] + PTest[kk,jj]
                PTest[kk,jj] = 0
            PTest[ii,jj] = 0
            SnumVisTest += SnumVis[jj]
        # test if the merge is good.        
        PTest = normP(PTest)
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, PTest, osIn, nSample = nSample, nProc=nProc)
        print('     Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))
        
        if pv > pValue:
            print('     merge accepted, Markov symbol, pv=',round(pv,3))
            flag = 1
            SnumVis[ii] = SnumVisTest
            P = PTest.copy()
            
            for j in range(len(iid)-1,0,-1):
                jj = iid[j]             
                S = S[:jj] + S[jj+1:]
                P = delete(P,jj,axis=0)
                P = delete(P,jj,axis=1)
                SnumVis = SnumVis[:jj] + SnumVis[jj+1:] 
        else:   
            print('     merge rejected.')
    
    # test merge states.
    syms = list(unique(S[2:]))
    for sm in syms:
        iid = where(sm == array(S))[0]
        if len(iid) <= 1:
            continue
        print('Test merging states for sym  = ',sm)
        iidsToDelete = []
        N = len(S)
        print(' S=',S)
        for i in range(len(iid)):
            ii = iid[i]
            if ii in iidsToDelete:
                continue
            for j in range(i+1,len(iid)):
                jj = iid[j]
                if jj in iidsToDelete:
                    continue
                print(' test merging states (',ii,jj,') sym=',sm)
                PTest = P.copy()
                for kk in range(N):
                    if kk == ii or kk == jj:
                        continue
                    if SnumVis[ii]+SnumVis[jj] == 0:
                        PTest[ii,kk] = 0
                        PTest[jj,kk] = 0
                        continue
                    PTest[ii,kk] = SnumVis[ii]*1.0/(SnumVis[ii]+SnumVis[jj]) * P[ii,kk] + \
                                   SnumVis[jj]*1.0/(SnumVis[ii]+SnumVis[jj]) * P[jj,kk]
                    PTest[jj,kk] = 0
                PTest[jj,1] = 1.0
                for kk in range(N):
                    if kk == ii or kk == jj:
                        continue
                    PTest[kk,ii] = P[kk,ii] + P[kk,jj]
                    PTest[kk,jj] = 0
                PTest[ii,jj] = 0
                PTest = normP(PTest)
                # test if the merge is good. 
                pv, PBs, PbT = getPVSampledSeqsPOMM(S, PTest, osIn, nSample = nSample, nProc=nProc)
                print('     Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))
                if pv > pValue:
                    print('     merge accepted pv=',round(pv,3))
                    flag = 1
                    SnumVis[ii] = SnumVis[ii] + SnumVis[jj] 
                    SnumVis[jj] = 0
                    P = PTest.copy()
                    iidsToDelete.append(jj) 
                else:
                    print('     merge rejected.')
        # delete states
        if len(iidsToDelete) > 0:   
            iidsToDelete = list(sort(iidsToDelete))
            for jj in iidsToDelete[::-1]:
                S = S[:jj] + S[jj+1:]
                P = delete(P,jj,axis=0)
                P = delete(P,jj,axis=1)
                SnumVis = SnumVis[:jj] + SnumVis[jj+1:]

    # test deleting state through grids.
    print('Further simplification with state deletion method...') 
    S, P, pv, PBs, PbT, Pc = MinPOMMSimpDeleteStates(S,osIn, nProc = nProc, nRerun = nRerun, pValue=pValue, nSample=nSample)
                    
    # final model. 
    print('Found model S=',S)
    print('pv=',pv) 
    print('Pb=',PbT,' Pc=',Pc)
    
    return S, P, pv, PBs, PbT, Pc
            

def testNGramOrTreePOMMSearch():
    
    iMethod = 1 # 1, NGram, 2, Tree 
    
    # test with a simple model, the toy model
    # state vector
    SO = [0,-1,'A','B','C','C','A','D','E']
    N = len(SO)
    # convert to numberic symbols. 
    SymsRev = {'A':1,'B':2,'C':3,'D':4,'E':5}
    Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
    # state transition probabilities
    PO = zeros((N,N))
    PO[0,2] = 0.5
    PO[0,3] = 0.5
    PO[2,4] = 0.8
    PO[2,1] = 0.2
    PO[3,5] = 0.5
    PO[3,6] = 0.5
    PO[4,7] = 0.9
    PO[4,8] = 0.1       
    PO[5,7] = 0.2
    PO[5,8] = 0.8
    PO[6,8] = 0.5
    PO[6,1] = 0.5
    PO[7,1] = 1.0
    PO[8,1] = 1.0   

    # generate sequences
    nTot = 90
    nSample = 10000
    nRerun = 100
    seqs = generateSequencePOMM(SO,PO,nTot)
    syllableLabels = list(SymsRev.keys())
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqs,syllableLabels)
    
    if iMethod == 1:    
        filenameSave = 'test.NGramPOMMSearch.ps'        
        fnSave = 'test.NGramPOMMSearch.json'
        print('Testing nGram POMM search...\n')
        S, P, pv, PBs, PbT, Pc = NGramPOMMSearch(osIn, nProc=nProc, nRerun = nRerun, pValue=pValue, nSample=nSample, fnSave=fnSave)
    else:
        filenameSave = 'test.TreePOMMSearch.ps'     
        print('Testing tree POMM search...\n')
        S, P, pv, PBs, PbT, Pc = TreePOMMSearch(osIn, nProc=nProc, nRerun = nRerun, pValue=pValue, nSample=nSample)
            
    # simplify by cutting connections       
    S, P, pv, PBs, PbT = MinPOMMSimp(S,osIn,minP = 0,nProc=nProc,nRerun= nRerun, pValue=pValue, nSample=nSample, factors=[0.5])                 
    print('After simplification pv=',pv)
                
    print('S = ',S)
    S2 = [0,-1]
    for ss in S[2:]:
        S2.append(Syms2[ss]) 
    plotTransitionDiagram(S2,P,Pcut=0.01,filenamePS=filenameSave, \
            removeUnreachable=False,markedStates=[])    
                
    
        
# Search for the minimum POMM describing the sequences.
# First expand the state in the diagonal directions [1,1,1,...], [2,2,2,...],[3,3,...], then delete
# The grid point is accepted if the POMM at the grid is not rejected by statistical test 
# that samples sets of sequences from the POMM, computes Pb distribution, and the Pb of the observed 
# seuqences is accepted with p > p_value. 
# input parameters:
#   osIn - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
#   maxIterBW - maximum number of iterations for BW to find P for a POMM,
#   nRerunBW - number of reruns for BW to find P for a POMM. 
#   maxNumSearch - maximum number of times of search 
#   nProc - number of processes used for BW
#   maxNumStates - maximum number of states for stopping the search. 
#   pValue - p-value for accepting the POMM using Pc. 
#   nSample - number of samples for calculating pValue. 
# return:
#   S - the final state
#   P - the final transition prob
#   pv - p-value of the observed seqeunce
#   PBs - PBs sampled from the final model
#   PbT - Pb of the observed sequences on the final model
#   
def MinPOMMExpansionDeletion(osIn,maxNumStates=50,maxIterBW=1000,nRerunBW=50,maxNumSearch=1000,pValue=pValue,nProc=2,nSample = 10000):
    osU,osK,symU = getUniqueSequences(osIn)
    Nsym = len(symU)    
    # expand starting from the Markov model
    for ns in range(1,maxNumStates):
        S = [0,-1]
        for ss in range(1,Nsym+1):
            for ii in range(ns):
                S.append(ss)
    
        print('Testing state vector:')
        print(S)
        # derive the POMM   
        N = len(S)  
        P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,maxSteps=maxIterBW,nProc=nProc,nRerun=nRerunBW)

        # evaluate the selected model. 
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample = nSample, nProc=nProc)
        print('Pb sampled range=(',PBs.min(),PBs.max(),') seq Pb=', PbT,' from BW Pc=',Pc)
        if pv > pValue:
            print('Accepted pv=',pv)
            break
        else:
            print('Rejected pv=',pv)
    # simplify by deleting states       
    S, P, pv, PBs, PbT, Pc = MinPOMMSimpDeleteStates(S,osIn,nProc=nProc,nRerun=nRerunBW,nSample=nSample)    
    return S, P, pv, PBs, PbT 
            
    
# Search for the minimum POMM describing the seqeunces.
# From the start state, the algorithm searches nearby grid points, and selects the one with maximum likelihood. 
# The grid point is accepted if the POMM at the grid is not rejected by statistical test 
# that samples sets of sequences from the POMM, computes Pc distribution, and the Pc of the observed 
# seuqences is accepted with p > p_value. 
# input parameters:
#   osInOrig - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
#   SIn - the state vector to start with. Default is [], which is the Markov model
#   sylsChange - only syllables in this list has number of state changed. 
#   maxIterBW - maximum number of iterations for BW to find P for a POMM,
#   nRerunBW - number of reruns for BW to find P for a POMM. 
#   maxNumSearch - maximum number of times of search 
#   nProc - number of processes used for BW
#   maxNumStates - maximum number of states for stopping the search. 
#   pValue - p-value for accepting the POMM on the grid. 
# return:
#   S0 - the final state
#   P0 - the final transition prob
#   MLT - maximum likelihood on the grids searched and test sets
#   pv - p-value of the observed seqeunce
#   PBs - PBs sampled from the final model
#   PbT - Pb of the observed sequences on the final model
#   
def MinPOMMGrid(osIn,SIn=[],sylsChange=[],maxNumStates=50,maxIterBW=1000,nRerunBW=50,maxNumSearch=1000,pValue=0.05,nProc=2,nSample = 10000):
    # get the starting grid point.  
    osU,osK,symU = getUniqueSequences(osIn)
    Nsym = len(symU)
    
    if len(SIn) == 0:
        print("\nStarting with the Markov model.")      
        P, S, C = ConstructMarkovModel(osIn,list(symU),pcut = 0.0)  
        S0 = S
        P0 = P  
        gridP = ones(Nsym).astype(int)
    else:
        print('\nStarting with state vector ',SIn)
        # derive the POMM       
        N = len(SIn)
        P = normP(rand(N,N))
        P, ml, Pc, stdml, ML = BWPOMMCParallel(SIn,osIn,maxSteps=maxIterBW,nProc=nProc,nRerun=nRerunBW)
        P0 = P.copy()
        S0 = SIn.copy()
        syms = list(unique(S0[2:]))
        gridP = ones(Nsym).astype(int)
        for sym in syms:
            iid = where(array(S0) == sym)[0]
            ii = syms.index(sym)
            gridP[ii] = len(iid)
    
    if len(sylsChange) > 0:
        iidsChange = []
        for ss in sylsChange:
            ii = list(symU).index(ss)
            iidsChange.append(ii)
            
    else:
        iidsChange = [ii for ii in range(Nsym)]
        
    # search the space of the models and see if ml increases.   
    PointsTested = []
    
    # construct a list of vectors for neaby points. Each state is added by 1. 
    vv = zeros(Nsym).astype(int)
    VN = []
    for ip in range(Nsym):
        if ip in iidsChange:
            sp = vv.copy()
            sp[ip] = 1
            VN.append(sp)       
    
    MLT = []
    NS = []         
    print('maxNumStates = ',maxNumStates)
    for isearch in range(maxNumSearch):

        # evaluate the selected model. 
        pv, PBs, PbT = getPVSampledSeqsPOMM(S0, P0, osIn, nSample = nSample, nProc=nProc)
            
        if pv > pValue:
            print('The selected model has pv=',pv)
            print('The minimum POMM found:')
            print('S=',S0)
            #printP(P0)
            break

        # test nearby grid points. 
        NP = []
        for vv in VN:
            ss = gridP + vv
            flag = 0    # see if the grid point has been tested. 
            for tt in PointsTested:
                if sum(abs(ss - tt)) == 0:
                    flag = 1
                    break
            if flag == 1:
                continue
            NP.append(ss)               

        # print the state vector to be tested
        print('State vectors to be tested. ')
        for ss in NP:
            print(ss)           
        PointsTested += NP  
        mlNP = zeros(len(NP))
        SSs = []
        PPs = []
        for ii in range(len(NP)):
            ss = NP[ii]
            # create the states
            Stest = [0,-1]
            for i in range(Nsym):
                for j in range(ss[i]):
                    Stest += [symU[i]]      
            N = len(Stest)

            # derive the POMM       
            P = normP(rand(N,N))
            P, ml, Pc, stdml, ML = BWPOMMCParallel(Stest,osIn,maxSteps=maxIterBW,nProc=nProc,nRerun=nRerunBW)

            mlNP[ii] = ml
            SSs.append(Stest)
            PPs.append(P)
            
        # select the grid point with maximum likelihood of the observed sequences on the model. 
        iid = mlNP.argmax()
        ml1 = mlNP[iid]
                    
        # accept the increase
        S0 = SSs[iid]
        P0 = PPs[iid]
        gridP = NP[iid].copy()
        NS.append(sum(gridP))
        MLT.append(ml1)
        print('Accepting gridP = ',list(gridP))
        print('mlTest=',ml1)
        if NS[-1] >= maxNumStates:
            print('Reached the maximum number of states set to ',maxNumStates)
            break

    return S0, P0, MLT, pv, PBs, PbT 
    
    
    
"""         
 Simplify by deleting states and making sure that the maximum likelihood remains within bound. 
 Input parameters:
    S - initial POMM
    osIn - observed sequences
    nProc - number of processes for parallel computing in BWPOMMFun
    nRerun - number of times B-W is run with different seeds. 
 Return S, P, Pc    
    S - state vector
    P - transition probabilities
    Pc - sequence completeness
""" 
    
def MinPOMMSimpDeleteStates(S,osIn,nProc = 2, nRerun = 50, pValue=pValue, nSample=10000, fnSave=''):
    
    syms = list(unique(S[2:]))

    for sm in syms:
        while (1):
            
            flag = 0

            iid = [ii for ii in range(len(S)) if S[ii] == sm]
            if len(iid) == 1:
                break
                
            kk = iid[0]
            STest = S[:kk]+S[kk+1:] # delete state kk
            print('Test removing state with sym ',S[kk])
            
            P, ml, Pc, stdml, ML = BWPOMMCParallel(STest,osIn,nProc=nProc,nRerun=nRerun)
            
            # evaluate the selected model. 
            pv, PBs, PbT = getPVSampledSeqsPOMM(STest, P, osIn, nProc=nProc, nSample = nSample)
                    
            if pv > pValue:
                S = STest.copy()
                print(' Deletion of a state for sym ',sm,' leads to pv=',pv,' deletion accepted. S=',S)
                flag = 1
            else:
                print(' Rejected deletion.')
            if flag == 0:
                break
                
            if fnSave !='':
                # save results to file. 
                print('Saving MinPOMMSimpDeleteStates results to ',fnSave)
                saveMinPOMMSimpDeleteStates(S, P, pv, PbT, Pc)
                                    
    print('Updated state S = ',S)
    P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,nProc=nProc,nRerun=nRerun)    
    print('After deleting states Pc = ', Pc)
    pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nProc=nProc, nSample = nSample)
        
    print('The final model pv=',pv)
    return S, P, pv, PBs, PbT, Pc   
    
# save NGramPOMMSearch intermediate results to file. 
def saveMinPOMMSimpDeleteStates(S,P,ng,pv,PbT,Pc,fnSave):
    if fnSave[-5:] != '.json':
        fnSave += '.json'
    res = {}    
    res["S"] = S
    res["ng"] = ng
    res["pv"] = pv
    res["PbT"] = PbT
    res["Pc"] = Pc      
    res["P"] = P
    # save results to file. 
    print('Saving MinPOMMSimpDeleteStates results to ',fnSave)
    json_object = json.dumps(res, indent=4)
    with open(fnSave, "w") as outfile:
        outfile.write(json_object)
        
# read the results.
def readMinPOMMSimpDeleteStates(fnSave):
    with open(fnSave,"r") as fpt:
        res = json.load(fpt)
    
    S = res["S"]
    ng = res["ng"]
    pv = res["pv"]
    PbT = res["PbT"]
    Pc = res["Pc"]      
    P = res["P"]
    
    return S, P, ng, pv, PbT, Pc    
    
    
"""
 check if all states lead to the end state. 
 Input:
    S - state vector.
    C - connectivity matrix. 
 Return:
    iReach - 1, all states can be reached; 0, infinite loop detected. 
"""
def checkEndStateReachability(S,C):
    SReached = []
    SFront = [1]
    while 1:
        flag = 0
        SFront2 = []
        for jj in SFront:
            iid = where(C[:,jj] == 1)[0]
            for ii in iid:
                if ii not in SReached:
                    SReached.append(ii)
                    SFront2.append(ii)
                    flag = 1
        if flag == 0:
            break
        SFront = SFront2            
    iReach = 1
    for ii in range(len(S)):
        if ii == 1:
            continue
        if ii not in SReached:
            iReach = 0
            break
    return iReach   
    
    
def testCheckEndStateReachability():
    
    iModel = 2  # 1, a model with all state can be reached; 2, a model some state cannot be reached. 
    
    if iModel == 1:
        S = [0,-1,1,2,3]
        N= len(S)
        C = zeros((N,N))
        C[0,2] = 1
        C[2,3] = 1
        C[2,4] = 1
        C[3,1] = 1
        C[4,3] = 1
    else:
        S = [0,-1,1,2,3,4]
        N = len(S)
        C = zeros((N,N))
        C[0,2] = 1
        C[2,3] = 1
        C[2,4] = 1
        C[3,1] = 1
        C[4,5] = 1
        C[5,4] = 1
    
    iReach = checkEndStateReachability(S,C)
    if iReach == 0:
        print('Inifinite loop detected!')
    else:
        print('All states lead to the end.')
                
                        
    
# Simplify by deleting connections and making sure that the maximum likelihood remains within bound. 
# Input parameters:
#   S - initial POMM
#   osIn - observed sequences
#   minP - minimum P for accepting the connection as non-zero. 
#   nProc - number of processes for parallel computing in BWPOMMFun
#   nRerun - number of times B-W is run with different seeds. 
#   pValue - final model needs to clear the pValue. 
#   factors - factors for deciding the threshold of cuts. 
# Return S, P, Pc   
#   S - state vector
#   P - transition probabilities
#   pv - pv achieved after cutting. 
#   PBs - sampled PBs 
#   PbT - of the observed sequences
def MinPOMMSimp(S,osIn,minP = 0.001,nProc = 2, nRerun = 50, pValue=pValue, nSample=10000, factors=[1.0,0.5,0.2,0.1]):
    
    P0, ml0, Pc, stdml0, ML0 = BWPOMMCParallel(S,osIn,nProc=nProc,nRerun=nRerun)
    N = len(S)
    nSyms = len(unique(S))
    pv = 0.0

    for factor in factors:
        mlTarget = ml0 - stdml0 * factor

        C0 = zeros((N,N)).astype(int)
        C0[where(P0 >= minP)] = 1
        Tested = [] 

        # now try to delete connections.
        while 1:

            flag = 0
            xind,yind = where(C0 == 1)
            for ii in range(len(xind)):
                i = int(xind[ii])
                j = int(yind[ii])
                if i == 1:  # this is the end state. 
                    continue
                if (i,j) in Tested or (j,i) in Tested:
                    continue
                C = C0.copy()
                print('Test removing edge (',i,',',j,') ',)
                C[i,j] = 0
                    
                # check if the end state can be reached from all other states. 
                ireach = checkEndStateReachability(S,C)
                if ireach == 0: # not all states have rout to the end state, indicating some kind of infinite loop!
                    print(' Not all states lead to the end state. Infinite loop detected. Skip the cut. ')
                    continue
                                                        
                P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,C=C,nProc=nProc,nRerun=nRerun)
                
                # test the differenece of the maximum likelihood distributions
                #ksstat, pv = ks_2samp(ML0, ML)
                #if pv > pValue:
                #   print(' Accepted according to the KS test of the ML distributions')
                
                if ml > mlTarget:
                    print(' Accepted according to the ml target')
                    flag = 1
                    C0[i,j] = 0
                    break
                else:
                    print(' Rejected.')
                    Tested.append((i,j))
            if flag == 0:
                break       
        # delete states that are not reached. 
        #Sn = [0,-1]
        #indx = [[0],[1]]
        #indy = [0,1]
        #for i in range(2,N):
        #   if sum(C0[i,:]) > 0:
        #       Sn.append(S[i])
        #       indx.append([i])
        #       indy.append(i)
        #C0 = C0[indx,indy]     
        #S = array(Sn)
        #N = len(S)
        print('Updated state S = ',S)
        P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,C=C0,nProc=nProc,nRerun=nRerun)       
        print('After simplification Pc = ', Pc)
        
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample = nSample, nProc=nProc)
            
        print('After cutting connections pv = ',pv)
        print(' ')
        if pv >= pValue:
            break
            
    if pv < pValue: # do not accept cuts, revert to the original transitionn matrix. 
        print('Could not achieve the pv goal. No cutting. ')
        P = P0  
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample = nSample, nProc=nProc)
                
    return S, P, pv, PBs, PbT

# get unique sequences osU, counts, osK, and unique symbols symU from sequence osIn.
def getUniqueSequences(osIn):
    symU = array([])
    osU = []
    osK = []
    for ss in osIn:
        ss = array(ss)
        flag = 0
        for kk in range(len(osU)):
            su = osU[kk]
            if len(su) == len(ss) and sum(abs(su - ss)) == 0:
                osK[kk] += 1
                flag = 1
                break
        if flag == 1:
            continue
        osU.append(ss)      
        osK.append(1)
        symU = unique(concatenate((symU,ss)))
    symU = symU.astype(int) 
    return osU,osK,symU 
    
# get the confidence intervals of the probabilities of unique sequences
# Input
#   osK - counts of the occurances of unique sequences
#   alpha - significance level
# Output
#   pL - array, lower bounds of the confidence intervals
#   pU - array, upper bounds of the confidence intervals. 
def getUniuqeSequencesProbConfidenceIntervals(osK, alpha):
    N = sum(osK)    # total number of sequence. 
    pL = []
    pU = []
    for k in osK:
        # Clopper-Pearson bound
        p_lower = dist.beta.ppf(alpha/2.0, k, N-k+1) 
        p_upper = dist.beta.ppf(1-alpha/2.0, k+1, N-k)
        # Bayesian bound
        #p_lower = dist.beta.ppf(alpha/2, k+1, N-k+1) 
        #p_upper = dist.beta.ppf(1-alpha/2, k+1, N-k+1)
            
        pL.append(p_lower)
        pU.append(p_upper)
    return pL, pU       
    

def BWPOMMCFun(Params):
    osU, osK, S, P, pTol, maxIter = Params
    N = len(S)
    S = array(S).astype(int32)
    P = array(P)
    osIn =[]
    for i in range(len(osU)):
        osIn.append(0)
        osIn += list(osU[i])
        osIn.append(-1)
    osIn = array(osIn).astype(int32)
    osK = array(osK).astype(int32)
    nU = len(osK)
    randSeed = int(rand() * 100000);
    nSeq = len(osIn)

    t1 = time.time()
    # set parameter types   
    lib.BWPOMMC.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_long), \
                            ctypes.c_long, ctypes.POINTER(ctypes.c_long), 
                            ctypes.c_long,\
                            ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double), \
                            ctypes.c_double, ctypes.c_long, ctypes.c_long]  
    lib.BWPOMMC.restype = ctypes.c_double
    # call the C function.
    ml = lib.BWPOMMC(ctypes.c_long(nSeq), osIn.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), \
                 ctypes.c_long(nU), osK.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), \
                 ctypes.c_long(N),S.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), \
                 P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                 ctypes.c_double(pTol), ctypes.c_long(maxIter), ctypes.c_long(randSeed))
    t2 = time.time()
    print('     BWPOMMC used ',t2-t1,'sec')
    
    #ml2 = computeLogLike(S,P,osU,osK)
    #print('ml C=',ml)
    #print('ml P=',ml2)
                    
    return (ml,P)                   

# multi-thread BWPOMMCMultiThread
def BWPOMMCFunMultiThread(osU, osK, S, P, pTol, maxIter, numThreads):
    N = len(S)
    S = array(S).astype(int32)
    P = array(P)
    osIn =[]
    for i in range(len(osU)):
        osIn.append(0)
        osIn += list(osU[i])
        osIn.append(-1)
    osIn = array(osIn).astype(int32)
    osK = array(osK).astype(int32)
    nU = len(osK)
    randSeed = int(rand() * 100000);
    nSeq = len(osIn)
    
    t1 = time.time()
    # set parameter types   
    lib.BWPOMMCMultiThread.argtypes = [
            ctypes.c_long, 
            ctypes.POINTER(ctypes.c_long), 
            ctypes.c_long, 
            ctypes.POINTER(ctypes.c_long), 
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_long), 
            ctypes.POINTER(ctypes.c_double), 
            ctypes.c_double, 
            ctypes.c_long, 
            ctypes.c_long, 
            ctypes.c_long
    ]   
    lib.BWPOMMCMultiThread.restype = ctypes.c_double
    # call the C function.
    ml = lib.BWPOMMCMultiThread(
            ctypes.c_long(nSeq), 
            osIn.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), 
            ctypes.c_long(nU), 
            osK.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), 
            ctypes.c_long(N),
            S.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), 
            P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ctypes.c_double(pTol), 
            ctypes.c_long(maxIter), 
            ctypes.c_long(randSeed), 
            ctypes.c_long(numThreads)
    )
    t2 = time.time()
    print('     BWPOMMCMultiThread used ',t2-t1,'sec')
        
    if 0:
        print('     Checking log-likelihood returned from C code with Python code...');
        printP(P)
        ml2 = computeLogLike(S,P,osU,osK)
        print('ml C=',ml)
        print('ml P=',ml2)
                            
    return (ml,P)                   


    
# Parallel version of BWPOMM, calling C function BWPOMMC from libPOMM.h 
# Inputs:
#   S, state vector
#   osInO, observed sequences
#   C, connectivity matrix, 1 or 0, those with 0 are cut.
#   maxSteps, maximum number of steps for updating the transition probabilities
#   pTol, tolerance for the transition probabilities
#   nRerun, number of times the algorithm is run. 
#   nProc, number of processors used. 
# Returns:
#   P, computed transition matrix
#   MLmax, maximum log likelihood
#   Pc, sequence completeness of the input sequences on the model
#   stdML, standard deviation of the maximum likelihood achieved for all runs. 
#   MK, list of maximum likelihoods
def BWPOMMCParallel(S,osInO,C=[],maxSteps=5000,pTol=1e-6, nRerun=BWRerun, nProc = nProc):
    osIn = osInO.copy()
    N = len(S)
    S = array(S)
    osU, osK, symU = getUniqueSequences(osIn)
    
    Ps = []
    for irun in range(nRerun):
        if len(C) == 0:
            P = normP(rand(N,N))
        else:
            P = normP(rand(N,N) * C)    
        Ps.append([osU,osK,S,P,pTol,maxSteps])

    # parallel conputation of multiple runs. 
    pool = Pool(processes = nProc)
    res = pool.map(BWPOMMCFun,Ps,chunksize = 1)
    pool.close()
    pool.join()
    
    ML = [ml for (ml, P) in res]
    ML = array(ML)
    iid = ML.argmax()
    P = res[iid][1]
    Pc, Ps = computeSequenceCompleteness(S,P,osIn,osU)
    mlSigma = std(ML)
    mlMax = ML[iid]
    return P, mlMax, Pc, mlSigma, ML    

# Compute log likelihood of the seqeucens given the POMM. 
# S, states
# P, transition matrix
# osU, unique sequences
# osK, counts. 
def computeLogLike(S,P,osU,osK):
    S = array(S)
    N = len(S)
    llk = 0
    for kk in range(len(osU)):
        ss = list(osU[kk])
        os = [0] + ss + [-1]
        T = len(os)             # number of steps. 
        # compute alphas
        A = zeros((N,T))
        A[0,0] = 1.0
        for t in range(1,T):
            iid = where(S == os[t])[0]      # these are the states of allowed transitions. 
            for jj in iid:
                for k in range(N):
                    A[jj,t] += P[k,jj] * A[k,t-1]
        llk += log(A[1,T-1]+1e-100) * osK[kk]
    return llk
                        
# normalize the transition matrix. Enforce the fact that the first row is the start state, and the second row is the end state. 
def normP(P):
    N = P.shape[0]
    for i in range(N):
        if i==1:
            continue
        elif i==0:
            P[i,1] = 0  # no transition to the end state from the start state.  
        P[i,0] = 0  # no transitions to the start state. 
        ss = sum(P[i,:])
        if ss > 0:
            P[i,    :] /= ss
    return P        

# given the state transition matrix, generate the observed seqeunces.
# Assumptionm, S[0], S[1] are the start and the end states. 
def generateSequencePOMM(S,P,nseq):
    N = len(S)
    gs = []
    for istep in range(nseq):
        ids = 0 # start state. 
        ss = []
        while 1:
            # sample the out going transition probabilities. 
            iid = list(multinomial(1,P[ids,:])).index(1)
            if iid == 1: # this is the end state.
                break
            ss.append(S[iid])
            ids = iid    
        gs.append(ss)
    return gs   

# print the transition matrix in a nice form. s
def printP(P):
    print('Transition matrix:')
    N = P.shape[0]
    for i in range(N):
        ff =""
        for j in range(N):
            ff += "%5.2f "
        print(ff % tuple(P[i,:]))

# Compute the probabilities of unique sequences in the dataset given the model. 
# Input parameters:
#   S, states
#   P, transition probabilities
#   osIn, observed sequences.
#   osU, unique sequences in osIn. If empty, computed. 
# Returns osU, PU
#   osU, unique sequences
#   PU, probabilities of unique sequences.  
def getSequenceProbModel(S,P,osIn,osU = []):
    S = array(S)
    N = len(S)
    if len(osU) == 0:
        osU,osK,symU = getUniqueSequences(osIn)
    PU = zeros(len(osU))
    for kk in range(len(osU)):
        ss = array(osU[kk])
        os = [0] + list(ss) + [-1]
        T = len(os)             # number of steps. 
        # compute alphas
        A = zeros((N,T))
        A[0,0] = 1.0
        for t in range(1,T):
            iid = where(S == os[t])[0]      # these are the states of allowed transitions. 
            for jj in iid:
                for k in range(N):
                    A[jj,t] += P[k,jj] * A[k,t-1]
        PU[kk] = A[1,T-1]           
    return osU, PU
    
# compute the sum of the probabilities of all unique sequences given the state machine.     
# Inputs
#   S, states
#   P, transition probabilities
#   osIn, observed sequences.
#   osU, unique sequences in osIn. If empty, computed. 
# Outputs
#   Pc, sequence completeness
#   Ps, probabilities of the sequences
def computeSequenceCompleteness(S,P,osIn,osU = []):
    S = array(S)
    N = len(S)
    if len(osU) == 0:
        osU,osK,symU = getUniqueSequences(osIn)
    Pc = 0
    Ps = []
    for ss in osU:
        ss = array(ss)
        os = [0] + list(ss) + [-1]
        T = len(os)             # number of steps. 
        # compute alphas
        A = zeros((N,T))
        A[0,0] = 1.0
        for t in range(1,T):
            iid = where(S == os[t])[0]      # these are the states of allowed transitions. 
            for jj in iid:
                for k in range(N):
                    A[jj,t] += P[k,jj] * A[k,t-1]
        Pc += A[1,T-1]      # this the probability of observing this unique sequence
        Ps.append(A[1,T-1])
    return Pc, Ps
    
# compute the probability of the sequence given the model
def computeSequenceProb(ss, S, P):
    S = array(S)
    N = len(S)

    ss = array(ss)
    os = [0] + list(ss) + [-1]
    T = len(os)             # number of steps. 
    # compute alphas
    A = zeros((N,T))
    A[0,0] = 1.0
    for t in range(1,T):
        iid = where(S == os[t])[0]      # these are the states of allowed transitions. 
        for jj in iid:
            for k in range(N):
                A[jj,t] += P[k,jj] * A[k,t-1]
    pS = A[1,T-1]                       # this the probability of observing this unique sequence
    return pS       

                        
# This function returns the number of transition sampled with transition probability P.
# The total number of sampling is N.                
def SampleTransitionCounts(P,N):    
        k = len(P)
        S = [P[0]]
        sm = P[0]
        for i in range(1,k):
            sm += P[i]
            S.append(sm)
        S = array(S)
        C = zeros(k)    
        for kk in range(N):
            iid = where(rand() <= S)[0][0]
            C[iid] += 1
        return C

# This function constructs Markov model 
# osIn, List of input sequences
# syms, symbols in the seuqencs     
# return P, C - counts of transitions
def ConstructMarkovModel(osIn,syms,pcut = 0.0):
    N = len(syms)+2
    C = zeros((N,N))    
    # go through the sequences.
    for seq in osIn:
        i = 0
        for s in seq:
            j = syms.index(s)+2
            C[i,j] += 1
            i = j
        C[i,1] += 1
    P = C.copy()    
    P = normP(P)
    C[where(P < pcut)] = 0
    P = normP(C)
    S = [0,-1]+syms
    S, P = removeUnreachableStates(S,P)
    return P, S, C

# create a Markov model with nSyms, with the fan out from each state maxed to nFanout. 
# the transition probabilities are equal for each transition. 
# Parameters:
#   nSyms - number of symbols
#   nFanOut - maximum number of fan out. the number of fan out can be small if some unreachable states are deleted.
#   returns S, P
def CreateMarkovModelFanout(nSyms,nFanOut):
    S = [0, -1]
    for i in range(nSyms):
        S.append(i+1)
    N = len(S)          # number of states. 
    P = zeros((N,N))    # transition probability. 
    # select states to be connected to the start state. 
    iid = randomSelectInd(nFanOut,nSyms)+2
    P[0,iid] = 1./len(iid)*ones(len(iid))
    n0 = len(iid)
    n2 = 0  # transitions to the end state
    for i in range(nSyms):
        # randomly select nFanOutStates, exclude self transitions. 
        iid = randomSelectInd(nFanOut,nSyms+1,excludeInd=i)+2
        for ind in iid:
            if ind == nSyms+2: # end state
                ind = 1
                n2 += 1
            P[i+2,ind] = 1./len(iid)
    # check if there are enough transitions to the end states. 
    while n2 < nFanOut:
        ind = int(rand() * nSyms)
        if ind == 1 or P[ind,1] > 0:
            continue
        iid = where(P[ind,:] > 0)[0]    
        iid2 = iid[int(rand()*len(iid))]
        P[ind,1] = P[ind,iid2]
        P[ind,iid2] = 0
        n2 += 1
    # find states that are not reached, and connect to the start state
    for i in range(nSyms):
        if sum(P[:,i+2]) == 0:
            n0 += 1
            P[0,i+2] = 1.0/n0
                
    for i in range(nSyms):
        if P[0,i+2] > 0:
            P[0,i+2] = 1.0/n0               
    # normalize. 
    P = normP(P)
    # plot
    #plotTransitionDiagram(S,P)
    return S, P

# create a POMM with nSyms, with the fan out from each state maxed to nFanout. 
# the transition probabilities are equal for each transition. 
# Parameters:
#   nSyms - number of symbols
#   nExtra - number of extra states for each symbol
#   nFanOut - maximum number of fan out. the number of fan out can be small if some unreachable states are deleted.
#   returns S, P
def CreatePOMMFanout(nSyms,nExtra,nFanOut):
    S = [0, -1]
    for i in range(nSyms):
        for j in range(nExtra+1):
            S.append(i+1)
    N = len(S)          # number of states. 
    P = zeros((N,N))    # transition probability. 
    # select states to be connected to the start state. 
    iid = randomSelectInd(nFanOut,N-2)+2
    P[0,iid] = 1./len(iid)*ones(len(iid))
    n0 = len(iid)
    n2 = 0  # transitions to the end state
    for i in range(2,N):
        # randomly select nFanOutStates, exclude self transitions. 
        iid = randomSelectInd(nFanOut,N-1,excludeInd=i-2)+2
        for ind in iid:
            if ind == N: # this should be assigned to the end state
                ind = 1
                n2 += 1
            P[i,ind] = 1./len(iid)
    # check if there are enough transitions to the end states. 
    while n2 < nFanOut:
        ind = int(rand() * nSyms)
        if ind == 1 or P[ind,1] > 0:
            continue
        iid = where(P[ind,:] > 0)[0]    
        iid2 = iid[int(rand()*len(iid))]
        P[ind,1] = P[ind,iid2]
        P[ind,iid2] = 0
        n2 += 1
    # find states that are not reached, and then connect to the start state
    for i in range(2,N):
        if sum(P[:,i]) == 0:
            n0 += 1
            P[0,i] = 1.0/n0
    for i in range(2,N):
        if P[0,i] > 0:
            P[0,i] = 1.0/n0             
    # normalize. 
    P = normP(P)
    # plot
    #plotTransitionDiagram(S,P)
    return S, P

    
# remove states that are unreachiable from the start state. Keep the transitions to the end states.     
def removeUnreachableStates(SIn,P):
    #print('Finding and removing unreachable states...')
    epsilon = 1e-10 # small number for judging to be zero. 
    v = zeros(len(SIn))
    v[0] = 1
    iid0 = [0]
    iid = [0]
    while 1:
        v= matmul(v,P)
        iid2 = where(v > epsilon)[0]
        iid = list(unique(iid + list(iid2)))
        v = zeros(len(SIn))
        v[iid] = 1
        if iid == iid0: # reached the end state. 
            break
        iid0 = iid
    S = [SIn[ii] for ii in iid]
    N = len(S)
    P2 = zeros((N,N))
    for i in range(N):
        for j in range(N):
            P2[i,j] = P[iid[i],iid[j]]
    return S,P2
    
# detete connections with small transition probabilities.   
def deleteTransitionSmallProb(SIn,PIn,Pcut = 0.001, iRemoveUnreachableState = 1):
    P = PIn.copy()
    S = SIn.copy()
    Pend = P[:,1].copy()
    P[where(PIn < Pcut)] = 0
    P[:,1] = Pend # preserve transitions to the end state
    P = normP(P)
    if iRemoveUnreachableState == 1:
        S,P = removeUnreachableStates(S,P)
    return S,P      

# convert sequences into numerical sequences with syms from 1 - n, where n is the number of symbols.
# parameters:
#   seqsIn      input sequences, array of arrays
#   symsIn      symbols in the input sequences, array
# returns:
#   seqs        numerical sequences
#   syms        numerical syms corresponding to symsIn, basically the numerical order of a symbol in symsIn
def convertToNumericalSequences(seqsIn,symsIn):
    nsyms = len(symsIn)
    seqs = []
    for sq in seqsIn:
        ssq = []
        for ss in sq:
            ssq.append(symsIn.index(ss)+1)
        seqs.append(ssq)
    syms = range(1,nsyms+1)
    return seqs,syms        

# Get non-repeat sequences in numberical form read for analysis from strings. 
#   Inputs
#       seqs, sequences
#       syllableLabels, labels of syllables in the sequences
#   returns 
#       osIn, numerical sequences generated
#       repeatNumSeqs, repeat numbers of each syllable in the sequence. 
#       Syms, Syms2, dictionary for convergting syms to numerics and vice versa. 
def getNumericalSequencesNonRepeat(seqs,syllableLabels):

    Syms = {}
    Syms2 = {}
    for i in range(len(syllableLabels)):
        sym = syllableLabels[i]
        Syms[sym] = i+1
        Syms2[i+1] = sym

    osIn = []
    repeatNumSeqs = []  # number of times each syllable is repeated. 
    for sq in seqs:
        ss = []
        rs = []
        iid0 = -1
        rn = 1
        for lb in sq:
            iid = Syms[lb]
            if iid != iid0:
                ss += [iid]
                if len(ss) > 1:
                    rs += [rn]
                iid0 = iid
                rn = 1
            else:
                rn += 1 
        rs += [rn]      
        #print(sq)
        #print(ss)
        #print(rs)
        osIn.append(ss)
        repeatNumSeqs.append(rs)
    

    return osIn, repeatNumSeqs, Syms, Syms2     
        
# plot the transition matrix diagram using Graphviz. 
# S, symbols associated with the states. 
# P, transition matrix. 
# Pcut, do not plot if the transition probability is below Pcut. 
# filenameDot, filenamePS, filenames for storing the dot file and the ps file. 
def plotTransitionDiagram(S,P0,Pcut=0.01,filenameDot='temp.dot',filenamePDF='temp.pdf',removeUnreachable=False,markedStates=[],labelStates=0,extraStateMarks={}):
    fp=open(filenameDot,'w')
    fp.write('digraph G {\n')
    fp.write('size="8,8";\n');
    fp.write('rankdir=LR;\n')
    fp.write('ranksep=0.8; \n')
    S,P = deleteTransitionSmallProb(S,P0,Pcut = Pcut)
    N = len(S)
    
    # find duplicate states
    SU = unique(S[2:])
    multipleStates = []
    for ss in SU:
        kk = 0
        for s2 in S[2:]:
            if ss == s2:
                kk += 1
        if kk > 1:  
            multipleStates.append(ss)       
    
    # plot the states
    for i in range(N):      
        if i==1:
            continue    
        sym = S[i]
        if sym in multipleStates:
            cl = 'red'
        else:
            cl = 'black'
        if labelStates == 1:
            symbol = str(i)+':'+str(sym)    
        else:
            symbol= str(sym)
        if len(extraStateMarks) > 0 and i > 1:
            symbol += ':'+extraStateMarks[i]
        if P[i,1] > 0 and (S[i] in markedStates):   # non-Markovian end state
            fp.write('%d [label="%s",style=filled,color="0.9 0.5 0.5",fontcolor=%s];\n' %(i,symbol,cl))
        elif P[i,1] > 0:    # the state transitions to the end state    
            fp.write('%d [label="%s",style=filled,color="0.5 0.5 0.9",fontcolor=%s];\n' % (i,symbol,cl))
        elif i==0: # start state
            fp.write('%d [label="S",style=filled,color="0.9 0.9 0.9",fontcolor=%s];\n' %(i,cl))
        elif S[i] in markedStates:
            fp.write('%d [label="%s",style=filled,color="0.9 0.5 0.9",fontcolor=%s];\n' %(i,symbol,cl))
        else:
            fp.write('%d [label="%s",fontcolor=%s];\n' %(i,symbol,cl))
                

    # get the edge list 
    edgeLists = []
    edgeProb = []
    for i in range(N):
        for j in range(2,N):
            if P[i,j] > 0:
                if P[i,j] >= 0.5:
                    cl='red'
                    pw=2
                elif P[i,j] >=0.1:
                    cl='green'
                    pw=1.5
                else:
                    cl='gray'
                    pw=1
                edgeLists.append([i,j,cl,pw])
                edgeProb.append(P[i,j])
    
                
    # sort the edges
    iid = argsort(edgeProb)
    for i in iid:
        ii,jj,cl,pw = edgeLists[i]
        p = edgeProb[i] 
        fp.write('%d -> %d [arrowhead=normal, arrowsize=%0.2g, penwidth=%0.2g, color="%s"];\n' %(ii,jj,pw,pw,cl))


    fp.write('}\n')
    fp.close()
    
    # save graph files. 
    command = ['dot', '-Tpdf', filenameDot, '-o', filenamePDF]      # this is for linux
    call(command)
        
                
# plot the transition matrix diagram using Graphviz. 
# S, symbols associated with the states. 
# P, transition matrix. 
# Pcut, do not plot if the transition probability is below Pcut. 
# filenameDot, filenamePS, filenames for storing the dot file and the ps file. 
def plotTransitionDiagramOld(S,P,Pcut=0.01,filenameDot='temp.dot',filenamePS='temp.ps',removeUnreachable=False,markedStates=[],labelStates=0):
    fp=open(filenameDot,'w')
    fp.write('digraph G {\n')
    fp.write('size="4,4";\n');
    fp.write('node [height=0.02, width=0.01, fixedsize=true]; \n');
    S,P = deleteTransitionSmallProb(S,P,Pcut = 0.01)
        
    N = len(S)
    for i in range(N):
        if i==1: 
            continue    # do not plot the end state.
        sym = S[i]
        Pout = P[i,:]
        ss = sum(Pout)
        if ss > 0:
            Pout /= ss
        
        for j in range(N):
            if Pout[j] == 0 or j == 1:
                continue
            fp.write('%d -> %d [label=" %.2g", arrowsize=1.5,arrowhead=normal];\n' %(i,j,Pout[j]))

        if labelStates == 1:
            symbol = str(i)+':'+str(sym)    
        else:
            symbol= str(sym)
        if Pout[1] > 0 and (S[i] in markedStates):  # non-Markovian end state
            fp.write('%d [label="%s",style=filled,color="0.9 0.5 0.5"];\n' %(i,symbol))
        elif Pout[1] > 0:   # the state transitions to the end state    
            fp.write('%d [label="%s",style=filled,color="0.5 0.5 0.9"];\n' % (i,symbol))
        elif i==0: # start state
            fp.write('%d [label="S",style=filled,color="0.9 0.9 0.9"];\n' %(i))
        elif S[i] in markedStates:
            fp.write('%d [label="%s",style=filled,color="0.9 0.5 0.9"];\n' %(i,symbol))
        else:
            fp.write('%d [label="%s"];\n' %(i,symbol))

    fp.write('}\n')
    fp.close()
    
    # save graph files. 
    #command = ['open', '-a', 'Graphviz', filenameDot] # this is for Mac
    #call(command)
    command = ['dot', '-Tps', filenameDot, '-o', filenamePS]        # this is for linux
    call(command)

# plot sequence completeness in a nice way. 
def plotSequenceCompleteness(PCs,ylimMax=-1,xlimlow=0, width=0.02, ticks = [0,0.5,1]):
    plt.xlim([-width,1+width])
    cc,bb = histogram(PCs,bins=30)
    plt.bar(bb[:-1],cc,width=width,color='gray')
    plt.axis('off')
    plt.plot([xlimlow,1],[0,0],color='gray')
    if ylimMax != -1:
        plt.ylim([-0.01*ylimMax,ylimMax])
    ylim = plt.ylim()
    for tt in ticks:
        tlen = -ylim[1] * 0.01
        plt.plot([tt,tt],[0,tlen],color='gray')
        plt.text(tt,-0.05*ylim[1],str(tt),horizontalalignment='center')

# plot sequence completeness in a nice way. 
def plotRepeatNumberDistribution(RepeatNums,rrmax=0, ylimMax=-1,width=0.2,syllableLabel=''):
    if rrmax == 0:
        rrmax = max(RepeatNums)
    bbins = [0.5]
    for ii in range(1,rrmax+1):
        bbins.append(ii+0.5)
    plt.xlim([0,rrmax+1])
    cc,bb = histogram(RepeatNums,bins=bbins)
    plt.bar(bb[:-1]+0.5,cc,width=width,color='gray')
    plt.axis('off')
    plt.plot([0,rrmax+1],[0,0],color='gray')
    if ylimMax != -1:
        plt.ylim([-0.01*ylimMax,ylimMax])
    ylim = plt.ylim()
    ticks = range(rrmax+1)
    for tt in ticks:
        tlen = -ylim[1] * 0.01
        plt.plot([tt,tt],[0,tlen],color='gray')
        plt.text(tt,-0.1*ylim[1],str(tt),horizontalalignment='center')
        plt.text(0.5,ylim[1] * 0.8, syllableLabel)


# randomly select nind out of ntot
# excludeInd != -1, exclude this index. 
def randomSelectInd(nind, ntot, excludeInd = -1):
    iid = array([]).astype(int)
    for j in range(nind):
        while 1:
            ind = int(rand() * ntot)
            if ind == excludeInd:
                continue
            if (len(where(iid == ind)[0]) == 0):
                iid = append(iid,ind)
                break
    return iid  

# merge states, keep the state vector structure but change the transition probability matrix
# merge state ii to jj. The list is given in mergeInds
# NOTE: merge is order dependent! Do not merge into empty state (1,2), (3,1) would be wrong because 1 is empty after (1,2). 
# keep the connections, recalcuate the transition proabilities. 
# returns updated transition probabilties. 
def MergeStates(S,P,mergeInds):
    P = normP(P)
    P2 = P.copy()
    for (ii,jj) in mergeInds:
        print('Merging state ',ii,' to ',jj)
        # all ins are merged.
        P2[:,jj] = P2[:,jj]+P2[:,ii]
        # all outs are merged. 
        P2[jj,:] = P2[jj,:]+P2[ii,:]
        # disconnect state ii
        P2[:,ii] = 0
        P2[ii,:] = 0
    P2 = normP(P2)
    return P2

# merge states, keep the state vector structure but change the transition probability matrix
# merge state ii to jj. The list is given in mergeInds
# NOTE: merge is order dependent! Do not merge into empty state (1,2), (3,1) would be wrong because 1 is empty after (1,2). 
# keep the connections, recalcuate the transition proabilities. 
# recalculate the transition probabilities with input sequencnes. 
# returns updated transition probabilties. 
# Inputs:
#   S - state vector
#   P - transitinn probabilities
#   mergeInds - list of pair of indices (ii,jj), merging state ii to state jj. 
# Return:
#   P2 - transition matrix. 
def MergeStatesRecalculateP(S,P,mergeInds,osT,maxIterBW=1000,nRerunBW=100,nProc=2):
    P = normP(P)
    P2 = P.copy()
    for (ii,jj) in mergeInds:
        print('Merging state ',ii,' to ',jj)
        # all ins are merged.
        P2[:,jj] = P2[:,jj]+P2[:,ii]
        # all outs are merged. 
        P2[jj,:] = P2[jj,:]+P2[ii,:]
        # disconnect state ii
        P2[:,ii] = 0
        P2[ii,:] = 0
    P2 = normP(P2)
    C = zeros(P2.shape)
    C[where(P2 > 0.001)] = 1    # only allow connections with P > 0.001
    print('Recalculating the transition probabilities...')
    P2, ml, Pc, stdml, ML = BWPOMMCParallel(S,osT,C=C,maxSteps=maxIterBW,nProc=nProc,nRerun=nRerunBW)
    return P2

# remove unique sequences with probability smaller than pCut. 
def RemoveRareSequences(osIn, pCut = 0.001):
    print('Deleting sequences with probabilty smaller than ',pCut)
    osU,osK,symU = getUniqueSequences(osIn)
    osP = osK/sum(osK)
    iid = where(osP < pCut)[0]
    osUDelete = [list(osU[ii]) for ii in iid]
    osOut = []
    for ss in osIn:
        if ss not in osUDelete:
            osOut.append(ss)
    print('Out of ',len(osIn),' sequences, deleted ',len(osIn) - len(osOut), ' sequences. Sequences remain ',len(osOut))
    return osOut
    

# get the step probability distribution.        
# Inputs:
#   osT, sequences, symbols are numerical 1 to nSym
#   nSym, number of sylmols 
#   nStep, number of steps for computing the probabilities. 
# Return
#   PSteps, nStep x (nSym+1) matrix. PSteps[:,0] is the probability of ending at the steps. 
def getStepProbability(osT,nSym,nSteps):
    PSteps = zeros((nSteps,nSym+1))
    for ss in osT:
        for istep in range(min(len(ss),nSteps)):
            PSteps[istep,ss[istep]] += 1
        if istep < nSteps:  # end symbol
            PSteps[istep,0] += 1
    # normalize
    for ii in range(nSteps):
        ssm = sum(PSteps[ii,:])
        if ssm > 0:
            PSteps[ii,:] /= ssm
        else:
            PSteps[ii,0] = 1 
    return PSteps

# blance load on multiple process, returns arrar of number of computations each process should handle. 
# usefule wen nStask is not multiples of nProc
# Inputs:
#   nTot - total number of tasks
#   nProc - number of processors
#  reurns
#   NS - list of length nProc, number of tasks assinged to each processor
def computeNumTasksProc(nTot, nProc = 2):
    nS = int(nTot/nProc)
    nR = nTot - nS * nProc
    NS = [nS for ii in range(nProc)]
    for ii in range(nR):
        NS[ii] += 1
    return NS

# getSequenceCompletenessSampleToSample(osRef, osIn)    
# Compute sequence completeness comparing samples
# Inputs:
#   osRef - reference sequences
#   osIn  - sequences to be compared
# Returns:
#   Pc - sequence completeness
def getSequenceCompletenessSampleToSample(osRef, osIn):
    
    osUR,osKR,symUR = getUniqueSequences(osRef)
    PR = osKR/sum(osKR)
    

    osU,osK,symU = getUniqueSequences(osIn)
    Pc = 0
    for ss in osU:
        iid = -1
        for ii in range(len(osUR)):
            if list(ss) == list(osUR[ii]):
                Pc += PR[ii]
                break
    return Pc
            
# plotSequenceLengthDistribution(seqs,fn='')
# Plot sequence length distribution and save to fn.         
def plotSequenceLengthDistribution(seqs,fn=''):

    # analyze the length of sequences
    LL = []
    for ss in seqs:
        LL.append(len(ss))
    mmax = max(LL)
    Bins = [0.5]
    for ii in range(mmax):
        Bins.append(ii+0.5)
    yy,xx=histogram(LL,bins=Bins)
    xx = (xx[1:] +xx[:-1])/2
    plt.figure()
    plt.bar(xx,yy,color='gray')
    plt.xlabel('Seq Length')
    plt.ylabel('Counts')
    if fn != '':
        print('Saving figure to ',fn)
        plt.savefig(fn)     
    plt.show()
    
# plotProbDistribution(Ps,ylimMax=-1,xlimlow=0, width=0.02, xticks = [0,0.2,0.4,0.6,0.8,1],yticks = []) 
# plot sequence completeness in a nice way. 
def plotProbDistribution(Ps,ylimMax=-1,xlimlow=0, width=0.02, xticks = [0,0.2,0.4,0.6,0.8,1],yticks = []):
    cc,bb = histogram(Ps,bins=20)
    plt.bar(bb[:-1],cc,width=width,color='gray')
    plt.axis('off')
    plt.plot([xlimlow,1],[0,0],color='k')
    if ylimMax != -1:
        plt.ylim([-0.01*ylimMax,ylimMax])
        plt.plot([0,0],[0,ylimMax],color='k')
    ylim = plt.ylim()
    for tt in xticks:
        tlen = -ylim[1] * 0.01
        plt.plot([tt,tt],[0,tlen],color='k')
        plt.text(tt,-0.05*ylim[1],str(tt),horizontalalignment='center')
    for tt in yticks:
        tlen = 0.005
        plt.plot([-tlen,0],[tt,tt],color='k')
        plt.text(-tlen-0.01,tt,str(tt),horizontalalignment='right',verticalalignment='center')
    txt='median P= {P:.2f}'
    plt.text(0.5,ylim[1]/2,txt.format(P=median(Ps)))
    
# plot two POMM models in a way such that the corresponding states occupy the same positions. 
# parameteres:
#   S1, P1, Syms21 - POMM 1, state vector, transition probabilities, Syms2
#   S2, P2, Syms22 - POMM 2
#   StateCorres - disctionary of state correspondence from POMM 2 to POMM 1. 
#   filename - filename of the file created. 
def plotTwoPOMMsStateCorrespondences(S1In,P1In,Syms21,S2In,P2In,Syms22,StateCorres21,filename, iRemoveUnreachableState=1):
    
    
    S1,P1 = deleteTransitionSmallProb(S1In,P1In,iRemoveUnreachableState = iRemoveUnreachableState)
    S2,P2 = deleteTransitionSmallProb(S2In,P2In,iRemoveUnreachableState = iRemoveUnreachableState)
    
    # check consistency
    print('Checking consistency of state correspondence...')
    ss = unique(StateCorres21.keys())
    for ii in ss:
        if  len(where(StateCorres21.keys() == ii)[0])>1:
            print('ERROR in plotTwoPOMMsStateCorrespondences: Dupilcating state indices in StateCorres. Exit.')
            exit(1)
    ss2 = [StateCorres21[ii] for ii in StateCorres21.keys()]
    ss = unique(ss2)
    for ii in ss:
        if  len(where(ss2 == ii)[0])>1:
            print('ERROR in plotTwoPOMMsStateCorrespondences: Dupilcating state indices in StateCorres. Exit.')
            exit(1) 
    
    for ii in StateCorres21.keys():
        jj = StateCorres21[ii]
        if Syms22[S2[ii]] != Syms21[S1[jj]]:
            print('ERROR in plotTwoPOMMsStateCorrespondences: the symbols of corresponding states in StateCorres do not match.')
            print(' ii=',ii,'sym in POMM2: ',Syms22[S2[ii]],' jj=',jj,' sym in POMM1:',Syms21[S1[jj]])
            exit(1)
    
    labelStates = 0 # if state labels needed set to 1. 
    
    # create .dot file. 
    fp=open('temp.dot','w')
    fp.write('digraph G {\n')
    fp.write('size="8,4";\n');
    fp.write('rankdir=LR;\n')
    fp.write('ranksep=0.8; \n')
    
    # states in POMM1 that also appear in POMM2
    POMM1InPOMM2 = []
    POMM2InPOMM1 = []
    StateCorres12 = {}
    for s2 in StateCorres21.keys():
        s1 = StateCorres21[s2]  # corresponding state in POMM1.
        POMM1InPOMM2.append(s1) 
        POMM2InPOMM1.append(s2)
        StateCorres12[s1] = s2  # look up table from POMM1 to POMM2. 
        
    for jL in range(2): # go through laying out the graph, first POMM2 then POMM1. So that POMM2 is at the bottom after rendering.

        if jL == 0: # This case is POMM2. 
            print('Laying out POMM2 ...')
            stateOffSet = 0
            S = S2
            P = P2
            Syms2 = Syms22
                
        else:       # This case is POMM1.
            print('Laying out POMM1 ...')
            stateOffSet = len(S1)+len(S2)
            S = S1
            P = P1
            Syms2 = Syms21

        # get the syllables with multiple states
        symsMulti = []  
        SU = unique(S[2:])
        for ss in SU:
            kk = 0
            for ss2 in S[2:]:
                if ss == ss2:
                    kk += 1
            if kk > 1:
                sym = Syms2[ss]
                symsMulti.append(sym)
            
        print('syllables in multiple states: ',symsMulti)   
                                    
        # set states in S1. 
        IDs1 = {}   # record ids of the states
        iid = 0
        for ii in range(len(S1)):
            if ii == 1: # end state
                continue
            II = iid + stateOffSet  
            IDs1[ii] = II
            iid += 1
                        
            if ii == 0: # start state
                fp.write('%d [label="S",fontsize=25,style=filled,color="0.9 0.9 0.9",fontcolor="black"];\n' %(II))
                continue
        
            sym = Syms21[S1[ii]]
            if sym in symsMulti:
                cl = 'red'
            else:
                cl = 'black'
                    
            if labelStates == 1:
                symbol = str(i)+':'+str(sym)    
            else:
                symbol= str(sym)
                
            if jL == 0:
                if ii in POMM1InPOMM2:
                    s2 = StateCorres12[ii]
                    if P2[s2,1] > 0:    # the state transitions to the end state    
                        fp.write('%d [label="%s",fontsize=25,  style=filled,color="0.5 0.5 0.9",fontcolor=%s];\n' % (II,symbol,cl))
                    else:
                        fp.write('%d [label="%s",fontsize=25, fontcolor=%s];\n' %(II,symbol,cl))        
                else:   # make the state invisible
                    fp.write('%d [label="%s",style=invis];\n' %(II,symbol))             
            else:
                if P1[ii,1] > 0:    # the state transitions to the end state    
                    fp.write('%d [label="%s",fontsize=25,  style=filled,color="0.5 0.5 0.9",fontcolor=%s];\n' % (II,symbol,cl))
                else:
                    fp.write('%d [label="%s",fontsize=25, fontcolor=%s];\n' %(II,symbol,cl))

        # set states in S2. 
        IDs2 = {}
        for ii in range(len(S2)):
            if ii == 1: # end state
                continue
            if ii == 0: # start state
                s1 = IDs1[0]
                IDs2[0] = s1    # point to the state state of POMM1 
                continue
                
            if ii in POMM2InPOMM1:  # this state is already created             
                s1 = StateCorres21[ii]
                II = IDs1[s1]
                IDs2[ii] = II
                continue
            
            # the state is not in POMM1 
            II = iid + stateOffSet  
            IDs2[ii] = II
            iid += 1
                            
            sym = Syms22[S2[ii]]
            
            if sym in symsMulti:
                cl = 'red'
            else:
                cl = 'black'    

            if labelStates == 1:
                symbol = str(i)+':'+str(sym)    
            else:
                symbol= str(sym)

            if jL == 0:
                if P2[ii,1] > 0:    # the state transitions to the end state    
                    fp.write('%d [label="%s",fontsize=25,  style=filled,color="0.5 0.5 0.9",fontcolor=%s];\n' % (II,symbol,cl))
                else:
                    fp.write('%d [label="%s",fontsize=25, fontcolor=%s];\n' %(II,symbol,cl))    
            else:   
                # make the state invisible
                fp.write('%d [label="%s",style=invis];\n' %(II,symbol))             

        # plot edges
        pwR = 2         # line witdth
        pwG = 1.5
        pwB = 1
        
        # plot the edges of POMM1 and POMM2
        
        for iL in range(2):
            if iL == 0:
                S = S1
                P = P1
                IDs = IDs1
            else:
                S = S2
                P = P2
                IDs = IDs2

            edgeLists = []
            edgeProb = []
            for ii in range(len(S)):
                if ii == 1:
                    continue
                if ii > 0 and ii not in IDs.keys(): # dropped syllable
                    continue
                II1 = IDs[ii]
                for jj in range(2,len(S)):
                    if jj not in IDs.keys():    # dropped syllable
                        continue
                    II2 = IDs[jj]
                    if P[ii,jj] > 0:
                        if P[ii,jj] >= 0.5:
                            cl='red'
                            pw=pwR
                        elif P[ii,jj] >=0.1:
                            cl='green'
                            pw=pwG
                        else:
                            pw=pwB
                            cl='gray'
                        edgeLists.append([II1,II2,cl,pw])
                        edgeProb.append(P[ii,jj])
                
            # sort the edges
            iid = argsort(edgeProb)
            for kk in iid:
                II1,II2,cl,pw = edgeLists[kk]
                p = edgeProb[kk]    
                if (jL == 0 and iL == 0) or (jL == 1 and iL == 1): # make edges invisible
                    fp.write('%d -> %d [style=invis];\n' %(II1,II2))    
                else:   
                    fp.write('%d -> %d [arrowhead=normal, arrowsize=%0.2g, penwidth=%0.2g, color="%s"];\n' %(II1,II2,pw,pw,cl))

    fp.write('}\n')
    fp.close()

    # save graph files. 
    filenamePS = filename + '.ps'
    command = ['dot', '-Tps', 'temp.dot', '-o', filenamePS]     # this is for linux
    call(command)

# test how the C codes handle matrices
# In C: getModifiedSequenceCompletenessSamplingModelC(int nSeqs, int N, int *S, double *P, int nSample, double *PBs, double beta, int randSeed)
# Uncommeny 
#   //printf("\nIn C getModifiedSequenceCompletenessSamplingModelC  \n");
#   //PrintTransitionMatrix(N,P);
def testMatrixFormInCCodes():           
    
    S = [0, -1, 1, 2, 3, 4, 5]
    
    N = len(S)
    P =zeros((N,N))
    
    P[0,2] = 0.5
    P[0,3] = 0.5
    P[2,1] = 0.33
    P[2,4] = 0.67
    P[3,2] = 0.2
    P[3,4] = 0.8
    P[4,5] = 0.25
    P[4,6] = 0.75
    P[5,1] = 1
    P[6,1] = 1
    
    osIn = generateSequencePOMM(S,P,100)
    nSeq = len(osIn)
    
    print('\n\nTesting one...')

    syms = [1,2,3,4,5]
    P2, S2, C = ConstructMarkovModel(osIn,syms,pcut = 0.0)
    
    print(S2)
    printP(P2)          

    print(' Testing getModifiedSequenceCompletenessSamplingModelC')
    nSeq = len(osIn)
    PBs = getModifiedSequenceCompletenessSamplingModelC((S2,P2,nSeq,1))
        
    print('\n\nTesting two...')
    
    print(S)
    printP(P)           

    print(' Testing getModifiedSequenceCompletenessSamplingModelC')
    PBs = getModifiedSequenceCompletenessSamplingModelC((S,P,nSeq,1))

# break up the sequences according to syllables in motifSyllabels. 
# returns a dictionnary of motif sequences motifSeqs
def breakupSequencesIntoMotifSeqs(seqs,motifSyllabels):
    motifSeqs = {}
    motifSeqs['motifSyllabels'] = motifSyllabels
    for sym in motifSyllabels:
        motifSeqs[sym] = []
    
    for seq in seqs:
        ii = 0
        ss = seq.copy()
        ss.append('-1')
        for jj in range(len(ss)):
            if (ss[jj] == '-1' or ss[jj] in motifSyllabels) and jj > ii:
                motifSeqs[ss[ii]].append(ss[ii:jj])
                ii = jj                     
    return motifSeqs

# get syllables that start. These are part of the motif syllables.  
def getStartingSyllables(seqs):
    ssyms = []
    for seq in seqs:
        if seq[0] not in ssyms:
            ssyms.append(seq[0])
    return ssyms    

# get the syllabel labels.  
def getLabels(seqs):
    labels = []
    for seq in seqs:
        for ss in seq:
            if ss not in labels:
                labels.append(ss)
    return labels
        

# connstruct POMMs for each motif sequences.    
def constructPOMMsMotifSeqs(motifSeqs,nRerun=100,pValue=0.05,nProc=2,nSample=10000):
    
    motifPOMMs = {}
    motifPOMMs['motifLabels'] = motifSeqs['motifLabels']
    for sym in motifSeqs['motifLabels']:
        motifPOMMs[sym] = {}
        print('\nConstructing POMM for the motif sequences label = ',sym)
        
        seqs  = motifSeqs[sym]
        labels = getLabels(seqs)
        print(' For label sym=',sym,' sequences are:')
        for seq in seqs:
            ss = ''
            for sss in seq:
                ss += str(sss)+' '
            print(ss,' ',end="")
        print(' ')
        
        osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqs,labels) 
        Syms2 = {}
        Syms2R = {}
        symsNumeric = list(symsNumeric)
        for ii in range(len(symsNumeric)):
            Syms2[symsNumeric[ii]] = labels[ii]
            Syms2R[labels[ii]] = symsNumeric[ii]
        motifPOMMs[sym]['Syms2'] = Syms2
        motifPOMMs[sym]['Syms2R'] = Syms2R
        #get the POMM. 
        S2, P, pv, PBs, PbT, Pc = NGramPOMMSearch(osIn,nRerun=nRerun,pValue=pValue,nProc=nProc,nSample =nSample)
        S = [0,-1]
        for ss in S2[2:]:
            S.append(Syms2[ss])             
        motifPOMMs[sym]['S'] = S
        motifPOMMs[sym]['P'] = P
    return motifPOMMs       

def printSeq(seq):
    ssq = ''
    for ss in seq:
        ssq += ' '+str(ss)
    print(ssq)  

def printSequences(osIn, Syms2):
    kk = 0
    for seq in osIn:
        ssq = ''
        for ss in seq:
            ssq += ' '+Syms2[ss]
        print(ssq)      
        kk += 1
        if mod(kk,10) == 0:
            print(' ')
        

# get the POMM with the motif sequences.        
def getMotifPOMM(osIn,motifSyllabels,nRerun=100,pValue=0.05,nProc=2,nSample=10000):
    print('\nGetting motif sequences ...')
    motifSeqs = []
    motifSeqsCollect = []
    for seq in osIn:
        ii = 0
        ss = seq.copy()
        ss.append(-1)
        mseq = []
        mseqCollect = []
        for jj in range(len(ss)):
            if (ss[jj] == -1 or ss[jj] in motifSyllabels) and jj > ii:
                mseq.append(ss[ii])
                mseqCollect.append(ss[ii:jj])   
                printSeq(ss[ii:jj])             
                ii = jj 
        motifSeqs.append(mseq)
        motifSeqsCollect.append(mseqCollect)

    # make thse motif sequences consecutive.    
    #osInM, symsM = convertToNumericalSequences(motifSeqs,motifSyllabels)
    osInM, repeatNumSeqs, symsNumericM = getNumericalSequencesNonRepeat(motifSeqs,motifSyllabels)
    print('Using N-gram method to get the motif level POMM...')
    S, Pm, pv, PBs, PbT, Pc = NGramPOMMSearch(osInM,nRerun=nRerun,pValue=pValue,nProc=nProc,nSample=nSample)
    Sm = [0,-1]
    for ss in S[2:]:
        Sm.append(motifSyllabels[ss-1])
    return Sm, Pm, motifSeqs, motifSeqsCollect
        
# construct POMM by breaking the sequences into motifs then recombination. 
# Inputs:
#   osIn - sequences non-repeat, numerical sequences
#   motifSyllabels - starting syllables for motifs, including the start syllables from the sequences. 
#   nRerun,pValue,nProc,nSample - the usual parameters. 
# Returns S, P, pv, PBs, PbT
#   S - state vector
#   P - transition probabilities.   
#   pv - pValue
#   PBs - sampled Pb
#   PbT - Pb of the original seqeuences. 
def MinPOMMmotif(osIn,motifSyllabels,nRerun=100,pValue=0.05,nProc=nProc,nSample=10000):

    startSyms = getStartingSyllables(osIn)
    motifSyllabels = list(unique(motifSyllabels + startSyms))

    Sm, Pm, motifSeqs, motifSeqsCollect = getMotifPOMM(osIn,motifSyllabels,nRerun=nRerun,pValue=pValue,nProc=nProc,nSample=nSample)
    print('Motif level POMM Sm=',Sm)
        
    # get the motif sequences. 
    mSeqs = {}
    mLabels = [ii for ii in range(2,len(Sm))]
    mSeqs['motifLabels'] = mLabels
    for ss in mLabels:
        mSeqs[ss] = []
    
    for ii in range(len(motifSeqs)):
        seq = motifSeqs[ii]
        stateSeq, prob = computeMostProbableStateSequence(Sm,Pm,seq)
        for jj in range(len(stateSeq)):
            ss = stateSeq[jj]
            mSeqs[ss].append(motifSeqsCollect[ii][jj])

    # construct motif sequences.        
    motifPOMMs = constructPOMMsMotifSeqs(mSeqs,nRerun=nRerun,pValue=pValue,nProc=nProc,nSample=nSample)

    S = []
    for ii in range(2,len(Sm)):
        SS = motifPOMMs[ii]['S']
        S += SS[2:]
    S = list(sort(S))
    S = [0,-1] + S
    print('After merging motif POMMs S=',S) 

    # simplify by deleting states       
    S, P, pv, PBs, PbT, Pc = MinPOMMSimpDeleteStates(S,osIn,nProc=nProc,nRerun=nRerun,nSample=nSample)          
    print('After state deletion pv=',pv)
    
    return S, P,  pv, PBs, PbT
            

# construct the POMM by introducing motifs starting with specific syllables. 
def testMinPOMMmotif():
    
    # model for sequence generation
    S0 = [0,-1,'a','b','c','a','c','d','e','b','c','e','c','d']
    N0 = len(S0)
    P0 = zeros((N0,N0))
    P0[0,2] = 1
    P0[2,3] = 1
    P0[3,4] = 1
    P0[4,5] = 0.2
    P0[4,8] = 0.8
    P0[5,6] = 1
    P0[6,7] = 1
    P0[7,1] = 1
    P0[8,9] = 1
    P0[9,10] = 1
    P0[10,11] = 1
    P0[11,12] = 1
    P0[12,13] = 1
    P0[13,5] = 0.5
    P0[13,1] = 0.5
    P0 = normP(P0)
    sylLabels = list(unique(S0[2:]))
    print(sylLabels)
        
    motifSyllabels = ['a','e']
    print('motif syllabels = ',motifSyllabels)

    plotTransitionDiagram(S0,P0,Pcut=0.01,filenamePS='temp.Original.POMM.ps')
        
    nSeq = 100
    seqsOrig = generateSequencePOMM(S0,P0,nSeq)

    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 
    Syms2 = {}
    Syms2R = {}
    symsNumeric = list(symsNumeric)
    for ii in range(len(symsNumeric)):
        Syms2[symsNumeric[ii]] = sylLabels[ii]
        Syms2R[sylLabels[ii]] = symsNumeric[ii]
    mSyls = []
    for ss in motifSyllabels:
        ii = Syms2R[ss]
        mSyls.append(ii)    
        
    for seq in seqsOrig:
        ss = ''
        for sym in seq:
            ss += sym
        print(ss,' ',end="")
            
    
    S, P, pv, PBs, PbT = MinPOMMmotif(osIn,mSyls,nRerun=100,pValue=0.05,nProc=nProc,nSample=10000)
    
    # simplify by cutting connections       
    S, P, pv, PBs, PbT = MinPOMMSimp(S,osIn,minP = 0.001,nProc=nProc,nRerun=100,pValue=0.05,factors=[0.5])                  
    print('After simplification pv=',pv)
    
    
    S2 = [0,-1]
    for ss in S[2:]:
        S2.append(Syms2[ss])
        
    plotTransitionDiagram(S2,P,Pcut=0.01,filenamePS='temp.Combined.POMM.ps')

# get unique sequences for POMM given S, P, calling C function  
def getUniqueSeqsProbsPOMM(S,P):

    N = len(S)
        
    # set parameter types   
    lib.getUniqueSeqProbsPOMM.argtypes = [
                ctypes.c_long,  
                ctypes.POINTER(ctypes.c_long), 
                ctypes.POINTER(ctypes.c_double)
    ]       
    lib.getUniqueSeqProbsPOMM.restype = ctypes.POINTER(ctypes.c_double)
    # call the C function.
    S = array(S,dtype=int)
    seqProbs = lib.getUniqueSeqProbsPOMM(
                ctypes.c_long(N),
                S.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), 
                P.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    nU = int(seqProbs[0])
    
    pU = [seqProbs[i] for i in range(1,nU+1)]
        
    lib.freeArray.argtypes = [ctypes.POINTER(ctypes.c_double)]  
    lib.freeArray(seqProbs)
    
    return nU, pU
    
    
def testGetUniqueSeqsProbsPOMM():
                    
    SO, PO = getTestModel(iModel = 1)           
                
    nU, pU = getUniqueSeqsProbsPOMM(SO,PO)
    print('Found ',nU,' unique sequences. ')
    print('Ptot = ',"{:.2f}".format(sum(pU)))

    # print prob in a nice way. 
    formatted_list = ["{:.5f}".format(x) for x in pU]
    formatted_list_string = ', '.join(formatted_list)
    print('Probs = ',formatted_list_string)
    
    
"""

Models used for testing. 
Returns SO, PO

iModel values:
    1, two state for each symbol, example used in the paper. 
    2, Markov model. 
    3, probability dependent POMM ACD, ACE (less probable), BCD (less probable), BCE, for Fig.1
    4, Markov model corresponding to the iModel=1, the example in the paper. 
    5, Linear POMM. 

"""     

def getTestModel(iModel = 1):

    
    if iModel == 1:
        # state vector
        SS = [0,-1,'A','B','C','C','A','D','E']
        N = len(SS)
        # convert to numberic symbols. 
        Syms = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        PO = zeros((N,N))
        PO[0,2] = 0.5
        PO[0,3] = 0.5
        PO[2,4] = 0.8
        PO[2,1] = 0.2
        PO[3,5] = 0.5
        PO[3,6] = 0.5
        PO[4,7] = 0.9
        PO[4,8] = 0.1       
        PO[5,7] = 0.2
        PO[5,8] = 0.8
        PO[6,8] = 0.5
        PO[6,1] = 0.5
        PO[7,1] = 1.0
        PO[8,1] = 1.0   
    elif iModel == 2:
        # state vector
        SS = [0,-1,'A','B','C']
        N = len(SS)
        # convert to numberic symbols. 
        Syms = {'A':1,'B':2,'C':3}
        Syms2 = {1:'A',2:'B',3:'C'}
        # state transition probabilities
        PO = zeros((N,N))
        PO[0,2] = 0.5
        PO[0,3] = 0.5
        PO[2,3] = 1.0/3
        PO[2,4] = 1.0/3
        PO[2,1] = 1.0/3
        PO[3,2] = 0.5
        PO[3,1] = 0.5
        PO[4,3] = 0.5
        PO[4,1] = 0.5
    elif iModel == 3:
                
        # state vector
        SS = [0,-1,'A','B','C','C','D','E']
        N = len(SS)
        # convert to numberic symbols. 
        Syms = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        PO = zeros((N,N))
        PO[0,2] = 0.5
        PO[0,3] = 0.5
        PO[2,4] = 1
        PO[3,5] = 1
        PO[4,6] = 0.8
        PO[4,7] = 1 - PO[4,6]
        PO[5,6] = 0.2
        PO[5,7] = 1 - PO[5,6]
        PO[6,1] = 1
        PO[7,1] = 1 
    elif iModel == 4:
        # state vector
        SS = [0,-1,'A','B','C','D','E']
        N = len(SS)
        # convert to numberic symbols. 
        Syms = {'A':1,'B':2,'C':3,'D':4,'E':5}
        Syms2 = {1:'A',2:'B',3:'C',4:'D',5:'E'}
        # state transition probabilities
        PO = zeros((N,N))
        PO[0,2] = 0.5
        PO[0,3] = 0.5
        PO[2,4] = 0.53
        PO[2,6] = 0.17
        PO[2,1] = 0.3
        PO[3,2] = 0.5
        PO[3,4] = 0.5
        PO[4,6] = 0.37
        PO[4,5] = 0.63
        PO[5,1] = 1
        PO[6,1] = 1 
    elif iModel == 5:
        # model for sequence generation
        SS = [0,-1,'a','b','c','a','c','d','e','b','c','e','c','d']
        N = len(SS)
        PO = zeros((N,N))
        PO[0,2] = 1
        PO[2,3] = 1
        PO[3,4] = 1
        PO[4,5] = 0.2
        PO[4,8] = 0.8
        PO[5,6] = 1
        PO[6,7] = 1
        PO[7,1] = 1
        PO[8,9] = 1
        PO[9,10] = 1
        PO[10,11] = 1
        PO[11,12] = 1
        PO[12,13] = 1
        PO[13,5] = 0.5
        PO[13,1] = 0.5
    PO = normP(PO)
            
        
    print('SS = ',SS)
    printP(PO)

    return SS, PO
    

"""

PB = constructSymPB(osIn, sym, ng)

Constructs the transition probability matrix P centered around sym using sequences in osIn. 
This is the transition probabilities of sequences of length up to ng left of sym to those right of sym. 
Let's call it the bottle neck matrix. It will be used to find out the number of hidden states needed fro sym in POMM. 

Inputs:
    
    osIn    - input sequences, list of lists of numberics ranging from 1 to N. 
    sym     - center symbol
    ng      - sequence length left or right of sym

Returns

    PB      - the bottle neck transition probabilities  

"""
def constructSymP(osIn, sym, ng):   
    
    uniqueSeqsLeft = []
    uniqueSeqsRight = []
    transitionsCounts = {}
    
    for seq in osIn:
        
        seq = [0] + seq + [-1]  # flank start and end symbols. 
        
        #print(seq)
        nn = len(seq)
        
        for ii in range(1,nn-1):
            if seq[ii] == sym:
                lseq = seq[max(0,ii-ng):ii]
                rseq = seq[ii+1:min(nn,ii+1+ng)] 
                
                #print('    l=',lseq, ' r=',rseq)
                
                if lseq in uniqueSeqsLeft:
                    i = uniqueSeqsLeft.index(lseq)
                else:
                    uniqueSeqsLeft.append(lseq)
                    i = len(uniqueSeqsLeft) - 1

                if rseq in uniqueSeqsRight:
                    j = uniqueSeqsRight.index(rseq)
                else:
                    uniqueSeqsRight.append(rseq)
                    j = len(uniqueSeqsRight) - 1
                
                if (i,j) in transitionsCounts.keys():                   
                    transitionsCounts[(i,j)] += 1
                else:
                    transitionsCounts[(i,j)] = 1
                    
    #print(uniqueSeqsLeft)
    #print(uniqueSeqsRight)
    #print(transitionsCounts)
    
    PB = zeros((len(uniqueSeqsLeft),len(uniqueSeqsRight)))
    MM = zeros(len(uniqueSeqsLeft))
    for i in range(len(uniqueSeqsLeft)):
        for j in range(len(uniqueSeqsRight)):
            if (i,j) in transitionsCounts.keys():
                PB[i,j] = transitionsCounts[(i,j)]
                MM[i] += PB[i,j] 
    
    #print('PB:')
    #print(PB)
    PB = array(PB,dtype=float64)
    # scale
    for i in range(len(uniqueSeqsLeft)):
        PB[i,:] /=MM[i]
        #PB[i,:] /= sqrt(sum(PB[i,:]*PB[i,:]))

    return PB
    

# use unique sequences and counts to construct PB for sym.  
def constructSymPuniqueSeqs(osU, osK, sym, ng): 
    
    uniqueSeqsLeft = []
    uniqueSeqsRight = []
    transitionsCounts = {}
    
    for iu in range(len(osU)):
        seq = list(osU[iu])
        cc = osK[iu]
        seq = [0] + seq + [-1]  # flank start and end symbols. 
                
        nn = len(seq)       
        for ii in range(1,nn-1):
            if seq[ii] == sym:
                lseq = seq[max(0,ii-ng):ii]
                rseq = seq[ii+1:min(nn,ii+1+ng)] 
                
                if lseq in uniqueSeqsLeft:
                    i = uniqueSeqsLeft.index(lseq)
                else:
                    uniqueSeqsLeft.append(lseq)
                    i = len(uniqueSeqsLeft) - 1

                if rseq in uniqueSeqsRight:
                    j = uniqueSeqsRight.index(rseq)
                else:
                    uniqueSeqsRight.append(rseq)
                    j = len(uniqueSeqsRight) - 1
                
                if (i,j) in transitionsCounts.keys():                   
                    transitionsCounts[(i,j)] += cc
                else:
                    transitionsCounts[(i,j)] = cc
                        
    PB = zeros((len(uniqueSeqsLeft),len(uniqueSeqsRight)))
    MM = zeros(len(uniqueSeqsLeft))
    for i in range(len(uniqueSeqsLeft)):
        for j in range(len(uniqueSeqsRight)):
            if (i,j) in transitionsCounts.keys():
                PB[i,j] = transitionsCounts[(i,j)]
                MM[i] += PB[i,j] 
    
    PB = array(PB,dtype=float64)
    # scale
    for i in range(len(uniqueSeqsLeft)):
        PB[i,:] /=MM[i]

    return PB
    
    
    
"""

Compute the state vector using approximate ranks of PBs for each symbol

S = getStateVecUsingRanks(osIn, symsNumeric, cutOff = 0.1)

Inputs:
    
    osIn        - input sequences, should be the numeric sequences. 
    symsNumeric - symbols in the sequences
    ng          - length of context dependence or n-gram degree. 
    cutOff      - cut off the sigular values for getting the approximate ranks. 
    
Returns:

    S       - state vectors. 

""" 

def getStateVecUsingRanks(osIn, symsNumeric, ng, cutOff = 0.1):
    
    print('Getting the state vector using rank estimates. ng=',ng,' cutOff=',cutOff)

    # get PB. 
    S = [0, -1]
    for sym in symsNumeric:
        print(' Constructing PB for sym=',sym)      
        PB = constructSymP(osIn, sym, ng-1)
            
        nS = 1
        
        if min(PB.shape) > 1:
                        
            cm = min(ng,min(PB.shape))          
            svd = TruncatedSVD(n_components=cm)

            # Fit the model to the data
            svd.fit(PB)

            # Access the singular values
            singular_values = svd.singular_values_
            singular_values /= singular_values[0]
            
            nS = len(where(singular_values > cutOff)[0])
        print('     numStates = ',nS)
        for i in range(nS):
            S.append(sym)
    print(' estimated S=',S)
    return S        
                
    
def testConstructSymP():
        
    S, P = getTestModel(iModel = 5)
    
    ng = 4          # n gram size. 
    
    nSeq = 90
    seqsOrig = generateSequencePOMM(S,P,nSeq)
    sylLabels = list(unique(S[2:]))
        
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 
    osU, osK, symU = getUniqueSequences(osIn)

    for sym in symsNumeric:
        print('Getting PB for sym full sequence method=',sym)
        PB = constructSymP(osIn, sym, ng-1)         
        print(PB)   
        print('Getting PB for sym unique sequences method=',sym)        
        PB = constructSymPuniqueSeqs(osU, osK, sym, ng)
        print(PB)
        print(' ')

def testGetStateVecUsingRanks():
                        
    S, P = getTestModel(iModel = 4)
    
    ng = 2          # n gram size. 
    
    nSeq = 200
    seqsOrig = generateSequencePOMM(S,P,nSeq)
    sylLabels = list(unique(S[2:]))
        
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 
    
    SE = getStateVecUsingRanks(osIn, symsNumeric, ng, cutOff = 0.1)
    
"""

Derive POMM by computing the approximate rank of PB matrix. This replaces the n-gram approach. 

S, P, pv, PbT, Pbetas = PBRankPOMM(osIn, ngramStart = 1, fnSave='') 

Parameters:

    osIn        - input sequences in numbers. 
    ngramStart  - starting value of ngram when constructing PB matrix. Default 1, Markov model. 
    fnSave      - filename of saving intermediate results.
    
Returns:

    S           - state vector
    P           - transition matrix
    pv          - p value
    PbT         - Pbeta of osIn given the model
    Pbetas      - Pbetas of the samples. 

"""

def PBRankPOMM(osIn, ngramStart = 1, fnSave=''):

    print('Constructing POMM with the approximate rank method...')
    
    osU,osK,symU = getUniqueSequences(osIn)
    print(' Num unique sequences = ',len(osU))
    print(' Syms ',symU)
    
    maxNG = 100
    cutOffs = [0.1,0.2,0.3,0.4,0.5]
    ng = ngramStart
    while ng < maxNG:
        print('\n   Testing nGram size ng = ',ng)
        
        print(' Computing number of states for syms using the approximate rank method...')
        t1 = time.time()
        ranksRets = [[None] * len(symU) for ic in range(len(cutOffs))]
        # List to hold the thread objects
        threads = []
        # Creating and starting threads
        for sym in symU:
            thread = threading.Thread(target=getApproxPBRank, args=(ranksRets,osU,osK,sym,ng,cutOffs))
            threads.append(thread)
            thread.start()
        # Wait for all threads to complete
        for thread in threads:
            thread.join()           
        t2 = time.time()
        print('     Used ',t2-t1,'sec')
        
        # try different cutoffs 
        flag = 0    
        S0 = []
        for ic in range(len(cutOffs)):
            print('     Testing states with cutOff=',cutOffs[ic])
            
            S = [0, -1]
            for sym in symU:
                nS = ranksRets[ic][sym-1]
                for ii in range(nS):
                    S.append(sym)
            print(' S=',S)
            N = len(S)
            
            if S == S0:
                print(' Already tested. Skip this cut off.\n')
                continue
            if flag == 1 and len(S) > len(S0):
                print(' More complex model then before. Skip this cut off.\n')
                continue    

            # get the transition matrix. 
            print(' BW algorithm for the transition matrix...') 
            t1 = time.time()
            pTol = 1e-3 # tolerance of the transition probabilties. 
            maxSteps = 10000    
            
            if 0:   # use multiple processors for BW selecting the best llk model
            
                Ps = []
                for irun in range(nProc):
                    P = normP(rand(N,N))
                    Ps.append([osU,osK,S,P,pTol,maxSteps])
                pool = Pool(processes = nProc)
                res = pool.map(BWPOMMCFun,Ps,chunksize = 1)
                pool.close()
                pool.join()     
                ML = [ml for (ml, P) in res]
                ML = array(ML)
                iid = ML.argmax()
                P = res[iid][1]
            
            else:   # call multi-thread C function model.
            
                P = normP(rand(N,N))
                ml,P = BWPOMMCFunMultiThread(osU, osK, S, P, pTol, maxSteps, nProc)         
                
            t2 = time.time()
            print('     Used ',t2-t1,'sec')
            
            # test the statsicial signifance. 
            t1 = time.time()
            pv, Pbetas, PbT = getPVSampledSeqsPOMM(S, P, osIn)
            print(' Pb sampled range=(',round(Pbetas.min(),3),round(Pbetas.max(),3),') seq Pb=', round(PbT,3))
            t2 = time.time()
            print('     Used ',t2-t1,'sec')

            if fnSave != '':
                savePBRankPOMMRes(S,P,ng,pv,PbT,fnSave)
                
            print(' S=',S)
            if pv > pValue:
                print(' Accepted pv=',round(pv,3))
                flag = 1
                S0 = S
                P0 = P
            elif flag == 1:         
                print(' Rejected pv=',round(pv,3))
                print(' Reverting back to previously accepted model.')
                flag = 2
                S = S0
                P = P0
                break
            else:   
                print(' Rejected pv=',round(pv,3))
                break
        if flag != 0:
            break       
        else:
            ng += 1  # test next n gram.    

    print('Final S=',S)             
    return S, P, pv, PbT, Pbetas 
    
    
def getApproxPBRank(ranksRets,osU,osK,sym,ng,cutOffs):
    
    PB = constructSymPuniqueSeqs(osU, osK, sym, ng-1)
    
    for ic in range(len(cutOffs)):
        cutOff = cutOffs[ic]
        nS = 1      
        if min(PB.shape) > 1:
            cm = min(PB.shape)      
            if 1: # use the sparse matrix randomized svd approach   

                # make it a parse matrix. 
                PB = csr_matrix(PB)
                # Perform randomized SVD
                U, Sigma, VT = randomized_svd(PB, n_components=cm, n_iter=5, random_state=None)
                thr = Sigma[0] * cutOff
                nS = sum(Sigma > thr)
        
            if 0:                   
                svd = TruncatedSVD(n_components=cm)
                # Fit the model to the data
                svd.fit(PB)
                # Access the singular values
                singular_values = svd.singular_values_
                singular_values /= singular_values[0]               
                nS = len(where(singular_values > cutOff)[0])

        ranksRets[ic][sym-1] = nS
    
def savePBRankPOMMRes(S,P,ng,pv,PbT,fnSave):
    print(' Saving the results to ',fnSave)
    fn = open(fnSave,'wb')
    pickle.dump([S,P,ng,pv,PbT],fn)
    fn.close()  

def loadPBRankPOMMRes(fnSave):
    print('Reading previous run from ',fnSave)
    fn = open(fnSave,'rb')
    S,P,ng,pv,PbT = pickle.load(fn)
    fn.close()
    return S,P,ng,pv,PbT        
    
def testPBRankPOMM():
    
    S, P = getTestModel(iModel = 5)
    
    nSeq = 200
    seqsOrig = generateSequencePOMM(S,P,nSeq)
    sylLabels = list(unique(S[2:]))
                        
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 

    PBRankPOMM(osIn, ngramStart = 2, fnSave='testPBRankPOMM.dat')   
    
"""
    
S, P, pv, PBs, PbT = PBnGramSearch(osIn, ngramStart = 1,  fnSave='')

    Construct ngram model, then merge states by comparing the downstream symbol sequences of length up to n. 
    
    Parameters:
        osIn        - sequences in numeric form
        ngramStart  - starting point of n gram for search.
        fnSave      - save the ngrams during the search. 
        
    Returns:
        S           - state vector
        P           - transition probabilities
        PBs         - sampled Pbs
        PbT         - Pb of the observed sequence. 

""" 

def PBnGramSearch(osIn, ngramStart = 1, fnSave=''):
        
    print('Constructing POMM with nGram transition diagram and state merging based on downstream sequences...')
    flag = 0
    maxNG = 200
    
    for ng in range(ngramStart,maxNG):
        print('\nTesting nGram size ng = ',ng)
        
        # construct nGram POMM
        S, P, SnumVis = constructNGramPOMMC(osIn, ng)
        #S, P, SnumVis = constructNGramPOMM(osIn, ng)
        
        # test the statsicial signifance. 
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn)
        print(' Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))

        if fnSave != '':
            saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave)
            
        print(' S=',S)
        if pv > pValue:
            print(' Accepted pv=',round(pv,3))
            flag = 1
            break
        else:
            print(' Rejected pv=',round(pv,3))
                    
    if flag == 0:
        print('WARNING: in NGramPOMMSearch: no NGram model accepted up until ng=',ng)
        return S, P, pv, PBs, PbT
    
    
    # test merge states.
    syms = list(unique(S[2:]))
    
    for sm in syms:
        iid = where(sm == array(S))[0]
        if len(iid) <= 1:
            continue
        print(' Test merging states for sym  = ',sm)

        # construct the PB matrix for the symbol. 

        t1 = time.time()
        results = {}
        threads = []
        # Creating and starting threads
        for i in range(len(iid)):
            iState = iid[i]
            thread = threading.Thread(target=computePBFromSym, args=(S,P,ng,iState,results,i))
            threads.append(thread)
            thread.start()
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        nGramInds = {}
        for i in range(len(iid)):
            seqs, probs = results[i]
            for ss in seqs:
                ss = tuple(ss)
                if ss not in nGramInds.keys():
                    nGramInds[ss] = len(nGramInds.keys())
                    
        M = len(nGramInds.keys())
        PB = zeros((len(iid),M))
        for i in range(len(iid)):
            seqs, probs = results[i]
            for kk in range(len(seqs)):
                ss = tuple(seqs[kk])
                pp = probs[kk]
                jj = nGramInds[ss]
                PB[i,jj] = pp
            PB[i,:] /= sum(PB[i,:]) 
        
        for i in range(len(iid)):
            PB[i,:] /= sqrt(sum(PB[i,:] * PB[i,:]))
        
        t2 = time.time()
        print('     Getting PB used ',t2-t1,' sec')
        
        t1 = time.time()
        merged0 = []
        for ccMerge in [0.2, 0.1, 0.05]:
            
            print('     Testing ccMerge=',ccMerge) 
            
            N = len(S)
            print(' S=',S)
            PTest = P.copy()
            SnumVisTest = SnumVis.copy()

            iidsToDelete = []
            merged = []
            for i in range(len(iid)):
                ii = iid[i]
                if ii in iidsToDelete:
                    continue

                for j in range(i+1,len(iid)):
                    jj = iid[j]
                    if jj in iidsToDelete:
                        continue
                    # check if this should be merged. 
                    cc = 1 - sum(PB[i,:] * PB[j,:])
                                    
                    if cc < ccMerge:
                        print('     merging states (',ii,jj,') sym=',sm)
                        for kk in range(N):
                            if kk == ii or kk == jj:
                                continue
                            if SnumVisTest[ii]+SnumVisTest[jj] == 0:
                                PTest[ii,kk] = 0
                                PTest[jj,kk] = 0
                                continue
                            PTest[ii,kk] = SnumVisTest[ii]*1.0/(SnumVisTest[ii]+SnumVisTest[jj]) * PTest[ii,kk] + \
                                           SnumVisTest[jj]*1.0/(SnumVisTest[ii]+SnumVisTest[jj]) * PTest[jj,kk]                                    
                            PTest[jj,kk] = 0
                            SnumVisTest[ii] = SnumVisTest[ii]+SnumVisTest[jj]
                            SnumVisTest[jj] = 0
                        PTest[jj,1] = 1.0
                        for kk in range(N):
                            if kk == ii or kk == jj:
                                continue
                            PTest[kk,ii] = PTest[kk,ii] + PTest[kk,jj]
                            PTest[kk,jj] = 0
                        PTest[ii,jj] = 0                                    
                        iidsToDelete.append(jj)
                        merged.append([ii,jj])
                                                
            # delete states
            if len(iidsToDelete) > 0:   
            
                if merged == merged0:
                    print('         This merge already tested. Skip. ')
                    continue
                merged0 = merged

                # test if the merge is good. 
                PTest = normP(PTest)
                pvTest, PBsTest, PbTTest = getPVSampledSeqsPOMM(S, PTest, osIn)
                print('     Pb sampled range=(',round(PBsTest.min(),3),round(PBsTest.max(),3),') seq Pb=', round(PbTTest,3))

                if pvTest > pValue:
                    print('     merge accepted pv=',round(pv,3))
                    SnumVis = SnumVisTest.copy()
                    iidsToDelete = list(sort(iidsToDelete))
                    P = PTest.copy()
                    for jj in iidsToDelete[::-1]:
                        S = S[:jj] + S[jj+1:]
                        P = delete(P,jj,axis=0)
                        P = delete(P,jj,axis=1)
                        SnumVis = SnumVis[:jj] + SnumVis[jj+1:]
                    pv = pvTest
                    PBs = PBsTest
                    PbT = PbTTest   
                    break
                else:
                    print('     merge rejected.')
                    iidsToDelete0 = iidsToDelete
                
                if fnSave != '':
                    saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave)
            else:
                print('     No merger candidate.')
                break
        t2 = time.time()
        print('     Testing mergers used ',t2-t1,' sec\n')
        
        
                            
    # final model. 
    print('Found model S=',S)
    print('pv=',pv) 
    print('Pb=',PbT)
    
    return S, P, pv, PBs, PbT
    
def computePBFromSym(S,P,ng,iState,results,i):
    
    frontStates = [iState]  # frontier of growth. 
    frontSeqs = [[S[iState]]]   # sequences correspoding to the front states
    frontLogProbs = [0.0]       # log probabilities. 
    
    nGramSeqs = []
    nGramProbs = []
    
    N = len(S)
    
    for istep in range(ng):
        
        fStates = []
        fSeqs = []
        fLogProbs = []

        for kk in range(len(frontStates)):
            
            ids = frontStates[kk]
            seq = frontSeqs[kk]
            logProb = frontLogProbs[kk]
            
            for jj in range(N):
                p = P[ids,jj]
                if p > pTolence: 
                    logP = logProb + log(p)
                    seq2 = seq + [S[jj]]
                    
                    if jj == 1: # this is the end state. 
                        nGramSeqs.append(seq2)
                        nGramProbs.append(exp(logP))
                        
                    else:
                        fStates.append(jj)
                        fSeqs.append(seq2)
                        fLogProbs.append(logP)


        
        frontStates = fStates
        frontSeqs = fSeqs
        frontLogProbs = fLogProbs

        if len(frontStates) == 0:
            break       
            
    for kk in range(len(frontStates)):
        seq = frontSeqs[kk]
        logProb = frontLogProbs[kk]
        nGramSeqs.append(seq)
        nGramProbs.append(exp(logProb))

    results[i] = [nGramSeqs,nGramProbs]     
    
    

def testPBnGramSearch():
    
    S, P = getTestModel(iModel = 5)
    
    nSeq = 100
    seqsOrig = generateSequencePOMM(S,P,nSeq)
    sylLabels = list(unique(S[2:]))
                        
    osIn, repeatNumSeqs, symsNumeric = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 
    
    S, P, pv, PBs, PbT = PBnGramSearch(osIn, ngramStart = 2, fnSave='testPBnGramSearch.dat')
        

                    
if __name__ == "__main__":
    
    #testComputeMostProbableStateSequence()
        
    #testGetModifiedSequenceCompletenessSamplingModelC()
    
    #testMatrixFormInCCodes()
    
    #testCheckEndStateReachability()
    
    #testMinPOMMmotif()
    
    #testGetUniqueSeqsProbsPOMM()

    #testNGramOrTreePOMMSearch()
    
    #testConstructNGramPOMM
    
    #testComputeSeqProbPOMMC()
    
    #testConstructSymP()
    
    #testGetStateVecUsingRanks()
    
    #testPBRankPOMM()
    
    testPBnGramSearch()
