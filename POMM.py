'''
 This python library contains functions for deriving POMM from observed seqeunces
 Useful concepts

   state vector: specifies symbols associated with the states.
       The first is the start state 0, and the second is the end state -1, and the results are the other states. 
       for example, S = [0, -1, 'A', 'B', 'A'] 
   transition probability P of the state vectors. 
       This specifes the transition probabilities between the states. 

 Written by Dezhe Jin, Department of Physics, Penn State, dzj2@psu.edu, 9/9/2015, updated 4/14/2022, 12/10/2025, 3/27/2026 

 Updates
    2025-12-10  plotTransitionDiagram(S,P,Pcut=0.01,filenameDot='temp.dot',filenamePDF='temp.pdf',removeUnreachable=False,markedStates=[],labelStates=0)
                    changd output to PDF instead of PS file format. filenamePDF='temp.pdf'
                
                getNumericalSequencesNonRepeat(seqs,syllableLabels)
                    changed return. Now returns numericSeqs, repeatNumSeqs, Syms, Syms2
                    Here Syms is a dictionary converting syms to numerics, and Syms2 converts numerics to syms. 
                    
    2026-03-27  Major update on NGramPOMMSearch, which infers POMM from observed sequences. 
    
   
'''

from subprocess import call
import numpy as np
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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Number of available CPUs

nCPU = os.cpu_count() or 1
nProc = nCPU - 1 if nCPU > 1 else 1

#import rpy2.robjects as robjects   # for Fisher exact test
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()

# load the C program library. 
lib = ctypes.CDLL(str(Path(__file__).resolve().parent / "libPOMM.so"))

# PARAMETERS
betaTotalVariationDistance = 0.2    # the factor for modifying the sequence completeness adding the total 
                                    # variation distance, to include the effects of transition probability dependent context dependence
                                    # set this to 0, it becomes pure sequence completeness. 
pValue = 0.05                       # p-value for accepting POMM based on the distributiuon of Pb.      
BWRerun = 20                        # number of times Bohm-Welsh alogrith is ran. 
nSamples = 10000                    # number of samples for getting pv from the Pbeta distribution.     
pTolence = 1e-6                     # smallest transition probability.                      
                                    
#print('In POMM, the total variation distance is weighted with the factor betaTotalVariationDistance=',betaTotalVariationDistance)
#print('In POMM, pValue is set to ',pValue)
    
"""

    List of all functions

    NGramPOMMSearch
   
        Default POMM inferenece method
   
        Construct POMM using n-gram model.
        Successively build n-gram transition models, and test for Pbeta significance.
        Then merge states. 
   
        S, P, pv, PBs, PbT = NGramPOMMSearch(osIn, pValue=0.05, Pcut=0.001, stateMergeParam=[1, 0.1, 0.1], nSample = 10000, ngramStart = 1, fnSave=''):

        Inputs: 
        
        osIn    - list of observed sequences. Symbols must be 1,2,...,n, where n is the number of symbols. 
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

    MinPOMMSimpDeleteStates(S,osIn, nRerun = 50, pValue=pValue, nSample=10000, fnSave='')
        
         Simplify by deleting states and making sure that the maximum likelihood remains within bound. 
         Input parameters:
            S - initial POMM
            osIn - observed sequences
            nRerun - number of times B-W is run with different seeds. 
         Return S, P, Pc    
            S - state vector
            P - transition probabilities
            Pc - sequence completeness
    
        
        
   Some other usefyl functions     

       getPVSampledSeqsPOMM(S, P, osIn,nSample = 10000)

           get the p-value of the observed seqeunces against the Pb of the sampled sequences for a given POMM
           This method is through generating sequences from POMM
           Inputs:
               S, state vector
               P, transition probabilities
                osIn, observed sequence
                nSample, number of samples, default 10000
            Returns:
                pv, p-value of the observed sequence on the model.
                PBs, modified sequence completeness sampled
                PbT, modified sequence completeness of the observed sequences
     
     
        getUniqueSequences(osIn)

            Input:
                osIn, list of sequences
            Returns:
                osU, unique sequences
                osK, number of times the unique sequences appear
                symU, symbols.
     
     
        BWPOMMCParallel(S,osInO,maxSteps=5000,pTol=1e-6, nRerun=100)

            Parallel version of BWPOMM, calling C function BWPOMMC from libPOMM.h   
            Inputs:
                S, state vector
                osInO, observed sequences
                maxSteps, maximum number of steps for updating the transition probabilities
                pTol, tolerance for the transition probabilities
                nRerun, number of times the algorithm is run. 
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
     
     
        computeLogLike(S, P,osU,osK)

            Compute log likelihood of the seqeucens given the POMM. 
            inputs
                S, states
                P, transition matrix
                osU, unique sequences
                osK, counts. 
            returns
                llk, log likelihood.
     
     
        normP(P, Pcut=0.0)

            normalize the transition matrix. Enforce the fact that the first row is the start state, and the second row is the end state. 
            return P
          
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
     
     
        MergeStatesRecalculateP(S,P,mergeInds,osT,maxIterBW=1000,nRerunBW=100)

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
                maxIterBW, nRerunBW, parameters for BW algorithm. 
            Return:
                P2 - transition matrix. 
     
     
        generateSequenceSamples(S,P,N,nSample=nSample)

            generegate nSample sets of N sequences from the POMM. 
            Inputs:
                S - state vector
                P - transition matrix
                N - number of sequences in each set.
                nSample - number of sets sampled. 
            Return:
                osSampled - sampled sets of sequences
     
        computePsStatsInSamples(osTSamples,ss,Ps0)  

            compute the Ps of subsequence ss in the sampled seqeunces, return confidence intervals. 
            Inputs:
                osTSamples - sampled sequences
                ss - subsequence
                Ps0 - Ps of the subsequence in the observed set
            Returns:
                pv - p-value of the observed Ps0 being larger than the smapled. 
                pL - lower bound of Ps in 95% confidence interval
                pS - upper bound of Ps in 95% confidence interval   
                pMedian - median value of the distribution
     
     
        computeNumTasksProc(nTot)

            blance load on multiple process, returns arrar of number of computations each process should handle. 
            usefule wen nStask is not multiples of nProc
            Inputs:
                nTot - total number of tasks
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

def NGramPOMMSearch(osIn, pValue=0.05, stateMergeParam=[1, 0.1, 0.1], Pcut=0.001, nSample = 10000, ngramStart = 1, fnSave=''):
 
    print('Constructing POMM with nGram transition diagram...')
    flag = 0
    maxNG = 200
    for ng in range(ngramStart,maxNG):
        print('\nTesting nGram size ng = ',ng)
        
        # construct nGram POMM
        S, P, SnumVis = constructNGramPOMMC(osIn, ng);
        P = normP(P, Pcut = Pcut)
                    
        # test the statsicial signifance. 
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample=nSample)
        print(' Pb sampled range=(',round(PBs.min(),3),round(PBs.max(),3),') seq Pb=', round(PbT,3))
            
        print(' S=',S)

        if pv > pValue:
            print(' Accepted pv=',round(pv,3))
            flag = 1
            break
        else:
            print(' Rejected pv=',round(pv,3))
            
    if flag == 0:
        print('WARNING: in NGramPOMMSearch: no NGram model accepted up until ng=',ng)
        Pc = 0
        return S, P, pv, PBs, PbT
 
    pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample=nSample)
    print(f'Ngram model pv={pv:0.4f}')
    
    # get all unique sequences of length ng
    seqDict, symU = getUniqueSequencesByStartSymbol(osIn, ng+1)


    def mergeStateJJtoII(ii, jj, N, SnumVisTestIn, PTestIn):
        """
        Merge state jj into state ii.

        Correctly handles transitions:
            ii -> jj
            jj -> ii
            jj -> jj
        by folding them into the new self-transition ii -> ii.

        Assumes ii and jj have the same emitted symbol.
        """

        print(f'    merging states {ii}, {jj}')

        SnumVisTest = SnumVisTestIn.copy()
        Pold = PTestIn.copy()
        PTest = PTestIn.copy()

        wi = SnumVisTest[ii]
        wj = SnumVisTest[jj]
        w = wi + wj

        if w > 0:
            ai = wi / w
            aj = wj / w
        else:
            # fallback if both states have zero visitation count
            ai = 0.5
            aj = 0.5

        # 1. Merge outgoing row jj into row ii.
        # For ordinary destination states k not in {ii, jj}.
        for kk in range(N):
            if kk == ii or kk == jj:
                continue

            PTest[ii, kk] = ai * Pold[ii, kk] + aj * Pold[jj, kk]

        # 2. Correct merged self-transition.
        # Any transition from {ii, jj} to {ii, jj} becomes ii -> ii.
        PTest[ii, ii] = (
            ai * (Pold[ii, ii] + Pold[ii, jj])
            + aj * (Pold[jj, ii] + Pold[jj, jj])
        )

        # 3. Merge incoming column jj into column ii.
        # For source states k not in {ii, jj}.
        for kk in range(N):
            if kk == ii or kk == jj:
                continue

            PTest[kk, ii] = Pold[kk, ii] + Pold[kk, jj]
            PTest[kk, jj] = 0.0

        # 4. Remove transitions into jj from ii and jj.
        PTest[ii, jj] = 0.0

        # 5. Deactivate jj.
        # You currently redirect deleted state jj to end state.
        # This is okay if removeUnreachableStates will remove jj later.
        PTest[jj, :] = 0.0
        PTest[:, jj] = 0.0
        PTest[jj, 1] = 1.0

        # 6. Update visitation count.
        SnumVisTest[ii] = wi + wj
        SnumVisTest[jj] = 0

        return SnumVisTest, PTest

    # Testing merging all states with the symbol to find symbols only need one state
    print('Testing merging all states with the same symbol...')
    for sm in symU:

        N = len(S)
        iid = np.where(sm == np.array(S))[0]

        if len(iid) <= 1:
            continue

        print(' ')
        print(' merging all states for sym  = ', sm)

        PTest = P.copy()
        SnumVisTest = SnumVis.copy()

        ii = iid[0]
        for jj in iid[1:]:
            SnumVisTest, PTest = mergeStateJJtoII(ii, jj, N, SnumVisTest, PTest)

        PTest = normP(PTest)

        pv, PBs, PbT = getPVSampledSeqsPOMM(S, PTest, osIn, nSample=nSample)
        print(f'     pv=', pv, ' Pb sampled range=(', round(PBs.min(), 3), round(PBs.max(), 3), ') seq Pb=', round(PbT, 3))

        if pv >= pValue:
            print(' Markov symbol. Mergers accepted. \n')
            S, P, iids = removeUnreachableStates(S, PTest, returniid=True)
            SnumVis = [SnumVisTest[k] for k in iids]
        else:
            print(' Mergers rejected.\n')            

    # clustering and merging states with the same symbol and similary distributions of future sequences. 
    S0 = S.copy()
    SnumVis0 = SnumVis.copy()
    P0 = P.copy()

    mmax, mmin, step = stateMergeParam
    if step <= 0:
        raise ValueError("step must be positive")

    mParams = []
    vv = mmax
    while vv >= mmin - 1e-12:
        mParams.append(vv)
        vv -= step

    for mergeClusterDistThreshod in mParams:

        S = S0.copy()
        SnumVis = SnumVis0.copy()
        P = P0.copy()

        PTest = P.copy()
        SnumVisTest = SnumVis.copy()

        print(f'\nTesting mergeClusterDistThreshod: {mergeClusterDistThreshod:.4f}')

        for sm in symU:
            N = len(S)
            iid = np.where(sm == np.array(S))[0]

            if len(iid) <= 1:
                continue

            print(' ')
            print(' merging states for sym  = ', sm)

            uSeqs = seqDict[sm]
            if len(uSeqs) == 0:
                continue

            Pseqs = np.zeros((len(iid), len(uSeqs)))
            for i, ii in enumerate(iid):
                Pstart = P.copy()
                Pstart[0, :] = 0.0
                Pstart[0, ii] = 1.0

                for j, ss in enumerate(uSeqs):
                    Pseqs[i, j] = computeSequenceProbNoEnd(ss, S, Pstart)

            row_sums = Pseqs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = Pseqs / row_sums

            def tv_dist(u, v):
                return 0.5 * sum(abs(u - v))

            D = pdist(X, metric=tv_dist)
            Z = linkage(D, method='average')
            labels = fcluster(Z, t=mergeClusterDistThreshod, criterion='distance')

            print(" cluster labels =", labels)

            for ll in set(labels):
                kkd = np.where(np.array(labels) == ll)[0]
                if len(kkd) < 2:
                    continue

                ii = iid[kkd[0]]
                for j in range(1, len(kkd)):
                    jj = iid[kkd[j]]
                    SnumVisTest, PTest = mergeStateJJtoII(ii, jj, N, SnumVisTest, PTest)

        PTest = normP(PTest, Pcut=Pcut)
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, PTest, osIn, nSample=nSample)
        print(f'     pv=', pv, ' Pb sampled range=(', round(PBs.min(), 3), round(PBs.max(), 3), ') seq Pb=', round(PbT, 3))

        if pv >= pValue:
            print(' Mergers accepted. ')
            S, P, iids = removeUnreachableStates(S, PTest, returniid=True)
            SnumVis = [SnumVisTest[k] for k in iids]

            break
                   
    pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample=nSample)
    print(f'\nAfter merging sequence prob based merging, pv=', pv, ' Pb sampled range=(', round(PBs.min(), 3), round(PBs.max(), 3), ') seq Pb=', round(PbT, 3),'\n')
     
    # test deleting state through grids.
    #print('Further simplification with state deletion method...') 
    S, P, pv, PBs, PbT, Pc = MinPOMMSimpDeleteStates(S, osIn, nRerun = BWRerun, pValue=pValue, nSample=nSample)
    print(f'    After deleting states pv={pv:0.4f}')

    # final model.     
    print(f'Found model S={S}')
    print(f'pv={pv:.4f}') 
    
    if fnSave != '':
        saveNGramPOMMSearchRes(S,P,ng,pv,PbT,SnumVis,fnSave)

    return S, P, pv, PBs, PbT


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
    
    PS = np.zeros((N,T))                   # probability of the the state sequence at state i time step t. 
    SS = np.zeros((N,T)).astype('int')     # the state selected for time t for state i. 
    PS[0,0] = 1         # starting from the start state 
    
    # forward pass
    for t in range(1,T):
        for jj in range(N):
            if S[jj] != sseq[t]:    # the state does not have the right symbol at this time point. 
                continue    
            imax = np.argmax(PS[:,t-1] * P[:,jj])
            SS[jj,t] = imax
            PS[jj,t] = PS[imax,t-1] * P[imax,jj]
        
    prob = PS[1,T-1]                # this is the maximum probability. 
    stateSeq = [SS[1,T-1]]      # this is the last state before the end state
    for t in range(T-2,1,-1):   # trace back.
        imax = stateSeq[0]
        stateSeq = [SS[imax,t]]+stateSeq
    return stateSeq, prob   
    

lib.computeSeqProbPOMM_CSR.argtypes = [
    ctypes.c_int,                         # N
    ctypes.POINTER(ctypes.c_int),         # S
    ctypes.POINTER(ctypes.c_int),         # row_ptr
    ctypes.POINTER(ctypes.c_int),         # col_ind
    ctypes.POINTER(ctypes.c_double),      # Pval
    ctypes.c_int,                         # ns
    ctypes.POINTER(ctypes.c_int),         # seq
]

lib.computeSeqProbPOMM_CSR.restype = ctypes.c_double

def computeSeqProbPOMMC(N, S, P, seq, PU, ii, Pcut=0.0):
    """
    Compute probability of a single sequence using CSR C function.

    Original dense P is converted to CSR before calling C.
    """

    S = np.asarray(S, dtype=np.int32)
    seq = np.asarray([0] + list(seq) + [-1], dtype=np.int32)
    nSeq = len(seq)
    
    P_csr = sparse.csr_matrix(P, dtype=np.float64)
    # Clean CSR representation
    P_csr.sum_duplicates()
    P_csr.eliminate_zeros()
    P_csr.sort_indices()

    rowPtr = np.ascontiguousarray(P_csr.indptr, dtype=np.int32)
    colInd = np.ascontiguousarray(P_csr.indices, dtype=np.int32)
    val = np.ascontiguousarray(P_csr.data, dtype=np.float64)
    
    prob = lib.computeSeqProbPOMM_CSR(
        ctypes.c_int(N),
        S.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rowPtr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        colInd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nSeq),
        seq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    PU[ii] = prob
    return prob

def computeSeqProbPOMMC_CSR_arrays(N, S, rowPtr, colInd, val, seq, PU=None, ii=None):
    seq = np.asarray([0] + list(seq) + [-1], dtype=np.int32)
    nSeq = len(seq)

    prob = lib.computeSeqProbPOMM_CSR(
        ctypes.c_int(N),
        S.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rowPtr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        colInd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nSeq),
        seq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    if PU is not None and ii is not None:
        PU[ii] = prob

    return prob    

# compute the modified sequence completeness
def computeModifiedSequenceCompleteness(S, P, osT):
    print(' Getting unique sequences...')
    osU, osK, symU = getUniqueSequences(osT)

    print(' Preparing CSR P...')
    N = len(S)

    S = np.ascontiguousarray(S, dtype=np.int32)

    P_csr = sparse.csr_matrix(P, dtype=np.float64)
    P_csr.sum_duplicates()
    P_csr.eliminate_zeros()
    P_csr.sort_indices()

    rowPtr = np.ascontiguousarray(P_csr.indptr, dtype=np.int32)
    colInd = np.ascontiguousarray(P_csr.indices, dtype=np.int32)
    val = np.ascontiguousarray(P_csr.data, dtype=np.float64)

    assert rowPtr.shape[0] == N + 1
    assert colInd.shape[0] == val.shape[0]

    print(' Computing PU...')

    def worker(item):
        i, seq = item

        prob = computeSeqProbPOMMC_CSR_arrays(
            N,
            S,
            rowPtr,
            colInd,
            val,
            seq,
            None,
            None,
        )

        return i, prob

    PU = np.zeros(len(osU), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=nProc) as executor:
        for i, prob in executor.map(worker, enumerate(osU)):
            PU[i] = prob

    print(' Done.')

    Pc = np.sum(PU)

    if Pc < 1e-5:
        return 0.0

    osK = np.asarray(osK, dtype=np.float64)
    PP = osK / np.sum(osK)

    PU = PU / Pc
    dd = 0.5 * np.sum(np.abs(PU - PP))

    Pb = (
        (1 - betaTotalVariationDistance) * Pc
        + betaTotalVariationDistance * (1 - dd)
    )

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


lib.getModifiedSequenceCompletenessSamplingModelCSR_C.argtypes = [
    ctypes.c_int,                          # nSeqs
    ctypes.c_int,                          # N
    ctypes.POINTER(ctypes.c_int),          # S
    ctypes.c_int,                          # nnz
    ctypes.POINTER(ctypes.c_int),          # rowPtr
    ctypes.POINTER(ctypes.c_int),          # colInd
    ctypes.POINTER(ctypes.c_double),       # val
    ctypes.c_int,                          # nSample
    ctypes.POINTER(ctypes.c_double),       # PBs
    ctypes.c_double,                       # beta
    ctypes.c_int,                          # randSeed
]
lib.getModifiedSequenceCompletenessSamplingModelCSR_C.restype = None

def getModifiedSequenceCompletenessSamplingModelC(params):
    S, P, nSeq, nSample = params

    # State-symbol vector
    S = np.ascontiguousarray(S, dtype=np.int32)
    N = len(S)

    # Dense input P -> CSR
    P = np.asarray(P, dtype=np.float64)

    assert P.size == N * N, f"P.size={P.size}, expected {N*N}"
    P = P.reshape((N, N))

    P_csr = sparse.csr_matrix(P, dtype=np.float64)

    # Clean CSR representation
    P_csr.sum_duplicates()
    P_csr.eliminate_zeros()
    P_csr.sort_indices()

    rowPtr = np.ascontiguousarray(P_csr.indptr, dtype=np.int32)
    colInd = np.ascontiguousarray(P_csr.indices, dtype=np.int32)
    val = np.ascontiguousarray(P_csr.data, dtype=np.float64)

    nnz = int(P_csr.nnz)

    assert ctypes.sizeof(ctypes.c_int) == S.dtype.itemsize
    assert ctypes.sizeof(ctypes.c_int) == rowPtr.dtype.itemsize
    assert ctypes.sizeof(ctypes.c_int) == colInd.dtype.itemsize

    assert rowPtr.size == N + 1
    assert colInd.size == nnz
    assert val.size == nnz
    assert nSeq >= 0
    assert nSample >= 0

    PBs = np.zeros(nSample, dtype=np.float64)

    randSeed = int(time.time()) & 0x7fffffff    
    
    lib.getModifiedSequenceCompletenessSamplingModelCSR_C(
        ctypes.c_int(nSeq),
        ctypes.c_int(N),
        S.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(nnz),
        rowPtr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        colInd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nSample),
        PBs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(betaTotalVariationDistance),
        ctypes.c_int(randSeed),
    )

    return PBs
           
                    
def getMaxLenSeqs(osIn):
    maxLenSeqs = 0
    for ss in osIn:
        if len(ss) > maxLenSeqs:
            maxLenSeqs = len(ss)
    return maxLenSeqs
                                        

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
def getPVSampledSeqsPOMM(S, P, osIn, nSample=10000):
    NS = computeNumTasksProc(nSample, nProc)
    N = len(osIn)
    Params = [[S, P, N, NS[ii]] for ii in range(nProc)]

    print("Sampling...")

    with Pool(processes=nProc) as pool:
        res = pool.map(getModifiedSequenceCompletenessSamplingModelC, Params, chunksize=1)

    PBs = [pb for PPBs in res for pb in PPBs]
    PBs = np.sort(PBs)

    print("getting PbT...")
    PbT = computeModifiedSequenceCompleteness(S, P, osIn)
    PbT += 1e-10

    jj = np.searchsorted(PBs, PbT, side='right')
    pv = jj / len(PBs)

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
# Returns:
#   pv, p-value of the observed sequence on the model.
#   D, the distances of the sampled sequences
#   ddT, the distance of the observed sequences. 
def getPVSampledSeqsPOMMTotalVariationDistance(S, P, osIn,nSample = 10000):
    print('Getting p-value using total variation distance...')
    NS = computeNumTasksProc(nSample,nProc)
    N = len(osIn)
    Params = [[S, P, N, NS[ii]] for ii in range(nProc)]

    pool = Pool(processes = nProc)
    res = pool.map(getTotalVariationDistanceSamplingModel,Params,chunksize = 1)
    pool.close()
    pool.join()
    
    DDs = [dd for DDDs in res for dd in DDDs]   
    DDs = np.sort(DDs)[::-1]

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
def generateSequenceSamples(S,P,N,nSample=10000):
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
        ss = np.array(ss)
        
        for subs in Subs:
            flag = 0
            m = len(subs)   
            subs = np.array(subs)      
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
def computePsStatsInSamples(osTSamples,ss,Ps0):

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
        
    PP3 = np.sort(PP3)
    for j3 in range(len(PP3)):
        if PP3[j3] > Ps0+1e-10:
            break
    pv = 1.0 * j3/len(PP3)
    pL = PP3[int(len(PP3)* 0.025)]
    pU = PP3[int(len(PP3)* 0.975)]
    pMedian = np.median(PP3)
                    
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
    P2 = np.zeros((len(S2),len(S2)))
    for (ii,jj) in StateTransitionCounts.keys():
        P2[ii,jj] = StateTransitionCounts[(ii,jj)]
    P2[0,0] = 0         
    P2 = normP(P2)
            
    # reorder the states. 
    P = np.zeros((len(S2),len(S2)))
    iids = np.argsort(S2[2:])
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

lib.constructNGramPOMMC.argtypes = [
    ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]

class ThreeArrays(ctypes.Structure):
    _fields_ = [("N", ctypes.c_int),
                ("S", ctypes.POINTER(ctypes.c_int)),
                ("P", ctypes.POINTER(ctypes.c_double)),
                ("StateNumVis", ctypes.POINTER(ctypes.c_int))
            ]

lib.constructNGramPOMMC.restype = ctypes.POINTER(ThreeArrays)


def constructNGramPOMMC(osIn, ng):
    
    #concatenate the sequences into a long int array. sequence sequence is flanked by 0...-1
    osInC =[]
    for i in range(len(osIn)):
        osInC.append(0)
        osInC += list(osIn[i])
        osInC.append(-1)
    osInC = np.array(osInC).astype(np.int32)
    nSeq = len(osInC)
            
    A = lib.constructNGramPOMMC(
        ctypes.c_int(nSeq), 
        osInC.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(ng)
    )
    
    # transfer the data from C memory to Python memory. 
    N = A.contents.N
    S2 = [A.contents.S[i] for i in range(N)]
    P2 = [A.contents.P[i] for i in range(N * N)]
    P2 = np.array(P2).reshape(N,N)
    StateNumVisits = [A.contents.StateNumVis[i] for i in range(N)]

    # free memory allocated in the C code.  
    lib.freeThreeArrays.argtypes = [ctypes.POINTER(ThreeArrays)]
    lib.freeThreeArrays(A)
    
    # reorder the states. 
    P = np.zeros((len(S2),len(S2)))
    iids = np.argsort(S2[2:])
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
    
        
def AdjustTransProbWithBWKeepConnections(S,P,osIn,nRerun=100,Pcut=0.001):
    N = len(S)
    P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn, Pcut= 0.001, nRerun=nRerun)    
    S, P = removeUnreachableStates(S,P)    
    return S, P, Pc

    
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
    P = np.array(res["P"],dtype="float64")
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

  
        
    
    
"""         
 Simplify by deleting states and making sure that the maximum likelihood remains within bound. 
 Input parameters:
    S - initial POMM
    osIn - observed sequences
    nRerun - number of times B-W is run with different seeds. 
 Return S, P, Pc    
    S - state vector
    P - transition probabilities
    Pc - sequence completeness
""" 
    
def MinPOMMSimpDeleteStates(S, osIn, nRerun = 50, pValue=pValue, nSample=10000, fnSave=''):
    
    syms = list(np.unique(S[2:]))
    
    for sm in syms:
        while True:
            iid = [ii for ii in range(len(S)) if S[ii] == sm]
            if len(iid) <= 1:
                break

            kk = iid[0]
            STest = S.copy()
            STest = STest[:kk] + STest[kk+1:]
            print('\nTest removing state', kk, 'with sym', S[kk])

            PTest, ml, Pc, stdml, ML = BWPOMMCParallel(STest, osIn, nRerun=nRerun)
            pv, PBs, PbT = getPVSampledSeqsPOMM(STest, PTest, osIn, nSample=nSample)

            if pv >= pValue:
                S = STest.copy()
                P = PTest.copy()
                print(' Deletion accepted. pv=', pv, ' S=', S)
            else:
                print(f' Rejected deletion. pv={pv:.4f}')
                break  # no need for further deleting test. 

                                    
    print('Updated state S = ',S)
    P, ml, Pc, stdml, ML = BWPOMMCParallel(S, osIn, nRerun=nRerun)    
    pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample = nSample)
        
    print(f'    After deleting states model pv= {pv}')
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
            iid = np.where(C[:,jj] == 1)[0]
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
    
    
# get unique sequences osU, counts, osK, and unique symbols symU from sequence osIn.
def getUniqueSequences(osIn):
    symU = np.array([])
    osU = []
    osK = []
    for ss in osIn:
        ss = np.array(ss)
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
        symU = np.unique(np.concatenate((symU,ss)))
    symU = symU.astype(int) 
    return osU,osK,symU 
    
def getUniqueSequencesByStartSymbol(osIn, ng):
    """
    Find all unique symbols in osIn.
    For each symbol sm, find unique subsequences that start with sm
    and have maximum length ng.

    Parameters
    ----------
    osIn : list
        List of sequences. Each sequence can be a list or numpy array of ints.
    ng : int
        Maximum subsequence length.

    Returns
    -------
    seqDict : dict
        Dictionary keyed by symbol.
        For each symbol sm:
            seqDict[sm]['osU'] = list of unique subsequences (lists)
            seqDict[sm]['osK'] = counts of those subsequences
    symU : numpy array
        Sorted unique symbols in osIn.
    """

    # collect all unique symbols
    symU = set()
    for ss in osIn:
        if len(ss) > 0:
            symU.update(int(x) for x in ss)
    symU = sorted(symU)    

    # initialize dictionary
    seqDict = {}
    for sm in symU:
        seqDict[sm] = []
        
    # collect unique subsequences by starting symbol
    for ss in osIn:
        L = len(ss)

        for i in range(L):
            sm = int(ss[i])
            if sm == -1:
                break

            # subsequence starting at i, truncated to max length ng
            j = i + ng
            if j > L:
                j = L
                sub = list(ss[i:L])+[-1]
            else:
                sub = list(ss[i:j])            
            if sub not in seqDict[sm]:
                seqDict[sm].append(sub)

    return seqDict, symU
        
    
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
    
# set parameter types   
lib.BWPOMMC.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), \
                        ctypes.c_int, ctypes.POINTER(ctypes.c_int), 
                        ctypes.c_int,\
                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), \
                        ctypes.c_double, ctypes.c_int, ctypes.c_int]  
lib.BWPOMMC.restype = ctypes.c_double

def BWPOMMCFun(Params):
    osU, osK, S, P, pTol, maxIter = Params
    N = len(S)
    S = np.array(S).astype(np.int32)
    P = np.array(P)
    osIn =[]
    for i in range(len(osU)):
        osIn.append(0)
        osIn += list(osU[i])
        osIn.append(-1)
    osIn = np.array(osIn).astype(np.int32)
    osK = np.array(osK).astype(np.int32)
    nU = len(osK)
    randSeed = int(rand() * 100000);
    nSeq = len(osIn)

    t1 = time.time()
    # call the C function.
    ml = lib.BWPOMMC(ctypes.c_int(nSeq), osIn.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), \
                 ctypes.c_int(nU), osK.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), \
                 ctypes.c_int(N),S.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), \
                 P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                 ctypes.c_double(pTol), ctypes.c_int(maxIter), ctypes.c_int(randSeed))
    t2 = time.time()
    #print(f'     BWPOMMC used {t2-t1:.4f} sec')
    
    #ml2 = computeLogLike(S,P,osU,osK)
    #print('ml C=',ml)
    #print('ml P=',ml2)
                    
    return (ml,P)                   

# multi-thread BWPOMMCMultiThread
lib.BWPOMMCMultiThread.argtypes = [
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_double), 
        ctypes.c_double, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int
]   
lib.BWPOMMCMultiThread.restype = ctypes.c_double

def BWPOMMCFunMultiThread(osU, osK, S, P, pTol, maxIter, numThreads):
    N = len(S)
    S = np.array(S).astype(np.int32)
    P = np.array(P)
    osIn =[]
    for i in range(len(osU)):
        osIn.append(0)
        osIn += list(osU[i])
        osIn.append(-1)
    osIn = np.array(osIn).astype(np.int32)
    osK = np.array(osK).astype(np.int32)
    nU = len(osK)
    randSeed = int(rand() * 100000);
    nSeq = len(osIn)
    
    t1 = time.time()
    # call the C function.
    ml = lib.BWPOMMCMultiThread(
            ctypes.c_int(nSeq), 
            osIn.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
            ctypes.c_int(nU), 
            osK.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
            ctypes.c_int(N),
            S.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 
            P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            ctypes.c_double(pTol), 
            ctypes.c_int(maxIter), 
            ctypes.c_int(randSeed), 
            ctypes.c_int(numThreads)
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
#   maxSteps, maximum number of steps for updating the transition probabilities
#   pTol, tolerance for the transition probabilities
#   nRerun, number of times the algorithm is run. 
# Returns:
#   return P, mlMax, Pc, mlSigma, ML   
#   P, computed transition matrix
#   MLmax, maximum log likelihood
#   Pc, sequence completeness of the input sequences on the model
#   stdML, standard deviation of the maximum likelihood achieved for all runs. 
#   MK, list of maximum likelihoods
def BWPOMMCParallel(S, osInO, Pcut=0.0, maxSteps=10000, pTol=1e-6, nRerun=BWRerun):
    osIn = osInO.copy()
    N = len(S)
    S = np.array(S)
    osU, osK, symU = getUniqueSequences(osIn)
    
    Ps = []
    for irun in range(nRerun):
        P = normP(rand(N,N))
        Ps.append([osU,osK,S,P,pTol,maxSteps])

    # parallel conputation of multiple runs. 
    pool = Pool(processes = nProc)
    res = pool.map(BWPOMMCFun,Ps,chunksize = 1)
    pool.close()
    pool.join()
    
    ML = [ml for (ml, P) in res]
    ML = np.array(ML)
    iid = ML.argmax()
    P = res[iid][1]
    Pc, Ps = computeSequenceCompleteness(S,P,osIn,osU)
    mlSigma = np.std(ML)
    mlMax = ML[iid]
    return P, mlMax, Pc, mlSigma, ML    

# Compute log likelihood of the seqeucens given the POMM. 
# S, states
# P, transition matrix
# osU, unique sequences
# osK, counts. 
def computeLogLike(S,P,osU,osK):
    S = np.array(S)
    N = len(S)
    llk = 0
    for kk in range(len(osU)):
        ss = list(osU[kk])
        os = [0] + ss + [-1]
        T = len(os)             # number of steps. 
        # compute alphas
        A = np.zeros((N,T))
        A[0,0] = 1.0
        for t in range(1,T):
            iid = np.where(S == os[t])[0]      # these are the states of allowed transitions. 
            for jj in iid:
                for k in range(N):
                    A[jj,t] += P[k,jj] * A[k,t-1]
        llk += np.log(A[1,T-1]+1e-100) * osK[kk]
    return llk
                        
# normalize the transition matrix. Enforce the fact that the first row is the start state, and the second row is the end state.
def normP(P, Pcut=0.0):
    N = P.shape[0]
    # Handle empty or None Pcut
    if Pcut is None or (hasattr(Pcut, '__len__') and len(Pcut) == 0):
        Pcut = 0.0
    for i in range(N):
        if i == 1:
            continue
        elif i == 0:
            P[i, 1] = 0  # no transition to the end state from the start state
        P[i, 0] = 0      # no transitions to the start state
        # hard threshold small entries
        P[i, P[i, :] < Pcut] = 0
        ss = P[i, :].sum()
        if ss > 0:
            P[i, :] /= ss
            # threshold again after normalization, since renormalization
            # can leave tiny entries below Pcut
            P[i, P[i, :] < Pcut] = 0
            ss = P[i, :].sum()
            if ss > 0:
                P[i, :] /= ss
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
    S = np.array(S)
    N = len(S)
    if len(osU) == 0:
        osU,osK,symU = getUniqueSequences(osIn)
    PU = np.zeros(len(osU))
    for kk in range(len(osU)):
        ss = np.array(osU[kk])
        os = [0] + list(ss) + [-1]
        T = len(os)             # number of steps. 
        # compute alphas
        A = np.zeros((N,T))
        A[0,0] = 1.0
        for t in range(1,T):
            iid = np.where(S == os[t])[0]      # these are the states of allowed transitions. 
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
    S = np.array(S)
    N = len(S)
    if len(osU) == 0:
        osU,osK,symU = getUniqueSequences(osIn)
    Pc = 0
    Ps = []
    for ss in osU:
        ss = np.array(ss)
        os = [0] + list(ss) + [-1]
        T = len(os)             # number of steps. 
        # compute alphas
        A = np.zeros((N,T))
        A[0,0] = 1.0
        for t in range(1,T):
            iid = np.where(S == os[t])[0]      # these are the states of allowed transitions. 
            for jj in iid:
                for k in range(N):
                    A[jj,t] += P[k,jj] * A[k,t-1]
        Pc += A[1,T-1]      # this the probability of observing this unique sequence
        Ps.append(A[1,T-1])
    return Pc, Ps
    
# compute the probability of the sequence given the model
def computeSequenceProb(ss, S, P):
    S = np.array(S)
    N = len(S)

    ss = np.array(ss)
    os = [0] + list(ss) + [-1]
    T = len(os)             # number of steps. 
    # compute alphas
    A = np.zeros((N,T))
    A[0,0] = 1.0
    for t in range(1,T):
        iid = np.where(S == os[t])[0]      # these are the states of allowed transitions. 
        for jj in iid:
            A[jj, t] = np.dot(A[:, t-1], P[:, jj])
    pS = A[1,T-1]                       # this the probability of observing this unique sequence
    return pS       

# compute the probability of the sequence given the model, the sub does not need match the ending. 
def computeSequenceProbNoEnd(ss, S, P):
    S = np.array(S)
    N = len(S)

    ss = np.array(ss)
    os = [0] + list(ss)
    T = len(os)             # number of steps. 
    # compute alphas
    A = np.zeros((N,T))
    A[0,0] = 1.0
    for t in range(1,T):
        iid = np.where(S == os[t])[0]      # these are the states of allowed transitions. 
        for jj in iid:
            A[jj, t] = np.dot(A[:, t-1], P[:, jj])
    pS = sum(A[:,T-1])                       # this the probability of observing this unique sequence
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
        S = np.array(S)
        C = np.zeros(k)    
        for kk in range(N):
            iid = np.where(rand() <= S)[0][0]
            C[iid] += 1
        return C

# This function constructs Markov model 
# osIn, List of input sequences
# syms, symbols in the seuqencs     
# return P, C - counts of transitions
def ConstructMarkovModel(osIn,syms,pcut = 0.0):
    N = len(syms)+2
    C = np.zeros((N,N))    
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
    C[np.where(P < pcut)] = 0
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
    P = np.zeros((N,N))    # transition probability. 
    # select states to be connected to the start state. 
    iid = randomSelectInd(nFanOut,nSyms)+2
    P[0,iid] = 1./len(iid) * np.ones(len(iid))
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
        iid = np.where(P[ind,:] > 0)[0]    
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
    P = np.zeros((N,N))    # transition probability. 
    # select states to be connected to the start state. 
    iid = randomSelectInd(nFanOut,N-2)+2
    P[0,iid] = 1./len(iid) * np.ones(len(iid))
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
        iid = np.where(P[ind,:] > 0)[0]    
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

# Remove states unreachable from the start state 0.
# The returned matrix is the submatrix induced by reachable states.
def removeUnreachableStates(SIn, PIn, epsilon=1e-10, returniid=False):
    P = PIn.copy()
    N0 = len(SIn)
    
    if N0 == 0:
        raise ValueError("SIn must be non-empty")
    if P.shape != (N0, N0):
        raise ValueError("P must have shape (len(SIn), len(SIn))")
    reachable = {0}
    frontier = {0}
    while frontier:
        new_frontier = set()
        for i in frontier:
            js = np.where(P[i, :] > epsilon)[0]
            for j in js:
                if j not in reachable:
                    new_frontier.add(j)
        reachable |= new_frontier
        frontier = new_frontier
    iid = sorted(reachable)
    unreachable = set(range(N0)) - reachable

    # Transfer probability mass going to unreachable states to end state (1)
    if 1 in reachable:
        for i in iid:
            lost_mass = sum(P[i, j] for j in unreachable)
            if lost_mass > epsilon:
                P[i, 1] += lost_mass

    S = [SIn[ii] for ii in iid]
    P2 = P[np.ix_(iid, iid)].copy()
    if not returniid:
        return S, P2
    return S, P2, iid

        
# detete connections with small transition probabilities.   
def deleteTransitionSmallProb(SIn,PIn,Pcut = 0.001, iRemoveUnreachableState = 1):
    P = PIn.copy()
    S = SIn.copy()
    Pend = P[:,1].copy()
    P[np.where(PIn < Pcut)] = 0
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
    if Pcut > 0:
        S,P = deleteTransitionSmallProb(S,P0,Pcut = Pcut)
    else:
        P = P0  
    N = len(S)
    
    # find duplicate states
    SU = np.unique(S[2:])
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
    iid = np.argsort(edgeProb)
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
    cc,bb = np.histogram(PCs,bins=30)
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
    cc,bb = np.histogram(RepeatNums,bins=bbins)
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
    iid = np.array([]).astype(int)
    for j in range(nind):
        while 1:
            ind = int(rand() * ntot)
            if ind == excludeInd:
                continue
            if (len(np.where(iid == ind)[0]) == 0):
                iid = np.append(iid,ind)
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
#   osT - observed sequences. 
# Return:
#   P2 - transition matrix. 
def MergeStatesRecalculateP(S,P,mergeInds,osT,maxIterBW=1000,nRerunBW=100,Pcut=0.0):
    P = normP(P, Pcut=Pcut)
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
    P2 = normP(P2, Pcut=Pcut)
    print('Recalculating the transition probabilities...')
    P2, ml, Pc, stdml, ML = BWPOMMCParallel(S,osT,maxSteps=maxIterBW, nRerun=nRerunBW)
    return P2

# remove unique sequences with probability smaller than pCut. 
def RemoveRareSequences(osIn, pCut = 0.001):
    print('Deleting sequences with probabilty smaller than ',pCut)
    osU, osK, symU = getUniqueSequences(osIn)
    osK = np.asarray(osK, dtype=np.float64)
    osP = osK / np.sum(osK)    
    
    iid = np.where(osP < pCut)[0]
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
def getStepProbability(osT, nSym, nSteps):
    PSteps = np.zeros((nSteps, nSym + 1))

    for ss in osT:
        L = min(len(ss), nSteps)

        for istep in range(L):
            PSteps[istep, ss[istep]] += 1

        if L < nSteps:
            PSteps[L, 0] += 1

    for ii in range(nSteps):
        ssm = np.sum(PSteps[ii, :])
        if ssm > 0:
            PSteps[ii, :] /= ssm
        else:
            PSteps[ii, 0] = 1.0

    return PSteps

# blance load on multiple process, returns arrar of number of computations each process should handle. 
# usefule wen nStask is not multiples of nProc
# Inputs:
#   nTot - total number of tasks
#   nProc - number of processors
#  reurns
#   NS - list of length nProc, number of tasks assinged to each processor
def computeNumTasksProc(nTot, nProc = nProc):
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

    osUR, osKR, symUR = getUniqueSequences(osRef)
    osKR = np.asarray(osKR, dtype=np.float64)
    PR = osKR / np.sum(osKR)
    
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
    yy,xx = np.histogram(LL,bins=Bins)
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
    cc,bb = np.histogram(Ps,bins=20)
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
    plt.text(0.5,ylim[1]/2,txt.format(P = np.median(Ps)))
    
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

    keys = np.array(list(StateCorres21.keys()))
    vals = np.array([StateCorres21[k] for k in keys])

    # A Python dict cannot actually contain duplicate keys, so this is mostly redundant.
    for ii in np.unique(keys):
        if np.sum(keys == ii) > 1:
            print('ERROR in plotTwoPOMMsStateCorrespondences: Duplicating state indices in StateCorres. Exit.')
            exit(1)

    for ii in np.unique(vals):
        if np.sum(vals == ii) > 1:
            print('ERROR in plotTwoPOMMsStateCorrespondences: Duplicating target state indices in StateCorres. Exit.')
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
        SU = np.unique(S[2:])
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
            iid = np.argsort(edgeProb)
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
def constructPOMMsMotifSeqs(motifSeqs,nRerun=100,pValue=0.05,nSample=10000):
    
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
        
        osIn, repeatNumSeqs, symsNumeric, _ = getNumericalSequencesNonRepeat(seqs,labels)       
        Syms2 = {}
        Syms2R = {}
        symsNumeric = list(symsNumeric)
        for ii in range(len(symsNumeric)):
            Syms2[symsNumeric[ii]] = labels[ii]
            Syms2R[labels[ii]] = symsNumeric[ii]
        motifPOMMs[sym]['Syms2'] = Syms2
        motifPOMMs[sym]['Syms2R'] = Syms2R
        #get the POMM. 
        S2, P, pv, PBs, PbT = NGramPOMMSearch(
            osIn,
            pValue=pValue,
            nSample=nSample,
        )
        
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
        if kk % 10 == 0:
            print(' ')
        

# get the POMM with the motif sequences.        
def getMotifPOMM(osIn,motifSyllabels,nRerun=100,pValue=0.05,nSample=10000):
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
    osInM, repeatNumSeqs, symsNumericM, _ = getNumericalSequencesNonRepeat(motifSeqs,motifSyllabels)
    print('Using N-gram method to get the motif level POMM...')

    S, Pm, pv, PBs, PbT = NGramPOMMSearch(
        osInM,
        pValue=pValue,
        nSample=nSample,
    )
        
    Sm = [0,-1]
    for ss in S[2:]:
        Sm.append(motifSyllabels[ss-1])
    return Sm, Pm, motifSeqs, motifSeqsCollect
        
# construct POMM by breaking the sequences into motifs then recombination. 
# Inputs:
#   osIn - sequences non-repeat, numerical sequences
#   motifSyllabels - starting syllables for motifs, including the start syllables from the sequences. 
#   nRerun,pValue,nSample - the usual parameters. 
# Returns S, P, pv, PBs, PbT
#   S - state vector
#   P - transition probabilities.   
#   pv - pValue
#   PBs - sampled Pb
#   PbT - Pb of the original seqeuences. 
def MinPOMMmotif(osIn,motifSyllabels,nRerun=100,pValue=0.05,nSample=10000):

    startSyms = getStartingSyllables(osIn)
    motifSyllabels = list(np.unique(motifSyllabels + startSyms))

    Sm, Pm, motifSeqs, motifSeqsCollect = getMotifPOMM(osIn,motifSyllabels,nRerun=nRerun,pValue=pValue,nSample=nSample)
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
    motifPOMMs = constructPOMMsMotifSeqs(mSeqs,nRerun=nRerun,pValue=pValue,nSample=nSample)

    S = []
    for ii in range(2,len(Sm)):
        SS = motifPOMMs[ii]['S']
        S += SS[2:]
    S = list(np.sort(S))
    S = [0,-1] + S
    print('After merging motif POMMs S=',S) 

    # simplify by deleting states       
    S, P, pv, PBs, PbT, Pc = MinPOMMSimpDeleteStates(S,osIn,nRerun=nRerun,nSample=nSample)          
    print('After state deletion pv=',pv)
    
    return S, P,  pv, PBs, PbT
                
        
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
        PO = np.zeros((N,N))
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
        PO = np.zeros((N,N))
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
        PO = np.zeros((N,N))
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
        PO = np.zeros((N,N))
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
        PO = np.zeros((N,N))
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
    
    PB = np.zeros((len(uniqueSeqsLeft),len(uniqueSeqsRight)))
    MM = np.zeros(len(uniqueSeqsLeft))
    for i in range(len(uniqueSeqsLeft)):
        for j in range(len(uniqueSeqsRight)):
            if (i,j) in transitionsCounts.keys():
                PB[i,j] = transitionsCounts[(i,j)]
                MM[i] += PB[i,j] 
    
    #print('PB:')
    #print(PB)
    PB = np.array(PB,dtype=np.float64)
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
                        
    PB = np.zeros((len(uniqueSeqsLeft),len(uniqueSeqsRight)))
    MM = np.zeros(len(uniqueSeqsLeft))
    for i in range(len(uniqueSeqsLeft)):
        for j in range(len(uniqueSeqsRight)):
            if (i,j) in transitionsCounts.keys():
                PB[i,j] = transitionsCounts[(i,j)]
                MM[i] += PB[i,j] 
    
    PB = np.array(PB,dtype=np.float64)
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
            
            nS = len(np.where(singular_values > cutOff)[0])
        print('     numStates = ',nS)
        for i in range(nS):
            S.append(sym)
    print(' estimated S=',S)
    return S        
                
    
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
                ML = np.array(ML)
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
                S = S0.copy()
                P = P0.copy()
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
                nS = len(np.where(singular_values > cutOff)[0])

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
    sylLabels = list(np.unique(S[2:]))
                        
    osIn, repeatNumSeqs, symsNumeric, _ = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 

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
    syms = list(np.unique(S[2:]))
    
    for sm in syms:
        iid = np.where(sm == np.array(S))[0]
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
        PB = np.zeros((len(iid),M))
        for i in range(len(iid)):
            seqs, probs = results[i]
            for kk in range(len(seqs)):
                ss = tuple(seqs[kk])
                pp = probs[kk]
                jj = nGramInds[ss]
                PB[i,jj] = pp
            PB[i,:] /= np.sum(PB[i,:]) 
        
        for i in range(len(iid)):
            PB[i,:] /= np.sqrt(np.sum(PB[i,:] * PB[i,:]))
        
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
                    cc = 1 - np.sum(PB[i,:] * PB[j,:])
                                    
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
                    iidsToDelete = list(np.sort(iidsToDelete))
                    P = PTest.copy()
                    for jj in iidsToDelete[::-1]:
                        S = S[:jj] + S[jj+1:]
                        P = np.delete(P,jj,axis=0)
                        P = np.delete(P,jj,axis=1)
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
                    logP = logProb + np.log(p)
                    seq2 = seq + [S[jj]]
                    
                    if jj == 1: # this is the end state. 
                        nGramSeqs.append(seq2)
                        nGramProbs.append(np.exp(logP))
                        
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
        nGramProbs.append(np.exp(logProb))

    results[i] = [nGramSeqs,nGramProbs]     
    
    

def testPBnGramSearch():
    
    S, P = getTestModel(iModel = 5)
    
    nSeq = 100
    seqsOrig = generateSequencePOMM(S,P,nSeq)
    sylLabels = list(np.unique(S[2:]))
                        
    osIn, repeatNumSeqs, symsNumeri, _ = getNumericalSequencesNonRepeat(seqsOrig,sylLabels) 
    
    S, P, pv, PBs, PbT = PBnGramSearch(osIn, ngramStart = 2, fnSave='testPBnGramSearch.dat')
        

                    
if __name__ == "__main__":
    
    
    #testPBRankPOMM()
    
    testPBnGramSearch()
