# This code is for analyzing syntax of Bengalese finch songs in the cooling experiments. 
#
# Written by Dezhe Jin, dzj2@psu.edu
# Date Start: 2025-12-10

from POMM import *
import pickle
import os
import POMM
import matplotlib
matplotlib.rcParams['font.family'] = 'Times'

nProc = 2

### common parameters
maxIterBW = 1000			# maximum iterations in BW algorithm
nRerunBW = 100				# number of re-runs of BW algorithm
nSample = 10000				# number of samples drawn from POMM for computing Pc distribution
pValue = 0.05               # pValue for inferring POMM. 

# set random number generator seed
random.seed(datetime.now().timestamp())	

def main():

    dataDir = '.'
    fn = '150mA_u_left_tl.annot_observed_sequences.txt'
    seqs, syllableLabels = getSequences(f'{dataDir}/{fn}')
    
    
    if 1: # learn POMM from sequences

        filenameSave = f'{dataDir}/{fn}.POMM.dat'
        learnPOMM(seqs, syllableLabels, filenameSave)

    if 1: # load the learned POMM and plot

        filenameSave = f'{dataDir}/{fn}.POMM.dat'
        
        S, P, pv, PBs, PbT, osIn, Syms, Syms2 = loadPOMM(filenameSave)            

        S2 = [0,-1]
        for ss in S[2:]:
            S2.append(Syms2[ss]) 
            
        print(f'S2: {S2} pv: {pv: .4f}')

        fnFig = f'{filenameSave}.pdf'
        print(f'Saving transition diagram to {fnFig}')
        plotTransitionDiagram(S2,P,Pcut=0.01,filenamePDF=fnFig, \
                removeUnreachable=False,markedStates=[])    
            

def learnPOMM(seqs, syllableLabels,  filenameSave):

    osIn, repeatNumSeqs, Syms, Syms2 = getNumericalSequencesNonRepeat(seqs, syllableLabels)

    
    print('print inferring POMM using the N-gram method ...')
    S, P, pv, PBs, PbT, Pc = NGramPOMMSearch(osIn,nRerun=nRerunBW,pValue=pValue,nProc=nProc,nSample =nSample)

    savePOMM(filenameSave, S, P, pv, PBs, PbT, osIn, Syms, Syms2)

    # simplify by cutting connections     
    print('Simplifying the connections...')  
    S, P, pv, PBs, PbT  = MinPOMMSimp(S,osIn, minP = 0.01,nProc=nProc,nRerun=nRerunBW,pValue=pValue, nSample=nSample, factors=[0.5])    
    print('After simplification pv=',pv)
                                                    
    savePOMM(filenameSave, S, P, pv, PBs, PbT, osIn, Syms, Syms2)

def savePOMM(filenameSave, S, P, pv, PBs, PbT, osIn, Syms, Syms2):
    print(f"Saving the POMM to {filenameSave}")
    with open(filenameSave, "wb") as f:
        pickle.dump([S, P, pv, PBs, PbT, osIn, Syms, Syms2], f)

def loadPOMM(fn):
    if not os.path.exists(fn):
        print(f'ERROR: The learned model does not exist: {fn}')
        exit(1)

    with open(fn,'rb')as f:
        S, P, pv, PBs, PbT, osIn, Syms, Syms2 = pickle.load(f)

    return S, P, pv, PBs, PbT, osIn, Syms, Syms2 
            
    
def getSequences(filename):
    
    print(f'\nLoading sequeces from {filename}')
    with open(filename,'r') as f:
        dat = f.read()
    dat = dat.strip()
    
    seqs = dat.split('\n')    
    syllableLabels = list(set("".join(seqs)))
    print(f'syllableLabels: {syllableLabels} numSeqs: {len(seqs)}\n') 
    return seqs, syllableLabels

    
if __name__ == "__main__":
    
    main()
