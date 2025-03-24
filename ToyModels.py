"""

 Written by 
    Dezhe Jin
    Department of Physics, Penn State
    dzj2@psu.edu
    
 05/06/2022 - 04/01/2023

 Example POMMS for illustrating the process of inferring POMMs. 
"""

from POMM import *
import pickle
import os
import POMM
import matplotlib
matplotlib.rcParams['font.family'] = 'Times'

ddr = './'  # directory for figures

### common parameters
maxIterBW = 1000            # maximum iterations in BW algorithm
nRerunBW = 100              # number of re-runs of BW algorithm
nSample = 10000             # number of samples drawn from POMM for computing Pc distribution

def main():
    
    # set random number generator seed
    random.seed(datetime.now().timestamp()) 
        
    # define the POMM
    iModel = 4      # 1, two state for each symbol, example used in the paper. 
                    # 2, Markov model. 
                    # 3, probability dependent POMM ACD, ACE (less probable), BCD (less probable), BCE, for Fig.1
                    # 4, Markov model corresponding to the iModel=1, the example in the paper. 
    
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

            
    SO = [0,-1]
    for ii in range(2,len(SS)):
        SO.append(Syms[SS[ii]])
    print('SO = ',SO)
    
    S2 = [0,-1]
    for ss in SO[2:]:
        S2.append(Syms2[ss]) 

    plotTransitionDiagram(S2,PO,Pcut=0.01,filenamePDF=ddr+'exampleGroundTruthPOMM.pdf', \
            removeUnreachable=False,markedStates=[])    
    
    inferringPOMMfromExampleSequences(SO,PO,Syms2, nTot = 60)  # generate nTot sequences from the ground truth model and use the sequences to infer POMM. Ideally the inferred POMM is the same as the ground truth POMM. 
    
    #testMarkovModel(SO, PO, Syms2)
    
    #plotMarkovModelTestRes()
    
    #Seqs, PU = computeUniqueSequencesProbs(SO, PO, Syms2)
    
    #testModelComplexityPc(SO,PO)
    
    #nGramModelIllustration(SO, PO, Syms2)
    
    #MarkovModelIllustration(SO, PO, Syms2)
    
    #AICBICPbetaModelInduction(SO, PO, Syms2)   
    #plotAICBICPbetaRes(Syms2)
    
    #testBetaValues(SO, PO, Syms2)
    
    #plotTestBetaValues()
        

def plotModel(S,P,finamePDF='exampleGroundTruthModel.pdf'):
    # plot the transition diagram
    plotTransitionDiagram(S,P,Pcut=0.01, filenamePDF=ddr+finamePDF, \
            removeUnreachable=False,markedStates=[])
            
def inferringPOMMfromExampleSequences(SO,PO,Syms2,nTot = 60):
    
    nRerunBW = 100
    nSample = 10000

    print('\n generating nTot=',nTot,' sequences from the ground truth model\n')
    osIn = generateSequencePOMM(SO,PO,nTot)
    print('\nGenerated sequence from the ground truth model: ')
    for seq in osIn:
        ss = ''
        for s in seq:
            ss += Syms2[s]
        print(ss)
    print('\n')
            
    print('\n Inferring POMM usinging the method of reducing from an accepted N-gram model...')
    S, P, pv, PBs, PbT, Pc = NGramPOMMSearch(osIn,nRerun=nRerunBW,pValue=pValue,nProc=nProc,nSample =nSample)
                
    # simplify by cutting connections       
    S, P, pv, PBs, PbT  = MinPOMMSimp(S,osIn,minP = 0,nProc=nProc,nRerun=nRerunBW,pValue=pValue, nSample=nSample, factors=[0.5])    
    print('After simplification pv=',pv)

    # plot the P-beta test
    print('Saving the Pbeta distribution')
    plt.hist(PBs,bins=50)
    yl = plt.ylim()
    plt.plot([PbT, PbT],yl,'r')
    plt.title(f'P_beta distribution of sampled sequences, red line pv={pv:.3f}')
    plt.xlabel('P_beta')
    plt.savefig('examplePbetaDist.pdf')
    
    print('Plotting the inferred POMM.')
    S2 = [0,-1]
    for ss in S[2:]:
        S2.append(Syms2[ss]) 
    plotTransitionDiagram(S2,P,Pcut=0.01,filenamePDF=ddr+'exampleInferredPOMM_nTot_'+str(nTot)+'.pdf', \
            removeUnreachable=False,markedStates=[])    
    
    
    # sample sequences from the POMM. 
    seqs = generateSequencePOMM(S,P,nTot)
    print('\nGenerated sequence from the inferred POMM: ')
    for seq in osIn:
        ss = ''
        for s in seq:
            ss += Syms2[s]
        print(ss)
    print('\n')
    
                                                                
        
def testMarkovModel(SO,PO,Syms2):
    
    nSample=10000
    maxHist = 2200
    
    NTot = [10, 30, 60, 90] # number of sequences. 
    iRun = 100          # number of runs. 
    PVsN = []

    for nTot in NTot:
        print('Testing Markov model for nTot=',nTot)
        PVs = []
        for itry in range(iRun):
            # sample observed sequence
            osIn = generateSequencePOMM(SO,PO,nTot)
            if 0:
                for ii in range(len(osIn)):
                    ss = ''
                    for sym in osIn[ii]:
                        ss += Syms2[sym]
                    print(ss)
                    if mod(ii+1,10) == 0:
                        print('\n') 
            # construct Markov model. 
            syms = list(unique(SO[2:]))
            P, S, C = ConstructMarkovModel(osIn,syms,pcut = 0.0)            
            pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample =nSample, nProc=nProc)
            PVs.append(pv)
            
            if itry == 0:
                print('Plotting the tests...')
                S2 = [0, -1]
                for ss in S[2:]:
                    S2 += Syms2[ss]
                plotTransitionDiagram(S2,P,Pcut=0.01,filenamePDF=ddr+'markovModel'+str(nTot)+'.pdf', \
                        removeUnreachable=False,markedStates=[])                
                plt.figure()
                plt.clf()
                plotSequenceCompleteness(PBs,width=0.01,ylimMax=maxHist,xlimlow=0, ticks=[0,0.5,1])
                ylim = plt.ylim()
                plt.plot([PbT,PbT],[0,ylim[1]],color='r')
                plt.text(0.3,400,'p='+str(pv))
                plt.savefig(ddr+'testMarkovModel'+str(nTot)+'.pdf')
            
        print(PVs)
        PVsN.append(PVs)
    
    if 1:
        filenameSave = ddr+'MarkovModelTest.beta_'+str(betaTotalVariationDistance)+'.dat'
        print('Saving the results to ',filenameSave)
        fn = open(filenameSave,'wb')
        pickle.dump([NTot,PVsN],fn)
        fn.close()  
    else:
        print('Not saving the results. ')
    
def plotMarkovModelTestRes():
            
    filenameSave = ddr + 'MarkovModelTest.beta_'+str(betaTotalVariationDistance)+'.dat'
    print('Loading the results from ',filenameSave)
    fn = open(filenameSave,'rb')
    NTot,PVsN = pickle.load(fn)
    fn.close()  
                    
    plt.figure()
    for ii in range(len(NTot)):
        plt.subplot(len(NTot),1,ii+1)
        plt.hist(PVsN[ii])
        plt.xlim([0,1])
        print('nTot=',NTot[ii], ' PV Mean = ', mean(PVsN[ii]),' std=', std(PVsN[ii]))
    plt.show()              
                    
            
# list unique sequences proabilities            
def computeUniqueSequencesProbs(SO, PO, Syms2):
    osIn = generateSequencePOMM(SO,PO,200)              
    osU, PU = getSequenceProbModel(SO,PO,osIn)  
    Seqs =[]
    for ii in range(len(PU)):
        SS = ''
        for jj in range(len(osU[ii])):
            SS += Syms2[int(osU[ii][jj])]
        print(SS,' P = ',round(PU[ii],3))
        Seqs.append(SS)
    return Seqs, PU
        

# test changes of Pc as the model complexity increases. 
def testModelComplexityPc(SO,PO):
    
    N = 10
    osIn = generateSequencePOMM(SO,PO,N)
    syms = unique(SO[2:])
    
    PCs = []
    for ns in range(1,5):
        S = [0,-1]
        for ss in syms:
            for i in range(ns):
                S.append(ss)
        print('S=',S)
        P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,maxSteps=maxIterBW,nProc=nProc,nRerun=nRerunBW)
        PCs.append(Pc)
    plt.figure()
    plt.plot(list(range(1,5)),PCs,'o')
    plt.plot(list(range(1,5)),PCs)
    plt.show()
    
def printSequences2(osIn, Syms2):
    kk = 0
    Seqs =[]
    for seq in osIn:
        ssq = ''
        for ss in seq:
            ssq += Syms2[ss]
        print(ssq)  
        Seqs.append(ssq)    
        kk += 1
        if mod(kk,10) == 0:
            print(' ')
    return Seqs

# illustrate the construction of n-gram model.  
def nGramModelIllustration(SO,PO, Syms2):

    ng = 2      # n-gram length 
    nSeq = 90
    osIn = generateSequencePOMM(SO,PO,nSeq)
    
    syllableLabels =[Syms2[ss] for ss in unique(SO[2:])]
    print('Syllable Labels = ',syllableLabels)
    
    Seqs = printSequences2(osIn, Syms2)
    
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

    # print the 2-gram states
    print(str(ng)+'-gram StateID StateSym')
    StateVectListReOrdered = [StateVecList[ii] for ii in iids]
    Syms2[0] = '0'
    Syms2[-1] = '-1'
    for ii in range(len(StateVectListReOrdered)):
        vec = StateVectListReOrdered[ii]
        sst = ''
        for kk in range(len(vec)):
            sst += Syms2[vec[kk]]           
        #print('State Id: ',ii, '   ',str(ng)+'-gam: ', sst, ' Sym: ',Syms2[S[ii]],'    vec=',vec)
        print(sst,' ',ii,'  ',Syms2[S[ii]])
    
    if 1:
            
        # plot the transition diagram
        S2 = ['S','E']
        for ss in S[2:]:
            S2.append(Syms2[ss])
        fn = ddr+'example'+str(ng)+'GramPOMM.ps'
        print('Saving '+str(ng)+'-gam POMM of the example to ',fn)
        plotTransitionDiagram(S2,P,filenamePDF=fn,labelStates=1)
        
        # test pv. 
        pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample =10000, nProc=nProc)
        print('pv = ',pv)
        
def sortPrintUniqueSequences(Seqs,pU):

    # Getting indices of the sorted list
    iid = [index for index, value in sorted(enumerate(Seqs), key=lambda x: x[1])]
    for ii in iid:
        print(Seqs[ii],' ',round(pU[ii],3))

# test a specific Markov model
def MarkovModelIllustration(SO, PO, Syms2): 

    nTot = 30
    print('Testing Markov model for nTot=',nTot)
            
    # sample observed sequence
    print('Original model...')
    osIn = generateSequencePOMM(SO,PO,nTot)
    print('Generated sequences: ')
    Seqs = printSequences2(osIn, Syms2) 
    print(' ')
    osU, osK, symU = getUniqueSequences(osIn)
    pU = [round(osK[i]/sum(osK),3) for i in range(len(osK))]
    print('Unique sequences')
    Seqs = printSequences2(osU, Syms2)
    print('Unique sequences')
    sortPrintUniqueSequences(Seqs,pU)
        
    # construct Markov model. 
    syms = list(unique(SO[2:]))
    P, S, C = ConstructMarkovModel(osIn,syms,pcut = 0.0)    
    print(' ')
    print('All unique sequences of the constrcuted model...')
    Seqs, PU = computeUniqueSequencesProbs(S, P, Syms2)
    print('Sorted')
    sortPrintUniqueSequences(Seqs,PU)

    osU, PU = getSequenceProbModel(S, P, osIn, osU)
    print(' ')
    print('Probabilities of the obsevrved unique sequences on the model')
    Seqs = printSequences2(osU, Syms2)
    print('Probabilities of Unique sequences given the model')
    sortPrintUniqueSequences(Seqs,PU)
    
    Pc =  sum(PU)       # this is the sequence completeness
    PP = osK/sum(osK)   # this is the emperical transition probabilities. 
    PU = PU/Pc          # normalize the transition probabilities of the observed sequences on the model.    
    dd = 0.5 * sum(abs(PU - PP))    
    Pb = (1 - betaTotalVariationDistance) * Pc + betaTotalVariationDistance * (1 - dd) 
    print('Pc=',round(Pc,3),' dd=',round(dd,3),' Pb=',round(Pb,3))
    
    osGen = generateSequencePOMM(S, P, nTot)
    print(' ')
    print('Markov model...')
    Seqs = printSequences2(osGen, Syms2)
    print(' ')
    osU, osK, symU = getUniqueSequences(osGen)
    pU = [round(osK[i]/sum(osK),3) for i in range(len(osK))]
    print('Unique sequences')
    Seqs = printSequences2(osU, Syms2)
    print('Probabilities of the unique sequences saampled from the model...')
    sortPrintUniqueSequences(Seqs,pU)
        
    osU, PU = getSequenceProbModel(S, P, osIn, osU)
    Pc =  sum(PU)       # this is the sequence completeness
    PP = osK/sum(osK)   # this is the emperical transition probabilities. 
    PU = PU/Pc          # normalize the transition probabilities of the observed sequences on the model.    
    dd = 0.5 * sum(abs(PU - PP))    
    Pb = (1 - betaTotalVariationDistance) * Pc + betaTotalVariationDistance * (1 - dd) 
    print('Pc=',round(Pc,3),' dd=',round(dd,3),' Pb=',round(Pb,3))

# use AIC and BIC criteria for selecting POMM for the toy model.            
def AICBICPbetaModelInduction(SO, PO, Syms2):

    filenameSave = ddr+'AICBICPbetaRes.dat' 
    
        
    NTot = [10,30,60,90]    # number of sequences. 
    iRun = 100              # number of runs. 
    nSyms = len(unique(SO[2:]))
        
    print('NumSyms = ',nSyms)
    
    # create list of all states with maximum number of states nMax
    nMax = 2    # maximum numbe of states 
    nStates = [1 for i in range(nSyms)]
    NS = [nStates]
    iStart = 0
    flag = 0
    while 1:
        NS2 = []
        for ns in NS[iStart:]:
            for i in range(nSyms):
                n1 = list(ns)
                if n1[i] < nMax:
                    n1[i] += 1
                    if n1 not in NS2:
                        NS2.append(n1)
                        if min(n1) == nMax:
                            flag = 1
                            break
            if flag == 1:
                break               
        iStart = len(NS)
        NS += NS2
        if flag == 1:
            break
    for ns in NS:
        print(ns)
        
    numGrids = len(NS)
    SS = []
    for ns in NS:
        S = []
        for i in range(nSyms):
            for j in range(ns[i]):
                S.append(i+1)
        S = [0,-1]+S
        SS.append(S)
        
    if os.path.exists(filenameSave):
        fn = open(filenameSave,'rb')
        print('Reading previous run from ',filenameSave)
        XX, NStatesAIC, SSAIC, PPAIC, NStatesBIC, SSBIC, PPBIC, NStatesPbeta, SSPbeta, PPPbeta, SO, PO, Syms2 = pickle.load(fn)
        fn.close()
    else:   
        NStatesAIC = []
        NStatesBIC = []
        NStatesPbeta = []
        XX = []
        SSAIC = []
        PPAIC = []
        SSBIC = []
        PPBIC = []
        SSPbeta = []
        PPPbeta = []
        
    for nTot in NTot:
        print(' ')
        print('Selecing models for nTot=',nTot)
        for itry in range(iRun):
            # sample observed sequence
            osIn = generateSequencePOMM(SO,PO,nTot)
        
            print(' Run num = ',itry)   
            # go through the grids and compute AIC and BIC
            AICSC = []
            BICSC = []
            PbetaPVs = []
            
            PPs = []
            for S in SS:
                #print('        Testing S=',S)
                N = len(S)
                # number of parameters
                K = (N-2)*(N-2) + 2 * (N-2) - N     # number of non-zero elements in the transition matrix. Take into account the transition prorbabilities normalize.   
                #print('            num parameters K = ',K)
                P, ml, Pc, stdml, ML = BWPOMMCParallel(S,osIn,nProc=nProc,nRerun=nRerunBW,maxSteps=maxIterBW)
                #print('            maximum log likelihood = ',ml)
                PPs.append(P)

                # AIC
                AIC = 2*K - 2*ml
                AICSC.append(AIC)
                #print('            AIC = ',AIC)

                # BIC
                BIC = log(nTot)*K - 2*ml
                BICSC.append(BIC)
                #print('            BIC = ',BIC)
                
                #Pbeta
                pv, PBs, PbT = getPVSampledSeqsPOMM(S, P, osIn, nSample = nSample, nProc=nProc)
                PbetaPVs.append(pv)
                #print('            Pbeta pv=',pv)          

            iid = argmin(AICSC)
            S = SS[iid]
            P = PPs[iid]
            print('     AIC Selected model: ',S)
            NStatesAIC.append(len(S)-2)         
            SSAIC.append(S)
            PPAIC.append(P)

            iid = argmin(BICSC)
            S = SS[iid]
            P = PPs[iid]
            print('     BIC Selected model: ',S)
            NStatesBIC.append(len(S)-2)
            SSBIC.append(S)
            PPBIC.append(P)
            
            iids = where(array(PbetaPVs) >= 0.05)[0]
            Ns = [len(SS[ii]) for ii in iids]
            jj = argmin(Ns)
            iid = iids[jj]
            S = SS[iid]
            P = PPs[iid]            
            print('     Pbeta Setected model: ',S)
            NStatesPbeta.append(len(S)-2)
            SSPbeta.append(S)
            PPPbeta.append(P)

            XX.append(nTot)

            print('Saving the results to ',filenameSave)
            fn = open(filenameSave,'wb')
            pickle.dump([XX,NStatesAIC,SSAIC,PPAIC,NStatesBIC,SSBIC,PPBIC,NStatesPbeta,SSPbeta,PPPbeta,SO,PO,Syms2],fn)
            fn.close()  
                            
def plotAICBICPbetaRes(Syms2):
        
    filenameSave = ddr+'AICBICPbetaRes.dat' 

    print('Loading the results from ',filenameSave)
    fn = open(filenameSave,'rb')
    [XX,NStatesAIC,SSAIC,PPAIC,NStatesBIC,SSBIC,PPBIC,NStatesPbeta,SSPbeta,PPPbeta,SO,PO,Syms2] = pickle.load(fn)
    fn.close()      
    

    NTot = unique(XX)
    XX = array(XX)
    print(XX)
    iids = where(XX == NTot[0])[0]

    minNum = len(Syms2.keys())-1
    maxNum = len(SO)-1
    histMax = len(iids)
    MM = len(NTot)
    NN = 3
    
    fig, axs = plt.subplots(NN, MM, figsize=(5, 3)) 

    for ii in range(len(NTot)):
        
        nTot = NTot[ii]

        iids = where(XX == nTot)[0]
        
        for iiter in range(3):
            if iiter == 0:
                NS = array([NStatesPbeta[kk] for kk in iids])
                SSS = [SSPbeta[kk] for kk in iids]
                PPP = [PPPbeta[kk] for kk in iids]              
            elif iiter == 1:        
                NS = array([NStatesAIC[kk] for kk in iids])
                SSS = [SSAIC[kk] for kk in iids]
                PPP = [PPAIC[kk] for kk in iids]
            else:
                NS = array([NStatesBIC[kk] for kk in iids])
                SSS = [SSBIC[kk] for kk in iids]
                PPP = [PPBIC[kk] for kk in iids]
                    
            ax = axs[iiter,ii]
            ax.set_xlim([minNum-0.3,maxNum+0.3])
            bb = list(range(maxNum))
            cc = [0 for i in range(len(bb))]
            for kk in range(len(bb)):
                cc[kk] = len(where(NS == kk)[0])
            print('nTot=',nTot)
            print(' numStates=',bb)
            print(' counts   =',cc)
            print(' ')
            ax.bar(bb,cc,width=0.3,color='gray')
            ax.axis('off')
            ax.plot([minNum-0.3,maxNum+0.3],[0,0],color='gray')
            ax.set_ylim([0,histMax])
            ylim = ax.get_ylim()
            if iiter == NN-1:
                for kk in range(minNum,maxNum+1):
                    ax.text(kk,-0.2*ylim[1],str(kk),horizontalalignment='center',fontsize=7)
                    ax.text((minNum+maxNum)/2,-0.4*ylim[1],'Number of States',horizontalalignment='center',fontsize=7)
            if iiter == 0:
                ax.set_title('N='+str(nTot),fontsize=7)     
    
    plt.tight_layout()          
    plt.savefig(ddr+'AICBICPbetaNumStates.pdf')

def testBetaValues(SO, PO, Syms2):

    filenameSave = ddr+'POMMTestingBetaChoiceRes.dat'   
    
    NTot = [10,30,90]       # number of sequences
    nRuns = 100             # number of runs. 
    nSyms = len(unique(SO[2:]))
    
    BetaValues = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    
    print('NumSyms = ',nSyms)

    # sample sequences from the model

    if os.path.exists(filenameSave):
        fn = open(filenameSave,'rb')
        print('Reading previous run from ',filenameSave)
        XX, NStates, SS, PP, Beta = pickle.load(fn)
        fn.close()
    else:   
        NStates = []
        XX = []
        SS = []
        PP = []
        Beta = []
    
    for nTot in NTot:       
        for beta  in BetaValues:    
            POMM.betaTotalVariationDistance = beta
            print('\n\nTesting nTot=',nTot,' beta = ',beta)

            for iRun in range(nRuns):
                print('\nAt run iRun=',iRun)
                osIn = generateSequencePOMM(SO,PO,nTot)
                
                print('N-gram method ...')
                S, P, pv, PBs, PbT, Pc = NGramPOMMSearch(osIn,nRerun=nRerunBW,pValue=pValue,nProc=nProc,nSample =nSample)
                
                # simplify by cutting connections       
                #S, P, pv, PBs, PbT  = MinPOMMSimp(S,osIn,minP = 0,nProc=nProc,nRerun=nRerunBW,pValue=pValue, nSample=nSample, factors=[0.5])   
                #print('After simplification pv=',pv)
                
                print(' S=',S)
                print(' Num States = ',len(S)-2)
                                                    
                NStates.append(len(S)-2)
                XX.append(nTot)
                SS.append(S)
                PP.append(P)
                Beta.append(beta)

                print('Saving the results to ',filenameSave)
                fn = open(filenameSave,'wb')
                pickle.dump([XX,NStates,SS,PP,Beta],fn)
                fn.close()  

# plot testing          
def plotTestBetaValues():
    
    filenameSave = ddr+'POMMTestingBetaChoiceRes.dat'   
    
    fn = open(filenameSave,'rb')
    print('Reading results from ',filenameSave)
    XX, NStates, SS, PP, Beta = pickle.load(fn)
    NTot = unique(XX)
    XX = array(XX)
    Beta = array(Beta)
    NStates = array(NStates)
    NSmin = int(NStates.min())
    NSmax = int(NStates.max())
    
    ccs = ['c','m','k']

    plt.figure(figsize=(5,3))
    kk = 0
    for nTot in NTot:
        iids = where(XX == nTot)[0]
        BBs = Beta[iids]
        NNs = NStates[iids]
        Medians = []
        Ranges = [[],[]]
        Bs = unique(BBs)
        for beta in Bs:
            iids2 = where(BBs == beta)
            nns = NNs[iids2]
            mm = median(nns)
            Medians.append(mm)
            Ranges[0].append(mm-min(nns))
            Ranges[1].append(max(nns)-mm)
        #plt.errorbar(Bs, Means, yerr=Ranges, fmt='o', ecolor='black', capsize=5, linestyle='-', color='black')
        ikk = int(mod(kk,len(ccs)))
        cc = ccs[ikk]
        kk += 1
        plt.errorbar(Bs, Medians, fmt='o', ecolor=cc, capsize=1, linestyle='-', color=cc,linewidth=0.5)
    
    yt = [k for k in range(NSmin,NSmax+2)]
    plt.yticks(yt,fontsize=7)
    xt = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(xt,fontsize=7)

    plt.xlabel(r'$\beta$',fontsize=7)
    plt.ylabel('Num States',fontsize=7)     
    fname = ddr+'POMMTestingBetaChoiceRes.pdf'
    print('Saving figurre to ',fname)
    plt.tight_layout()          
    plt.savefig(fname)

        
        
if __name__ == "__main__":
    
    main()
    
    
                
