/*
Implementing libPOMM.h
*/

#include "libPOMM.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h> // For DBL_MAX
#include <stddef.h> // For size_t
#include <pthread.h> //multi thread

// A simple implementation of the logsumexp function.
double logsumexp(const double* x, size_t n) {
    double max_val = -DBL_MAX;
    double sum = 0.0;

    // First, find the maximum value in the input array.
    for (size_t i = 0; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // If the max value is extremely negative, return it to avoid underflow.
    if (max_val == -DBL_MAX) {
        return max_val;
    }

    // Compute the sum of the exponentials of the input elements, scaled by the max value.
    for (size_t i = 0; i < n; ++i) {
        sum += exp(x[i] - max_val);
    }

    // Return the log of the computed sum, adjusted by the max value.
    return max_val + log(sum);
}


int getIndex(int i, int j, int N) {
	return i*N + j;
}

//index of N x M matrix. 
int getIndex2(int i,int j,int N,int M)
{
	return i*M+j;
}


//normalize the transition matrix. The first row is the start state, the second row is the end state. 
void norm(int N, double *P) {
	int i, j;
	double sum;
	P[getIndex(0,1,N)] = 0.0;	//no transition to the end state from the start state. 
	for (i=0; i<N; i++) {
		if (i==1) {	//end state, no transitions
			for (j=0; j<N; j++) {
				P[getIndex(i,j,N)] = 0;
			}
		} else {
			P[getIndex(i,0,N)] = 0.0;	//no transition to the start state. 
			sum = 0.0;
			for (j=0; j<N; j++) {
				sum += P[getIndex(i,j,N)];
			}
			if (sum > 0.0) {
				for (j=0; j<N; j++) {
					P[getIndex(i,j,N)] /= sum;
				}
			}
		}
	}
}

int getMaxSeqLen(int nSeq,int *osIn)
{
	int i, sl, maxSL, lb;
	maxSL = 0;
	sl = 0;
	for (i=0; i<nSeq; i++) {
		lb = osIn[i];
		if (lb == 0) {
			sl = 1;		//start sequence. 
		} else if (lb == -1) {
			sl += 1;	//end sequence. 
			if (sl > maxSL) maxSL = sl;
		} else {
			sl += 1;
		}
	}
	return maxSL;
}

//set all matrix elements to zero. 
void zeros(int n, int m, double *M) 
{
	int i;
	for (i=0; i<n*m; i++) M[i] = 0.0;
}

//set all matrix elements to -DBL_MAX. 
void logZeros(int n, int m, double *M) 
{
	int i;
	for (i=0; i<n*m; i++) M[i] = -DBL_MAX;
}



//print transition matrix. 
void PrintTransitionMatrix(int N, double * P)
{
	int i,j;
	printf("\nTransition Probabilities N=%d\n",N);
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			printf("%5.2f ",P[getIndex(i,j,N)]);
		}
		printf("\n");
	} 
}

//get unique sequences and the counts. 
void GetUniqueSequences(int nSeq, int *osIn, int *nSeqU, int *osU, int *nUnique, int *osK)
{
	int i,j,k,iU,nU,maxSL,sl,lb,nSU,flag,flag1;
	int *os;

	maxSL = getMaxSeqLen(nSeq,osIn);
	os = (int *) malloc(maxSL * sizeof(int));
	nU = 0;
	nSU = 0;
	for (k=0; k<nSeq; k++) osK[k] = 0;
	for (k=0; k<nSeq; k++) { 
		lb = osIn[k];
		if (lb == 0) {	// start sequence. 
			os[0] = 0;
			sl = 1;
		} else if (lb == -1) {// end sequence
			os[sl++] = -1;
			//check if this sequence has been observed. 
			iU = 0;
			j = 0;
			flag1 = 0;
			while (j<nSU) {
				if (osU[j] == 0) { //this is the start. now compare. 
					flag = 1;
					for (i=0; i<sl; i++) {
						if (osU[j+i] != os[i]) {
							flag = 0;
							break;
						}
					}
					if (flag == 1) {//found match. 
						osK[iU] += 1;
						flag1 = 1;
						break;
					}
					iU++;
				}
				j += i;
			}
			if (flag1 == 0) {//not found any match. Append. 
				nU += 1;
				for (i=0; i<sl; i++) {
					osU[nSU+i] = os[i];
				}
				nSU += sl;
			}	
		} else {
			os[sl++] = lb;
		}
	}
	free(os);
	(*nUnique) = nU;
	(*nSeqU) = nSU;
}

double BWPOMMC(int nSeq, int *osIn, int nU, int *osK, int N, int *stateSyms, double *P, double pTol, int maxIter, int randSeed)
{
	int i, j, iid, maxSL, sl, lb, istep, kk, ik, T, t, iU;
	double A0,mmax;
	double *logA, *logB, *PO, *x;
	int *os;
	double llk = 0.0;	//log-likelihood of the seqeunces. 
	double minP = 1e-10;	//minimum transition probability. 

	//set initial transition probabilities. 
	if (randSeed != -1) {
		//srand(time(NULL));
		srand((unsigned int)randSeed);	
	}	
	
	//normalize the transition probability. Enforce the properties of the start and end states.
	norm(N,P);
		
	//get the maximum sequence length. 
	maxSL = getMaxSeqLen(nSeq,osIn);
	
	//allocate memory. 
	logA = (double *) malloc(maxSL * N * sizeof(double));
	logB = (double *) malloc(maxSL * N * sizeof(double));
	PO = (double *) malloc(N * N * sizeof(double));
	os = (int *) malloc(maxSL * sizeof(int));
	if (maxSL > N) {
		x = (double *) malloc(maxSL * sizeof(double));
	} else {
		x = (double *) malloc(N * sizeof(double));
	}			
	
	for (istep=0; istep < maxIter; istep++) {
		zeros(N,N,PO);
		kk = 0;
		T = 0;
		iU = 0;
		llk = 0.0;
		while (kk < nSeq) {
			for (ik=kk; ik<nSeq; ik++) { // unpack the unique seqeunces in osIn
				lb = osIn[ik];
				if (lb == 0) {	// start sequence. 
					os[0] = 0;
					T = 1;
				} else if (lb == -1) {// end sequence
					os[T++] = -1;
					break;
				} else {
					os[T++] = lb;
				}
			}
			if (lb != -1) break;	// sequence did not end with -1. 
			kk += T;
			
			//compute alphas
			logZeros(N,T,logA);
			logA[0] = 0.0;
			for (t=1; t<T; t++) {
				for (i=0; i<N; i++) { //log sum 
					if (stateSyms[i] == os[t]) {//allowed transition, destination.  
						for (j=0; j < N; j++) {
							x[j] = logA[getIndex2(j,t-1,N,T)] + log(P[getIndex(j,i,N)] + minP);
						}
						logA[getIndex2(i,t,N,T)] = logsumexp(x, N);	
					}
				}
			}
			
			// compute betas
			logZeros(N,T,logB);
			logB[getIndex2(1,T-1,N,T)] = 0.0; 
			for (t=T-2; t>-1; t--) {
				for (i=0; i<N; i++) {
					if (stateSyms[i] == os[t]) {//allowed from transition, source.  
						for (j=0; j < N; j++) {
							x[j] = log(P[getIndex(i,j,N)] + minP) + logB[getIndex2(j,t+1,N,T)];
						}
						logB[getIndex2(i,t,N,T)] = logsumexp(x, N);	
					}
				}
			}
					
			// update transition probabilities. 
			A0 = exp(logA[getIndex2(1,T-1,N,T)]); 
			if ( A0 == 0) continue; //no update due to the denominator being zero. 
			for (i=0; i<N; i++) {
				for (j=0; j<N; j++) {					
					for (t=0; t<T-1; t++) {
						x[t] = logA[getIndex2(i,t,N,T)] + logB[getIndex2(j,t+1,N,T)] +  log(P[getIndex(i,j,N)] + minP);
					}
					PO[getIndex(i,j,N)] += osK[iU] * exp(logsumexp(x,T-1))/A0;
				}
			}

			llk += logA[getIndex2(1,T-1,N,T)] * osK[iU] ;	//add to log-likelihood. 
			
			iU++;
		}
		// compute the new transition probability.
		norm(N,PO);

		mmax = 0.0;
		for (i=0; i< N*N; i++) {
			if (fabs(PO[i] - P[i]) > mmax) mmax = fabs(PO[i] - P[i]);
		} 
		if (mmax < pTol) break;
		for (i=0; i< N*N; i++) P[i] = PO[i];
		
	}	
	free(logA);
	free(logB);	
	free(PO);
	free(os);
	free(x);
	
	return llk;
}


void* computePOLogLikehood(void *arg)
{
	
	BWData *data = (BWData *)arg;
	
	int N = data->N;
	int T = data->T;
	int	*osK;
	double *logA; 
	double *logB;
	double *x;
	osK = data->osK; 
	logA = data->logA;
	logB = data->logB;
	x = data->x;
	
	double *P;
	int *stateSyms, *os, *osIn;
	stateSyms = data->stateSyms;
	os = data->os;
	osIn = data->osIn;
	P = data->P;	
	
	//initialize. 	
	int t, i, j, ik, lb;
	for (i=0; i<N*N; i++) (data->PO)[i] = 0.0;
	data->llk = 0.0;

	int kk = data->kkStart;
	int iU = data->iUStart;
	
	while (kk < data->kkEnd) {
		for (ik=kk; ik<data->kkEnd; ik++) { // unpack the unique seqeunces in osIn
			lb = osIn[ik];
			if (lb == 0) {	// start sequence. 
				os[0] = 0;
				T = 1;
			} else if (lb == -1) {// end sequence
				os[T++] = -1;
				kk += T;	
				break;			
			} else {
				os[T++] = lb;
			}
		}
		if (lb != -1) break;	// sequence did not end with -1.
		 
		//compute alphas
		logZeros(N,T,logA);
		logA[0] = 0.0;
		double minP = 1e-10;
		for (t=1; t<T; t++) {
			for (i=0; i<N; i++) { //log sum 
				if (stateSyms[i] == os[t]) {//allowed transition, destination.  
					for (j=0; j < N; j++) {
						x[j] = logA[getIndex2(j,t-1,N,T)] + log(P[getIndex(j,i,N)] + minP);
					}
					logA[getIndex2(i,t,N,T)] = logsumexp(x, N);	
				}
			}
		}
		
		// compute betas
		logZeros(N,T,logB);
		logB[getIndex2(1,T-1,N,T)] = 0.0; 
		for (t=T-2; t>-1; t--) {
			for (i=0; i<N; i++) {
				if (stateSyms[i] == os[t]) {//allowed from transition, source.  
					for (j=0; j < N; j++) {
						x[j] = log(P[getIndex(i,j,N)] + minP) + logB[getIndex2(j,t+1,N,T)];
					}
					logB[getIndex2(i,t,N,T)] = logsumexp(x, N);	
				}
			}
		}
				
		// update transition probabilities. 
		double A0 = exp(logA[getIndex2(1,T-1,N,T)]); 
		if ( A0 == 0) {//no update due to the denominator being zero. 
		} else {
			for (i=0; i<N; i++) {
				for (j=0; j<N; j++) {					
					for (t=0; t<T-1; t++) {
						x[t] = logA[getIndex2(i,t,N,T)] + logB[getIndex2(j,t+1,N,T)] +  log(P[getIndex(i,j,N)] + minP);
					}
					(data->PO)[getIndex(i,j,N)] += data->osK[iU] * exp(logsumexp(x,T-1))/A0;
				}
			}
		}	
		//log-likelihood
		data->llk += logA[getIndex2(1,T-1,N,T)] * data->osK[iU] ;	

		++iU;
	}
	
	return NULL;
}


/*
Multi-thread version of BWPOMMC. 
*/ 
double BWPOMMCMultiThread(int nSeq, int *osIn, int nU, int *osK, int N, int *stateSyms, double *P, double pTol, int maxIter, int randSeed, int numThreads)
{

	int i, j, iid, rc, maxSL, sl, lb, istep, kk, ik, T, t, iU;
	double *PO;
	double llk = 0.0;	//log-likelihood of the seqeunces. 
	double mmax, seqDiv; 
	int indStarts[numThreads], indEnds[numThreads], iUStarts[numThreads];
	
	//get max lengths, start and end indices for each thread. 
	maxSL = 0;
	sl = 0;
	ik = 0;
	seqDiv = nSeq/numThreads;
	if (nU <= numThreads) {
		seqDiv = 1;
		numThreads = nU;
	}
	int flag = 1;
	iU = 0;
	for (i=0; i<nSeq; i++) {
		lb = osIn[i];
		if (lb == 0) {
			sl = 1;		//start sequence. 
			if (flag == 1) {
				indStarts[ik] = i;
				iUStarts[ik] = iU;
				flag = 0;				
			}
		} else if (lb == -1) {
			sl += 1;	//end sequence. 
			if (sl > maxSL) maxSL = sl;
			++iU;
			if (seqDiv * (ik+1) <= i || i == nSeq-1) {
				indEnds[ik] = i+1;
				flag = 1;
				++ik;		
			}	
		} else {
			sl += 1;
		}
	}
						
	//allocate data for multiple threads. 
	BWData *data[numThreads];
	for (i=0; i<numThreads; i++){
		data[i] = (BWData *) malloc(sizeof(BWData));
		data[i]->N = N;
		data[i]->T = 0;
		data[i]->osK = osK;
		data[i]->iUStart = iUStarts[i];
		data[i]->osIn = osIn;
		data[i]->kkStart = indStarts[i];
		data[i]->kkEnd = indEnds[i];
		data[i]->stateSyms = stateSyms;	
		data[i]->os = (int *) malloc(maxSL * sizeof(int));
		data[i]->P = P;	
		data[i]->PO = (double *) malloc(N * N * sizeof(double));
		data[i]->logA = (double *) malloc(maxSL * N * sizeof(double));
		data[i]->logB = (double *) malloc(maxSL * N * sizeof(double));
		if (maxSL > N) {
			data[i]->x = (double *) malloc(maxSL * sizeof(double));
		} else {
			data[i]->x = (double *) malloc(N * sizeof(double));
		}
		data[i]->llk = 0.0;
	}
	pthread_t threads[numThreads];
	
	PO = (double *) malloc(N * N * sizeof(double));

	//set initial transition probabilities. 
	if (randSeed != -1) {
		//srand(time(NULL));
		srand((unsigned int)randSeed);	
	}	
	
	//normalize the transition probability. Enforce the properties of the start and end states.
	norm(N,P);

	for (istep=0; istep < maxIter; istep++) {
		
		zeros(N,N,PO);
		llk = 0.0;

		//launch threads
		for (i = 0; i < numThreads; i++) {
			rc = pthread_create(&threads[i], NULL, computePOLogLikehood, data[i]);
		}
		for (i = 0; i < numThreads; i++) {//wait
			pthread_join(threads[i], NULL);
		}			
		//gather all data. 
		for (i = 0; i < numThreads; i++) {
			llk += data[i]->llk;				
			for (j=0; j< N*N; j++) PO[j] += (data[i]->PO)[j];			
		}
				
		// compute the new transition probability.
		norm(N,PO);

		mmax = 0.0;
		for (i=0; i< N*N; i++) {
			if (fabs(PO[i] - P[i]) > mmax) mmax = fabs(PO[i] - P[i]);
		} 
		if (mmax < pTol) break;
		for (i=0; i< N*N; i++) P[i] = PO[i];
		
	}	
	
	for (i=0; i<numThreads; i++){
		free(data[i]->PO);
		free(data[i]->os);
		free(data[i]->logA);
		free(data[i]->logB);
		free(data[i]->x);
		free(data[i]);
	}
	
	
	free(PO);
		
	return llk;	
}


//single linked list of nodes
LinkedList* GetLastInList(LinkedList *list)
{
	LinkedList *p;
	p = list;
	while (p->next != NULL) {
		p = p->next;
	}
	return p;
}

void AppendNodeToList(LinkedList **list, Node *node)
{
	LinkedList *new,*last;
	new = (LinkedList *) malloc(sizeof(LinkedList));
	new->node = node;
	new->next = NULL;
	if (*list == NULL) {
		*list = new;
	} else {
		last = GetLastInList(*list);
		last->next = new;
	}
}

void AddNodeToHead(LinkedList **list, Node *node)
{
	LinkedList *new,*last;
	new = (LinkedList *) malloc(sizeof(LinkedList));
	new->node = node;
	new->next = NULL;
	if (*list == NULL) {
		*list = new;
	} else {
		new->next = *list;
		*list = new;
	}
}

void DeleteList(LinkedList **list)
{
	LinkedList *p, *next;
	p = *list;
	while (p != NULL) {
		next = p->next;
		free(p);
		p = next;
	}
	*list = NULL;
}

void AddNodeToHeadD(DLinkedList **list, Node *node)
{
	DLinkedList *new;
	new = (DLinkedList *) malloc(sizeof(DLinkedList));
	new->node = node;
	new->next = NULL;
	new->pre = NULL;
	if (*list == NULL) {
		*list = new;
	} else {
		new->next = *list;
		(*list)->pre = new;
		*list = new;
	}
}

void DeleteListD(DLinkedList **list)
{
	DLinkedList *p, *next;
	p = *list;
	while (p != NULL) {
		next = p->next;
		free(p);
		p = next;
	}
	*list = NULL;
}


void FindUniqueStateSequencesC(int N, double *P, LinkedList **ends, LinkedList **allNodes, double PSsmall)
{
	//parameters
	double Pcut=0.001;
	int maxSteps = 1000;
	double Pterminal = 0.001;	//allowed residue total probabilities of the sequences. 
	int ii,i,j;
	double p0, p2;
	DLinkedList *pp,*pp0, *pp2; 
	int *flags;
	Node *nNode, *nodeOld;
	double Pa;
	int flag, istep, flagNoNext;
	DLinkedList *activeEnds;
	LinkedList *pp3;
	printf("In FindUniqueStateSequencesC PSsmall=%lf\n",PSsmall);
	
	activeEnds = (DLinkedList *) malloc(sizeof(DLinkedList));
	activeEnds->pre = NULL;
	activeEnds->next = NULL;
	nNode = (Node *) malloc(sizeof(Node));
	nNode->parent = NULL;
	nNode->ii = 0;
	nNode->P = 1.0;
	activeEnds->node = nNode;
		
	for (istep=0; istep<maxSteps; istep++) {
		pp = activeEnds;	
		if (pp == NULL) break;
		while (1) {	
			ii = pp->node->ii;
			flag = 0;
			flagNoNext = 0;
			nodeOld = pp->node;
			p0 = pp->node->P;
			for (j=0; j<N; j++) {
				if (P[ii*N+j] > Pcut) {
					p2 = p0 * P[ii*N+j]; 
					nNode = (Node *) malloc(sizeof(Node));
					nNode->parent = nodeOld;
					nNode->P = p2;	
					nNode->ii = j;
					if (j == 1 || p2 < PSsmall) {	//end state or small probability sequence, mark stop. 
						nNode->ii = 1;
					}
					AppendNodeToList(allNodes, nNode);
					
					if (flag == 0) {//replace the node.
						if (nNode->ii == 1) {//add to ends and remove from activeEnds
							AddNodeToHead(ends,nNode);					
							if (pp->pre == NULL) {//this is the head.
								pp0 = pp->next;
								//free(pp);
								pp = pp0;
								if (pp != NULL) {
									pp->pre = NULL;
								}
								activeEnds = pp;
								flagNoNext = 1;	
							} else {	
								pp0 = pp->pre;
								pp0->next = pp->next;
								if (pp->next != NULL) {
									pp->next->pre = pp0;
								} else {
									pp0->next = NULL;
								}
								free(pp);
								pp = pp0;
							}

						} else {
							pp->node = nNode;
						}
						flag = 1;
					} else { //add to the head. 
						if (nNode->ii == 1) {
							AddNodeToHead(ends,nNode);
						} else {//active node.
							AddNodeToHeadD(&activeEnds,nNode);
						}
					}
				}
			}
			if (flagNoNext == 0) pp = pp->next;
			if (pp == NULL) break;	
		}
			
		Pa = 0.0;
		pp3 = *ends;
		while (pp3 != NULL) {
			Pa += pp3->node->P;
			pp3 = pp3->next;
		}	
		if (Pa > 1.0 - Pterminal) {
			break;
		} 	
	}
	DeleteListD(&activeEnds);
}	
		

void FindUniqueSequencesC(int N, int *S, double *P, int *Ns, double **Ps, LinkedList ***Seqs, double PSsmall)
{	
	LinkedList *ends, *pp, *pp2, *pp3, *pp0, *allNodes, *ss;
	int ii, jj, kk, ns, flag, flag2;
	Node *node, *node2;
	double PP;
		
	ends = NULL;
	allNodes = (LinkedList *) malloc(sizeof(LinkedList));
	allNodes->node = NULL;
	allNodes->next = NULL;
	
	FindUniqueStateSequencesC(N, P, &ends, &allNodes, PSsmall);
	
	ns = 0;
	pp = ends;
	while(1) {
		ns += 1;
		pp = pp->next;
		if (pp == NULL) break;
	}
	
	//print state sequences. 
	pp = ends;
	kk = 0;
	while(1) {
		kk += 1;
		if (kk > ns - 10) {//only print top 10}
			node = pp->node;
			while (1) {
				node = node->parent;
				if (node == NULL) break;
			}
		} 
		pp = pp->next;
		if (pp == NULL) break;
	}
	
	
	//allocate memory for sequences. 
	*Seqs = (LinkedList **) malloc(ns * sizeof(LinkedList *));
	*Ps = (double *) malloc(ns * sizeof(double));
	for (ii=0; ii<ns; ii++) *(*Ps+ii) = 0.0;
	*Ns = 0;
	
	//get the sequences
	pp0 = ends;
	while (1) {
		node = pp0->node;
		PP = node->P;
		if (node->ii == 1)	{//end state
			node = node->parent;
		}
		ss = (LinkedList *) malloc(sizeof(LinkedList));
		ss->next = NULL;
		kk=0;
		while (1) {
			node2 = (Node *)malloc(sizeof(Node));			
			node2->ii = S[node->ii];
			node2->P = PP;
			node2->parent = NULL;
			if (kk==0) {
				ss->node = node2;
				ss->node->P = PP;
				ss->next = NULL;
				kk = 1;
			}
			else {
				AddNodeToHead(&ss,node2);
			}
			node = node->parent;
			if (node == NULL) break;
			if (node->ii == 0) break;	//start symbol
		}
					
		//check if the sequence is already in the list. 
		flag = 1;
		for (ii=0; ii<*Ns; ii++) {
			pp2 = *(*Seqs+ii);
			pp3 = ss;
			flag = 0;
			while (1) {
				if (pp2->node->ii != pp3->node->ii) {	//not the same sequence
					flag = 1;
					break;
				}
				pp2 = pp2->next;
				pp3 = pp3->next;
				if ((pp2 != NULL && pp3 == NULL) || (pp2 == NULL && pp3 != NULL)) {
					flag = 1;
					break;
				}
				if (pp2 == NULL && pp3 == NULL) {
					break;
				}
			}
			if (flag == 0) {	//found the matching sequence.
				break;
			}
		}	
		if (flag == 0) {//old sequence. 
			*(*Ps+ii) += ss->node->P;
		} else {	//new sequence				
			//assign to Seqs
			*(*Seqs + *Ns) = ss;
			*(*Ps + *Ns) = ss->node->P;
			*Ns +=1;
		}
		
		pp0 = pp0->next;
		if (pp0 == NULL) break;
	}
			
	//free memory, remove all nodes. 
	pp = allNodes->next;
	while (1) {
		node = pp->node;
		free(node);
		pp = pp->next;
		if (pp == NULL) break;
	}
		
	//clear linked list. 
	DeleteList(&ends);	
	DeleteList(&allNodes);
	
}

//Add sym to the tree structure. Returns the pointer to the node. 
NodeSym *AddSym(NodeSym *parent, int sym,  LinkedListNodeSym *allNodes)
{
	if (parent == NULL) {
		printf("ERROR in AddSym: parent pointer is NULL.\n");
		exit(1);
	}
	//check the linked list of the daughters. 
	LinkedListNodeSym *daughter;
	daughter = parent->daughterNodeSym;
	while (daughter != NULL) {
		if (daughter->node->sym == sym) {//found the node with the sym.
			daughter->node->count += 1;
			return &(*(daughter->node));
		}
		daughter = daughter->next;
	}
	//did not find the sym, create a new node.
	NodeSym *node;
	node = (NodeSym*) malloc(sizeof(NodeSym));
	node->parent = parent;
	node->daughterNodeSym = NULL;
	node->sym = sym;
	node->count = 1;
	
	allNodes->next = (LinkedListNodeSym *) malloc(sizeof(LinkedListNodeSym));
	allNodes->next->node = node; //collect the pointer for freeing memory at the end. 
	allNodes->next->next = NULL;
	allNodes = allNodes->next;
	
	daughter = parent->daughterNodeSym;
	if (daughter == NULL) {
		daughter = (LinkedListNodeSym *) malloc(sizeof(LinkedListNodeSym));
		daughter->node = node;
		daughter->next = NULL;
		parent->daughterNodeSym = daughter;
	} else {
		while (1) {
			if (daughter -> next == NULL) {
				daughter->next = (LinkedListNodeSym *) malloc(sizeof(LinkedListNodeSym));
				daughter->next->node = node;
				daughter->next->next = NULL;
				break;
			}
			daughter = daughter ->next;		
		}
	}
	return &(*node);
}

void DeleteLinkedListNodeSym(LinkedListNodeSym *list)
{
	LinkedListNodeSym *p, *next;
	p = list;
	while (p != NULL) {
		next = p->next;
		free(p);
		p = next;
	}
	list = NULL;
}
	

//compute the sequennce probability. 
//start with 0 and end with -1. 
double computeSeqProbPOMM(int N, int *S, double *P, int ns, int *seq) {
	
	double *A;
	A = (double *) malloc(N*ns*sizeof(double));
	
	int i,j;
	for (i=0; i<N*ns; i++) A[i] = 0;
	A[getIndex2(0,0,N,ns)] = 1.0;
	for (int t=1; t<ns; t++) {
		for (j=0; j<N; j++) {
			if (S[j] == seq[t]) {
				for (i=0; i<N; i++) {
					A[getIndex2(j,t,N,ns)] += P[getIndex(i,j,N)] * A[getIndex2(i,t-1,N,ns)];
				}
			}
		}
	}	
	double pseq = A[getIndex2(1,ns-1,N,ns)];
	
	free(A);
	return pseq;
}


//double linked list of NodeDs 
void AddNodeToHeadNodeD(NodeD **HeadNode, NodeD *node)
{
	if (HeadNode == NULL) return; 
	
	if (*HeadNode == NULL) {
		*HeadNode = node;
		(*HeadNode)->parent = NULL;
		(*HeadNode)->next = NULL;
	} else {
		node->parent = NULL;
		node->next = *HeadNode;
		(*HeadNode)->parent = node;
		*HeadNode = node;
	}
}

void DeleteLinkNodeD(NodeD **HeadNode){
	NodeD *pos, *next;
	
	if (HeadNode == NULL) return; 
	if (*HeadNode == NULL) return;
	pos = *HeadNode;
	while (pos != NULL) {
		next = pos->next;
		free(pos);
		pos = next;
	}	
	*HeadNode = NULL;
}

void DeleteNodeD(NodeD **HeadNode, NodeD *node)
{
	if (HeadNode == NULL) return; 
	if (*HeadNode == NULL) return;
	if (node == NULL) return;
	if (node->parent == NULL) {
		if (node->next == NULL) {//all deleted. 
			*HeadNode = NULL;
		} else {
			*HeadNode = node->next;
			(*HeadNode)->parent = NULL;
		}	
	} else {
		node->parent->next = node->next;
		if (node->next != NULL) {
			node->next->parent = node->parent;
		}			
	}
	free(node);
}

int LengthLinkNodeD(NodeD **HeadNode)
{
	if (HeadNode == NULL) return 0; 
	if (*HeadNode == NULL) return 0;
	NodeD *pos;
	int n = 0;
	pos = *HeadNode;
	while (pos != NULL) {
		n += 1;
		pos = pos->next;
	}
	return n;
}



void freeArray(double *pt)
{
	free(pt);
}

void freeArrayInt(int *pt)
{
	free(pt);
}



//find all unique sequences and probabilities up to a tolerance. 
//returns number of unique sequences. 
//pointer seqP will be pointing to the transition probabilities of the unique sequences
double *getUniqueSeqProbsPOMM(int N, int *S, double *P)
{
	double PtotTol = 0.99; 		//if the total probability exceeds this, stop. 
	double pTol = 1e-10;			//if the probability of the sequence is smaller than this, the sequence is discarded. 
	double pTransTol = 0.001;	//cut off for transition matrix probability. Below this considered 0.
	int nU = 0;					//number of unique sequences
	int memSize = 0;				//memory size of seqP
	int memSizeInc = 1000;		//size increase of seqP
	double p, Ptot; 					//probability of the sequence
	double *seqP;				//seqP[0] is the number of unique sequences.  
	
	NodeD *HeadNode, *pos, *pos2, *newNode;		
	HeadNode = (NodeD *) malloc(sizeof(NodeD));	
	HeadNode->ii = 0;	//starting at the start state. 
	HeadNode->P = 0.0;  //log probability of sequence to this point. 
	HeadNode->parent = NULL;
	HeadNode->next = NULL; 

	memSize = memSizeInc;
	seqP = (double *) malloc(memSizeInc * sizeof(double));	//initial allocation of memory of the unique sequence probabilities. The calling function must destroy the memory after use. 
	seqP[0] = 0;

	Ptot = 0.0;
	
	int nLen=0;
	
	while (1) {
		pos = HeadNode;
		int flag = 0;
		while (pos != NULL) {
			int ii = pos->ii;
			double Pii = pos->P;
			pos2 = pos->next;
			
			if (P[getIndex(ii,1,N)] > pTransTol) { //reached the end state. 
				p = Pii + log(P[getIndex(ii,1,N)]);
				p = exp(p);			// this is the probability of the sequence. 
				nU += 1;			// found a unique sequence. 
				seqP[nU] = p;
				
				if (nU >= memSize-1) { // increase the mempry size
					memSize += memSizeInc; 
					seqP = (double *) realloc(seqP, memSize * sizeof(double));					
				}
				
				Ptot += p;
			}
			for (int k=2; k<N; k++) {
				if (P[getIndex(ii,k,N)] > pTransTol){
					p = Pii + log(P[getIndex(ii,k,N)]);
					if (p > log(pTol)) { //sequence is growing. add new front to the list
						newNode = (NodeD *) malloc(sizeof(NodeD));
						newNode->parent = NULL;
						newNode->next = NULL;
						newNode->ii = k;
						newNode->P = p;
						AddNodeToHeadNodeD(&HeadNode, newNode);						
						//mark that the front is growing. 
						flag = 1;
					} 						
				}
			}
			
			if (pos != NULL) { //done with this front, delete. 
				DeleteNodeD(&HeadNode,pos);
			}
			pos = pos2;
		}
		
		if (Ptot > PtotTol) {// found enough unique sequences, quit. 
			//printf("Ptot=%f\n",Ptot);
			break;		
		}
		if (flag == 0) {//no new front, quit.
			break;
		}
	}

	seqP[0] = nU;
	//normalize. 
	double ssum = 0.0;
	for (int i=0; i<nU; i++) ssum += seqP[i+1]; //note seqP[0] = nU. 
	for (int i=0; i<nU; i++) seqP[i+1] /= ssum; 	
	
	if (HeadNode != NULL) DeleteLinkNodeD(&HeadNode);	
	return seqP;	
}

int selectSeq(double* pU, int nU) 
{
    // Generate a random number between 0 and 1
    double randProb = (double)rand() / RAND_MAX;
    // Accumulate probabilities until randProb is less than the accumulated
    double accumulatedProb = 0.0;
    for (int i = 0; i < nU; ++i) {
        accumulatedProb += pU[i];
        if (randProb <= accumulatedProb) {
            return i; // Return the index of the selected item
        }
    }
    return -1; //something wrong. 
}


// START 
// Alias method of sampling. Sample M times from N items using probabilities p_i. 
			
typedef struct {
    int alias;
    double prob;
} AliasTableEntry;

void initializeAliasTable(double *probs, int N, AliasTableEntry *aliasTable) {

	double *prob = (double *)malloc(N * sizeof(double));
    int *small = (int *)malloc(N * sizeof(int));
    int *large = (int *)malloc(N * sizeof(int));

    int smallCount = 0, largeCount = 0;

    // Step 1: Initialize the prob array and classify each probability as small or large
    for (int i = 0; i < N; ++i) {
        prob[i] = probs[i] * N; // Scale up probabilities by N
        if (prob[i] < 1.0)
            small[smallCount++] = i;
        else
            large[largeCount++] = i;
    }

    // Step 2: Process the small and large lists
    while (smallCount > 0 && largeCount > 0) {
        int less = small[--smallCount];
        int more = large[--largeCount];

        aliasTable[less].prob = prob[less];
        aliasTable[less].alias = more;

        prob[more] = (prob[more] + prob[less]) - 1.0;
        if (prob[more] < 1.0)
            small[smallCount++] = more;
        else
            large[largeCount++] = more;
    }

    while (largeCount > 0)
        aliasTable[large[--largeCount]].prob = 1.0;

    while (smallCount > 0)
        aliasTable[small[--smallCount]].prob = 1.0;

    free(small);
    free(large);
    free(prob);
}

int sampleAliasMethod(AliasTableEntry *aliasTable, int N) {
    double r = (double)rand() / RAND_MAX * N;
    int idx = (int)r;
    double prob = r - idx;

    if (prob < aliasTable[idx].prob)
        return idx;
    else
        return aliasTable[idx].alias;
}
//END Alias method.  

//sample the sequences for the POMM and compute Pb. 
void getModifiedSequenceCompletenessSamplingModelC(int nSeqs, int N, int *S, double *P, int nSample, double *PBs, double beta, int randSeed)
{
	//set initial transition probabilities. 
	if (randSeed != -1) {
		srand(time(NULL));
		srand((unsigned int)randSeed);	
	} else {
		srand(time(NULL));
	}			
	
	//printf("\nIn C getModifiedSequenceCompletenessSamplingModelC  \n");
	//PrintTransitionMatrix(N,P);
	//for (int i=0; i<N; i++) {
	//	printf("%6.2f ",P[getIndex(0,i,N)]);
	//}
	//printf("\n");
	
	//get unique sequences of the model. 
	double *seqP, *pU;
	int nU;
	
	seqP = getUniqueSeqProbsPOMM(N, S, P);
	nU = (int) seqP[0];
	pU = (double *) malloc(nU * sizeof(double));
	for (int i = 0; i<nU; i++) pU[i] = seqP[i+1];
	free(seqP);

	int *selIDs, *KIDs;
	selIDs = (int *) malloc(nU * sizeof(int));
	KIDs = (int *) malloc(nU * sizeof(int));
	
	printf("	Found nU=%d unique sequences\n",nU);
	
	AliasTableEntry *aliasTable;
	aliasTable = (AliasTableEntry *) malloc(nU * sizeof(AliasTableEntry));
    initializeAliasTable(pU, nU, aliasTable);
		
	for (int isam=0; isam<nSample; isam++) {
	
		int nSel = 0;
		for (int iseq=0; iseq<nSeqs; iseq++) {
			
			int k = sampleAliasMethod(aliasTable, nU);	//Alias method, faster when sampling many times. 			
			//int k = selectSeq(pU, nU);	//select a unique sequences. slower method. 
			
			//check if this already sampled. 
			int flag = 0, mk = -1;
			for (int j=0; j<nSel; j++) {
				if (selIDs[j] == k) {
					flag = 1;
					mk = j;
					break;
				}
			}
			if (flag == 0) {//new unique sequence. 
				selIDs[nSel] = k;
				KIDs[nSel] = 1;
				nSel += 1;
			} else {
				KIDs[mk] += 1;
			}
		}	
		
		double Pc = 0.0;	// sequence completeness
		for (int i=0; i<nSel; i++) {
			Pc += pU[selIDs[i]];
		}
				
		double dd = 0.0;
		for (int i=0; i<nSel; i++) {
			double ps = KIDs[i]*1.0/nSeqs;
			double pm = pU[selIDs[i]]/Pc;
			dd += 0.5 * fabs(ps - pm);
		}
		
		double Pb = (1-beta)*Pc + beta*(1-dd);
		PBs[isam] = Pb;
		
		//printf("Pc=%f Pb=%f\n",Pc,Pb);
		
	}
	
	free(pU);	
	free(selIDs); 
	free(KIDs);
	free(aliasTable);
}

// useful data structures and functionns used in constructNGramPOMMC
typedef struct {
    int* items; // Pointer to the array of integers
    int size;   // Number of integers in the list
	int numVis;	// numbe of the times the list appears	
} IntList;

typedef struct {
    IntList* lists; // Pointer to the array of IntList
    int size;       // Number of IntList in the lists
} ListOfIntLists;

bool are_lists_equal(IntList a, IntList b) {
    if (a.size != b.size) return false;
    for (int i = 0; i < a.size; i++) {
        if (a.items[i] != b.items[i]) return false;
    }
    return true;
}

// get index of newList. If new, add to lol and return the new index. 
int StoreInList(ListOfIntLists* Vecs, IntList newList) {
    // Check if newList is already in Vecs
    for (int i = 0; i < Vecs->size; i++) {
        if (are_lists_equal(Vecs->lists[i], newList)) {
            // increate the number of times it is visited.           
            Vecs->lists[i].numVis += 1;
            return i;
        }
    }
    // newList is not in Vecs, so add it
    Vecs->size++;
    if (Vecs->size == 1) {
		Vecs->lists  = (IntList*) malloc(sizeof(IntList));
	} else {
		Vecs->lists = (IntList*) realloc(Vecs->lists, Vecs->size * sizeof(IntList));
	}
    if (Vecs->lists == NULL) {
        exit(1); // Handle realloc failure, for simplicity just exit here
    }
    int newInd = Vecs->size - 1;
    Vecs->lists[newInd].size = newList.size;
    Vecs->lists[newInd].items = (int *) malloc(newList.size * sizeof(int));
    for (int i=0; i<newList.size; i++) Vecs->lists[newInd].items[i] = newList.items[i];  
    Vecs->lists[newInd].numVis = 1;
    
    return newInd;
}

//free memory.
void freeStoredLists(ListOfIntLists *Vecs)
{
    // Clean up: Remember to free the memory allocated for each list and the ListOfIntLists itself
    for (int i = 0; i < Vecs->size; i++) {
        free(Vecs->lists[i].items); // Assuming the items were dynamically allocated
    }
    free(Vecs->lists);
 }
 
//free memory of ThreeArrays
void freeThreeArrays(ThreeArrays *pt)
{
	free(pt->S);
	free(pt->P);
	free(pt->StateNumVis);
	free(pt);
} 

//note pointers in the three arrays must be freed after calling this function. 
ThreeArrays* constructNGramPOMMC(int nSeq, int *osIn, int ng)
{
	
	int N=0, iState;
	
    int *nGram;	//list of integers in the ngram
    nGram = (int *) malloc(ng * sizeof(int));
		
    ListOfIntLists Vecs; // empty ListOfIntLists for maintaining n-gram seqs associated with the states. 
    IntList vec;
    Vecs.size = 0;
    Vecs.lists = NULL;
    //set up the start state
    nGram[0] = 0;
	vec.size = 1;
	vec.items = nGram;
	vec.numVis = 0;
	iState = StoreInList(&Vecs, vec); 
	//set up the end state
    nGram[0] = -1;
	vec.size = 1;
	vec.items = nGram;
	vec.numVis = 0;
	iState = StoreInList(&Vecs, vec); 
		 
    ListOfIntLists StatePairs; // empty ListOfIntLists for storing state pairs. 
    StatePairs.size = 0;
    StatePairs.lists = NULL;
    IntList statePair;	
    int sp[2];
     
    int *stateSeq;	//state sequence
    stateSeq = (int *) malloc(nSeq * sizeof(int));
    int nStateSeq=0; 
	    
	int k = 0;
	int iS = 0;
	int ipair;
	
	int nZ = 0;
	
	for (int i=0; i < nSeq; i++) {
				
		if (osIn[i] == -1)	{//hit the end state. 
			nGram[0] = -1;
			vec.size = 1;
			vec.items = nGram;
			iState = StoreInList(&Vecs, vec); 
			stateSeq[nStateSeq] = iState;
			nStateSeq += 1;
			
			sp[0] = iS;
			sp[1] = 1;
			statePair.items = sp;
			statePair.size = 2;
			ipair = StoreInList(&StatePairs, statePair); //count the transition frequency. 
			
			continue;
		}
		
		if (osIn[i] == 0) {//start of a new sequence. 
			k=0;
		}

		if (k < ng) {			
			nGram[k] = osIn[i];
			k += 1;
		} else {
			for (int j=1; j < ng; j++) nGram[j-1] = nGram[j];
			nGram[ng-1]= osIn[i];				
		}

		if (k==1 && osIn[i] == 0) nZ +=1;

		vec.size = k;
		vec.items = nGram;		
		iState = StoreInList(&Vecs, vec); 
		stateSeq[nStateSeq] = iState;
		nStateSeq += 1;

		if (iState ==0) {
			iS = 0;
		} else {
			sp[0] = iS;
			sp[1] = iState;
			statePair.items = sp;
			statePair.size = 2;
			ipair = StoreInList(&StatePairs, statePair); //count the transition frequency.
			iS = iState; 
		}
	}	
	

	N = Vecs.size;
	printf("In constructNGramPOMMC: Nums states = %d\n",N);	
	printf("	Number of state transition pairs = %d\n",StatePairs.size);
		
	//get the state vector and transition probabilities.
	ThreeArrays *A;
	A = (ThreeArrays *) malloc(sizeof(ThreeArrays));
	A->S = (int *) malloc(N*sizeof(int));
	A->P = (double *) malloc(N*N*sizeof(double)); zeros(N,N,A->P);	
	A->StateNumVis = (int *) malloc(N*sizeof(int));
	A->N = N;
	
	for (int i=0; i < Vecs.size; i++) {
		int sym = Vecs.lists[i].items[Vecs.lists[i].size - 1];
		A->S[i] = sym;
		A->StateNumVis[i] = Vecs.lists[i].numVis;
	}	
	
	//the start and the end states have one too many numVis, delete by 1. This is because we inserted them in the begining to anchor their positions in S. 
	A->StateNumVis[0] -= 1;
	A->StateNumVis[1] -= 1;

	for (int i=0; i < StatePairs.size; i++) {
		int ii = StatePairs.lists[i].items[0];
		int jj = StatePairs.lists[i].items[1];				
		A->P[getIndex(ii,jj,N)] = (double) StatePairs.lists[i].numVis;
	}
	norm(N, A->P);
	
	//print information. 
	//PrintTransitionMatrix(N, A->P);	
	//for (int i=0; i < Vecs.size; i++) {
	//	printf("	state: %d numVis: %d\n",A->S[i],A->StateNumVis[i]);
	//}
		
	freeStoredLists(&Vecs);
	freeStoredLists(&StatePairs);
	free(stateSeq);
	free(nGram);
	
	return A; 
}




  

