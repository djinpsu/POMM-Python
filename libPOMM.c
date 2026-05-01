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

//2026-3-26, fixed the problem. if P(i,j) = 0, it will remain 0. 
double BWPOMMC(int nSeq, int *osIn, int nU, int *osK, int N, int *stateSyms,
               double *P, double pTol, int maxIter, int randSeed)
{
    int i, j, iid, maxSL, sl, lb, istep, kk, ik, T, t, iU;
    double A0, mmax;
    double *logA, *logB, *PO, *x;
    int *os;
    int *allowed;
    double llk = 0.0;

    if (randSeed != -1) {
        srand((unsigned int)randSeed);
    }

    /* Structural mask: transitions initially > 0 are allowed forever.
       Initial zeros remain forbidden forever. */
    allowed = (int *) malloc(N * N * sizeof(int));
    if (allowed == NULL) {
        return -INFINITY;   /* or handle error */
    }       
    for (i = 0; i < N * N; i++) {
        allowed[i] = (P[i] > 0.0) ? 1 : 0;
    }
    /* hard structural constraints */
    allowed[getIndex(0,1,N)] = 0;   /* no start -> end */
    for (i = 0; i < N; i++) {
        allowed[getIndex(i,0,N)] = 0;   /* no -> start */
        allowed[getIndex(1,i,N)] = 0;   /* no outgoing from end */
    }    

    /* Normalize initial P while preserving constraints. */
    for (i = 0; i < N * N; i++) {
        if (!allowed[i]) P[i] = 0.0;
    }
    norm(N, P);
    for (i = 0; i < N * N; i++) {
        if (!allowed[i]) P[i] = 0.0;
    }

    maxSL = getMaxSeqLen(nSeq, osIn);

    logA = (double *) malloc(maxSL * N * sizeof(double));
    logB = (double *) malloc(maxSL * N * sizeof(double));
    PO   = (double *) malloc(N * N * sizeof(double));
    os   = (int *) malloc(maxSL * sizeof(int));
    if (maxSL > N) x = (double *) malloc(maxSL * sizeof(double));
    else           x = (double *) malloc(N * sizeof(double));

    for (istep = 0; istep < maxIter; istep++) {
        zeros(N, N, PO);
        kk = 0;
        T = 0;
        iU = 0;
        llk = 0.0;

        while (kk < nSeq) {
            for (ik = kk; ik < nSeq; ik++) {
                lb = osIn[ik];
                if (lb == 0) {
                    os[0] = 0;
                    T = 1;
                } else if (lb == -1) {
                    os[T++] = -1;
                    break;
                } else {
                    os[T++] = lb;
                }
            }
            if (lb != -1) break;
            kk += T;

            /* Forward */
            logZeros(N, T, logA);
            logA[0] = 0.0;   /* state 0 at t=0 */

            for (t = 1; t < T; t++) {
                for (i = 0; i < N; i++) {
                    if (stateSyms[i] == os[t]) {
                        for (j = 0; j < N; j++) {
                            iid = getIndex(j, i, N);
                            if (allowed[iid] && P[iid] > 0.0) {
                                x[j] = logA[getIndex2(j, t - 1, N, T)] + log(P[iid]);
                            } else {
                                x[j] = -INFINITY;
                            }
                        }
                        logA[getIndex2(i, t, N, T)] = logsumexp(x, N);
                    }
                }
            }

            /* Backward */
            logZeros(N, T, logB);
            logB[getIndex2(1, T - 1, N, T)] = 0.0;   /* end state at final time */

            for (t = T - 2; t > -1; t--) {
                for (i = 0; i < N; i++) {
                    if (stateSyms[i] == os[t]) {
                        for (j = 0; j < N; j++) {
                            iid = getIndex(i, j, N);
                            if (allowed[iid] && P[iid] > 0.0) {
                                x[j] = log(P[iid]) + logB[getIndex2(j, t + 1, N, T)];
                            } else {
                                x[j] = -INFINITY;
                            }
                        }
                        logB[getIndex2(i, t, N, T)] = logsumexp(x, N);
                    }
                }
            }

            /* Sequence likelihood */
            A0 = exp(logA[getIndex2(1, T - 1, N, T)]);
            if (A0 == 0.0) {
                iU++;
                continue;
            }

            /* Expected transition counts */
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    iid = getIndex(i, j, N);

                    if (!allowed[iid] || P[iid] <= 0.0) {
                        continue;
                    }

                    for (t = 0; t < T - 1; t++) {
                        x[t] = logA[getIndex2(i, t, N, T)]
                             + log(P[iid])
                             + logB[getIndex2(j, t + 1, N, T)];
                    }

                    PO[iid] += osK[iU] * exp(logsumexp(x, T - 1)) / A0;
                }
            }

            llk += logA[getIndex2(1, T - 1, N, T)] * osK[iU];
            iU++;
        }

        /* Enforce forbidden edges before normalization */
        for (i = 0; i < N * N; i++) {
            if (!allowed[i]) PO[i] = 0.0;
        }

        norm(N, PO);

        /* Enforce mask again after normalization */
        for (i = 0; i < N * N; i++) {
            if (!allowed[i]) PO[i] = 0.0;
        }

        mmax = 0.0;
        for (i = 0; i < N * N; i++) {
            if (fabs(PO[i] - P[i]) > mmax) mmax = fabs(PO[i] - P[i]);
        }

        for (i = 0; i < N * N; i++) {
            P[i] = PO[i];
        }

        if (mmax < pTol) break;
    }

    free(logA);
    free(logB);
    free(PO);
    free(os);
    free(x);
    free(allowed);

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

void FreeSequenceList(LinkedList *ss) {
    LinkedList *p = ss;
    while (p != NULL) {
        LinkedList *next = p->next;
        if (p->node != NULL) free(p->node);
        free(p);
        p = next;
    }
}		

void FindUniqueStateSequencesC_CSR(
    CSRMatrix *A,
    LinkedList **ends,
    LinkedList **allNodes,
    double PSsmall
)
{
    double Pcut = 0.0;       // safer than 0.001 for sparse model
    int maxSteps = 1000;
    double Pterminal = 0.001;

    int ii, j, istep;
    double p0, p2, Pa;

    DLinkedList *pp, *pp0;
    Node *nNode, *nodeOld;
    int flag, flagNoNext;
    DLinkedList *activeEnds;
    LinkedList *pp3;

    int N = A->N;

    activeEnds = (DLinkedList *) malloc(sizeof(DLinkedList));
    if (activeEnds == NULL) return;

    activeEnds->pre = NULL;
    activeEnds->next = NULL;

    nNode = (Node *) malloc(sizeof(Node));
    if (nNode == NULL) {
        free(activeEnds);
        return;
    }

    nNode->parent = NULL;
    nNode->ii = 0;
    nNode->P = 1.0;
    activeEnds->node = nNode;

    AppendNodeToList(allNodes, nNode);

    for (istep = 0; istep < maxSteps; istep++) {
        pp = activeEnds;
        if (pp == NULL) break;

        while (pp != NULL) {
            ii = pp->node->ii;
            flag = 0;
            flagNoNext = 0;
            nodeOld = pp->node;
            p0 = pp->node->P;

            for (int kk = A->rowPtr[ii]; kk < A->rowPtr[ii + 1]; kk++) {
                j = A->colInd[kk];
                double pij = A->val[kk];

                if (pij > Pcut) {
                    p2 = p0 * pij;

                    nNode = (Node *) malloc(sizeof(Node));
                    if (nNode == NULL) {
                        DeleteListD(&activeEnds);
                        return;
                    }

                    nNode->parent = nodeOld;
                    nNode->P = p2;
                    nNode->ii = j;

                    if (j == 1 || p2 < PSsmall) {
                        nNode->ii = 1;
                    }

                    AppendNodeToList(allNodes, nNode);

                    if (flag == 0) {
                        if (nNode->ii == 1) {
                            AddNodeToHead(ends, nNode);

                            if (pp->pre == NULL) {
                                pp0 = pp->next;

                                DLinkedList *old = pp;

                                pp = pp0;
                                if (pp != NULL) {
                                    pp->pre = NULL;
                                }

                                activeEnds = pp;
                                free(old);

                                flagNoNext = 1;
                            } else {
                                pp0 = pp->pre;
                                pp0->next = pp->next;

                                if (pp->next != NULL) {
                                    pp->next->pre = pp0;
                                }

                                DLinkedList *old = pp;
                                pp = pp0;
                                free(old);
                            }
                        } else {
                            pp->node = nNode;
                        }

                        flag = 1;
                    } else {
                        if (nNode->ii == 1) {
                            AddNodeToHead(ends, nNode);
                        } else {
                            AddNodeToHeadD(&activeEnds, nNode);
                        }
                    }
                }
            }

            if (flagNoNext == 0) {
                pp = pp->next;
            }
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

void FreeLinkedSequence(LinkedList *ss)
{
    LinkedList *p = ss;

    while (p != NULL) {
        LinkedList *next = p->next;

        if (p->node != NULL) {
            free(p->node);
        }

        free(p);
        p = next;
    }
}

void FindUniqueSequencesC_CSR(
    int N,
    int *S,
    CSRMatrix *A,
    int *Ns,
    double **Ps,
    LinkedList ***Seqs,
    double PSsmall
)
{
    LinkedList *ends, *pp, *pp2, *pp3, *pp0, *allNodes, *ss;
    int ii, kk, ns, flag;
    Node *node, *node2;
    double PP;

    ends = NULL;

    allNodes = (LinkedList *) malloc(sizeof(LinkedList));
    if (allNodes == NULL) {
        *Ns = 0;
        *Ps = NULL;
        *Seqs = NULL;
        return;
    }

    allNodes->node = NULL;
    allNodes->next = NULL;

    FindUniqueStateSequencesC_CSR(A, &ends, &allNodes, PSsmall);

    if (ends == NULL) {
        *Ns = 0;
        *Ps = NULL;
        *Seqs = NULL;

        DeleteList(&allNodes);
        return;
    }

    ns = 0;
    pp = ends;
    while (pp != NULL) {
        ns += 1;
        pp = pp->next;
    }

    *Seqs = (LinkedList **) malloc(ns * sizeof(LinkedList *));
    *Ps = (double *) malloc(ns * sizeof(double));

    if (*Seqs == NULL || *Ps == NULL) {
        if (*Seqs != NULL) free(*Seqs);
        if (*Ps != NULL) free(*Ps);

        *Seqs = NULL;
        *Ps = NULL;
        *Ns = 0;

        pp = allNodes->next;
        while (pp != NULL) {
            LinkedList *next = pp->next;
            if (pp->node != NULL) free(pp->node);
            free(pp);
            pp = next;
        }

        DeleteList(&ends);
        free(allNodes);

        return;
    }

    for (ii = 0; ii < ns; ii++) {
        (*Seqs)[ii] = NULL;
        (*Ps)[ii] = 0.0;
    }

    *Ns = 0;

    pp0 = ends;
    while (pp0 != NULL) {
        node = pp0->node;
        PP = node->P;

        if (node->ii == 1) {
            node = node->parent;
        }

        if (node == NULL) {
            pp0 = pp0->next;
            continue;
        }

        ss = (LinkedList *) malloc(sizeof(LinkedList));
        if (ss == NULL) {
            pp0 = pp0->next;
            continue;
        }

        ss->next = NULL;
        ss->node = NULL;

        kk = 0;

        while (node != NULL) {
            node2 = (Node *) malloc(sizeof(Node));
            if (node2 == NULL) {
                FreeLinkedSequence(ss);
                ss = NULL;
                break;
            }

            node2->ii = S[node->ii];
            node2->P = PP;
            node2->parent = NULL;

            if (kk == 0) {
                ss->node = node2;
                ss->node->P = PP;
                ss->next = NULL;
                kk = 1;
            } else {
                AddNodeToHead(&ss, node2);
            }

            node = node->parent;

            if (node == NULL) break;
            if (node->ii == 0) break;
        }

        if (ss == NULL) {
            pp0 = pp0->next;
            continue;
        }

        /*
           Check whether this emitted symbolic sequence already exists.
           flag == 0 means found matching old sequence.
           flag == 1 means new sequence.
        */
        flag = 1;

        for (ii = 0; ii < *Ns; ii++) {
            pp2 = (*Seqs)[ii];
            pp3 = ss;

            flag = 0;

            while (1) {
                if (pp2->node->ii != pp3->node->ii) {
                    flag = 1;
                    break;
                }

                pp2 = pp2->next;
                pp3 = pp3->next;

                if ((pp2 != NULL && pp3 == NULL) ||
                    (pp2 == NULL && pp3 != NULL)) {
                    flag = 1;
                    break;
                }

                if (pp2 == NULL && pp3 == NULL) {
                    break;
                }
            }

            if (flag == 0) {
                break;
            }
        }

        if (flag == 0) {
            /*
               Old emitted sequence. Add probability and free temporary ss.
            */
            (*Ps)[ii] += ss->node->P;
            FreeLinkedSequence(ss);
        } else {
            /*
               New emitted sequence.
            */
            (*Seqs)[*Ns] = ss;
            (*Ps)[*Ns] = ss->node->P;
            *Ns += 1;
        }

        pp0 = pp0->next;
    }

    /*
       Free state-path nodes stored in allNodes.
       allNodes is a sentinel. Its first real node is allNodes->next.
    */
    pp = allNodes->next;
    while (pp != NULL) {
        LinkedList *next = pp->next;

        if (pp->node != NULL) {
            free(pp->node);
        }

        free(pp);
        pp = next;
    }

    DeleteList(&ends);
    free(allNodes);
}

double *getUniqueSeqProbsPOMM_CSR(int N, int *S, CSRMatrix *A)
{
    const double PSsmall = 1e-5;

    int Ns = 0;
    double *Ps = NULL;
    LinkedList **Seqs = NULL;

    FindUniqueSequencesC_CSR(N, S, A, &Ns, &Ps, &Seqs, PSsmall);

    if (Ns <= 0 || Ps == NULL) {
        if (Ps != NULL) {
            free(Ps);
        }

        if (Seqs != NULL) {
            free(Seqs);
        }

        double *seqP = (double *) malloc(sizeof(double));
        if (seqP != NULL) {
            seqP[0] = 0.0;
        }

        return seqP;
    }

    double *seqP = (double *) malloc((Ns + 1) * sizeof(double));

    if (seqP == NULL) {
        if (Ps != NULL) {
            free(Ps);
        }

        if (Seqs != NULL) {
            for (int i = 0; i < Ns; i++) {
                if (Seqs[i] != NULL) {
                    FreeLinkedSequence(Seqs[i]);
                }
            }
            free(Seqs);
        }

        return NULL;
    }

    seqP[0] = (double) Ns;

    double ssum = 0.0;

    for (int i = 0; i < Ns; i++) {
        seqP[i + 1] = Ps[i];
        ssum += Ps[i];
    }

    if (ssum > 0.0) {
        for (int i = 0; i < Ns; i++) {
            seqP[i + 1] /= ssum;
        }
    }

    free(Ps);

    if (Seqs != NULL) {
        for (int i = 0; i < Ns; i++) {
            if (Seqs[i] != NULL) {
                FreeLinkedSequence(Seqs[i]);
            }
        }
        free(Seqs);
    }

    return seqP;
}

typedef struct {
    int alias;
    double prob;
} AliasTableEntry;

void initializeAliasTable(double *probs, int N, AliasTableEntry *aliasTable);
static int sampleAliasMethod(AliasTableEntry *aliasTable, int N);

void getModifiedSequenceCompletenessSamplingModelCSR_C(
    int nSeqs,
    int N,
    int *S,
    int nnz,
    int *rowPtr,
    int *colInd,
    double *val,
    int nSample,
    double *PBs,
    double beta,
    int randSeed
)
{
    if (randSeed != -1) {
        srand((unsigned int) randSeed);
    } else {
        srand((unsigned int) time(NULL));
    }

    if (nSeqs <= 0 || nSample <= 0) {
        return;
    }

    CSRMatrix A;
    A.N = N;
    A.nnz = nnz;
    A.rowPtr = rowPtr;
    A.colInd = colInd;
    A.val = val;

    double *seqP = getUniqueSeqProbsPOMM_CSR(N, S, &A);

    if (!seqP) return;

    int nU = (int) seqP[0];

    if (nU <= 0) {
        free(seqP);
        return;
    }

    double *pU = (double *) malloc(nU * sizeof(double));
    int *counts = (int *) malloc(nU * sizeof(int));
    AliasTableEntry *aliasTable =
        (AliasTableEntry *) malloc(nU * sizeof(AliasTableEntry));

    if (!pU || !counts || !aliasTable) {
        free(seqP);
        free(pU);
        free(counts);
        free(aliasTable);
        return;
    }

    for (int i = 0; i < nU; i++) {
        pU[i] = seqP[i + 1];
    }

    free(seqP);

    initializeAliasTable(pU, nU, aliasTable);

    for (int isam = 0; isam < nSample; isam++) {
        for (int i = 0; i < nU; i++) {
            counts[i] = 0;
        }

        for (int iseq = 0; iseq < nSeqs; iseq++) {
            int k = sampleAliasMethod(aliasTable, nU);

            if (k >= 0 && k < nU) {
                counts[k] += 1;
            }
        }

        double Pc = 0.0;

        for (int i = 0; i < nU; i++) {
            if (counts[i] > 0) {
                Pc += pU[i];
            }
        }

        double dd = 0.0;

        if (Pc > 0.0) {
            for (int i = 0; i < nU; i++) {
                if (counts[i] > 0) {
                    double ps = (double) counts[i] / (double) nSeqs;
                    double pm = pU[i] / Pc;
                    dd += 0.5 * fabs(ps - pm);
                }
            }
        }

        PBs[isam] = (1.0 - beta) * Pc + beta * (1.0 - dd);
    }

    free(pU);
    free(counts);
    free(aliasTable);
}

// compute the sequence probability using CSR transition matrix P.
// start state is 0 and end state is 1.
// seq[0] should correspond to start symbol/state condition,
// seq[ns-1] should correspond to end condition.
double computeSeqProbPOMM_CSR(
    int N,
    int *S,
    int *row_ptr,
    int *col_ind,
    double *Pval,
    int ns,
    int *seq
) {
    double *A_prev = (double *) malloc(N * sizeof(double));
    double *A_curr = (double *) malloc(N * sizeof(double));

    if (A_prev == NULL || A_curr == NULL) {
        if (A_prev) free(A_prev);
        if (A_curr) free(A_curr);
        return 0.0;
    }

    int i, j, k, t;

    for (i = 0; i < N; i++) {
        A_prev[i] = 0.0;
        A_curr[i] = 0.0;
    }

    // Start state
    A_prev[0] = 1.0;

    for (t = 1; t < ns; t++) {

        // reset current alpha
        for (j = 0; j < N; j++) {
            A_curr[j] = 0.0;
        }

        // propagate probability mass through sparse outgoing transitions
        for (i = 0; i < N; i++) {
            double ai = A_prev[i];

            if (ai == 0.0) continue;

            for (k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
                j = col_ind[k];

                if (S[j] == seq[t]) {
                    A_curr[j] += Pval[k] * ai;
                }
            }
        }

        // swap A_prev and A_curr
        double *tmp = A_prev;
        A_prev = A_curr;
        A_curr = tmp;
    }

    double pseq = A_prev[1];

    free(A_prev);
    free(A_curr);

    return pseq;
}

void freeArray(double *pt)
{
	free(pt);
}

void freeArrayInt(int *pt)
{
	free(pt);
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
			
void initializeAliasTable(double *probs, int N, AliasTableEntry *aliasTable) {
    double *prob = (double *)malloc(N * sizeof(double));
    int *small = (int *)malloc(N * sizeof(int));
    int *large = (int *)malloc(N * sizeof(int));

    if (!prob || !small || !large) {
        free(prob);
        free(small);
        free(large);
        return;
    }

    int smallCount = 0, largeCount = 0;

    for (int i = 0; i < N; ++i) {
        aliasTable[i].alias = i;
        aliasTable[i].prob = 1.0;

        prob[i] = probs[i] * N;
        if (prob[i] < 1.0)
            small[smallCount++] = i;
        else
            large[largeCount++] = i;
    }

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

    while (largeCount > 0) {
        int idx = large[--largeCount];
        aliasTable[idx].prob = 1.0;
        aliasTable[idx].alias = idx;
    }

    while (smallCount > 0) {
        int idx = small[--smallCount];
        aliasTable[idx].prob = 1.0;
        aliasTable[idx].alias = idx;
    }

    free(small);
    free(large);
    free(prob);
}

int sampleAliasMethod(AliasTableEntry *aliasTable, int N) {
    double u = ((double)rand()) / ((double)RAND_MAX + 1.0);
    double r = u * N;

    int idx = (int)r;
    if (idx < 0) idx = 0;
    if (idx >= N) idx = N - 1;

    double prob = r - idx;

    if (prob < aliasTable[idx].prob)
        return idx;
    else
        return aliasTable[idx].alias;
}

//END Alias method.  


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




  

