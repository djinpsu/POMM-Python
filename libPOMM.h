/*
These are C-functions used in POMM.py
 
Copyright (C) 2016, Dezhe Z. Jin (dzj2@psu.edu)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LIBPOMM_H_INCLUDED
#define LIBPOMM_H_INCLUDED

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdint.h>
#include <time.h>

/*
This the Baum-Welch algorithm for learning POMM. 
The sequences are assumed to have the state and end states. 
Returns the transition matrix P after using the observed sequence osIn.
Note that osIn is the list of unique sequences observed. osK is number of times the sequences are observed. 
Use GetUniqueSequences before calling this function. 
Parameters. 
	nSeq - length of concatenated sequence osIn
	osIn - concatenated sequence. use 0 and -1 signal start and end of a sequence. 
	nU - number of unique sequences.
	osK - number of occurance of each sequence 
	N - total number of states, includeing the start and the end states. 
	stateSym - array of length  N, symbols of each state. The first two are 0, -1 for the start and the end states/ 
	P - array of length N^2, transition matrix
	ml - maximum likelihood
	pTol - maximum changes in P for stopping iteration.
	maxIter - maximum iterations. 
	randSeed - seed for random number generator. If -1, assumes that P is setup already. 
Returns log-likelihood of the sequences
*/ 
double BWPOMMC(int nSeq, int *osIn, int nU, int *osK, int N, int *stateSyms, double *P, double pTol, int maxIter, int randSeed);

/*
Data structure for using multi-thread for BW algorithm. 
*/
typedef struct BWData BWData;
struct BWData {
	int N;			//number of states. 
	int T;			//length of sequence. 
	int	*osK;		//copy number of unique sequences
	int iUStart;	//start of the copy number
	int *osIn;		//input sequence.
	int kkStart;	//start index in osIn to work in this thread
	int kkEnd;		//end index in osIn to work in this thread
	int *stateSyms;	//size N
	int *os;		//size T
	double *P;		//size N x N
	double *PO;		//size N x N, transitions counts to be added to the total
	double *logA; 	//size N x T, log alphas
	double *logB;	//size N x T, log betas
	double *x;		//size max(N, T), work space. 
	double llk;	//number, maximum likelihood to be added to the total
};
//multi thread version of BW. 
double BWPOMMCMultiThread(int nSeq, int *osIn, int nU, int *osK, int N, int *stateSyms, double *P, double pTol, int maxIter, int randSeed, int numThreads);


void PrintTransitionMatrix(int N, double * P);

/*
Get unique sequences in sequence osIn.
Parameters:
	nSeq - length of sequences. 
	osIn - sequences. sequences start with 0, end with -1. 
	nSeqU - length of osU. 
	osU - unique sequencecs. sequences start with 0, end with -1. 
	nUniique - number of unique sequences.
	osK - array, number of time each sequence occured. 
*/
void GetUniqueSequences(int nSeq, int *osIn, int *nSeqU, int *osU, int *nUnique, int *osK);

//normalize the transition matrix. The first row is the start state, the second row is the end state. 
void norm(int N, double *P);

//tree of state notes. 

typedef struct Node Node;
struct Node {
	Node *parent; //pointer to the parent. 
	int ii;	  		//state id. 
	double P;	  	//probability of the state sequence to this point. 
};


typedef struct LinkedList LinkedList;
struct LinkedList {
	LinkedList *next; 	//pointer to the next in the list. 
	Node *node;			//pointer to the node at the end of the tree structure. 
};

//single linked list of nodes
LinkedList* GetLastInList(LinkedList *list);
void AppendNodeToList(LinkedList **list, Node *node);
void AddNodeToHead(LinkedList **list, Node *node);
void DeleteList(LinkedList **list);

typedef struct DLinkedList DLinkedList;
struct DLinkedList {
	DLinkedList *next; 	//pointer to the next in the list. 
	DLinkedList *pre; 	//pointer to the next in the list. 
	Node *node;			//pointer to the node at the end of the tree structure. 
};

//double linked list of nodes
void AddNodeToHeadD(DLinkedList **list, Node *node);
void DeleteListD(DLinkedList **list);

//find all unique state seqeunces given the transition probability. 
void FindUniqueStateSequencesC(int N, double *P, LinkedList **ends, LinkedList **allNodes, double PSsmall);
//find all unique sequences given the state and transition probabilities. 
void FindUniqueSequencesC(int N, int *S, double *P, int *Ns, double **Ps, LinkedList ***Seqs, double PSsmall);


//data structure for generating sequences and computing Pb.
//the unique sequences are stored in a tree structure. The leafs of the tree contain the last sym in the seq 
//and how many times the sequence generated. 
typedef struct LinkedListNodeSym LinkedListNodeSym;
typedef struct NodeSym NodeSym;
struct NodeSym {
	NodeSym *parent;		//parent of this node. 
	LinkedListNodeSym *daughterNodeSym;	//list of all nodes going from this node. 
	int sym; 		//symbol associated with this node
	int count;		//numbe of times the node is visited.
};
struct LinkedListNodeSym {
	NodeSym *node;
	LinkedListNodeSym *next;
};
//Add sym to the tree structure. Returns the pointer to the node. 
NodeSym *AddSym(NodeSym *parent, int sym, LinkedListNodeSym *allNodes);
void DeleteLinkedListNodeSym(LinkedListNodeSym *list); 

//compute the sequennce probability. 
//seq starts with 0 and ends with -1. 
double computeSeqProbPOMM(int N, int *S, double *P, int ns, int *seq);

//find modified sequence completeness distribution. the size of set is nSeqs
void getModifiedSequenceCompletenessSamplingModelC(int nSeqs, int N, int *S, double *P, int nSample, double *PBs, double beta, int randSeed);

//find all unique sequences and probabilities up to a tolerance. 
//returns number of unique sequences. 
//pointer seqP will be pointing to the transition probabilities of the unique sequences
//seqP[0] is the sequence length. 
double *getUniqueSeqProbsPOMM(int N, int *S, double *P); 

typedef struct NodeD NodeD;
struct NodeD {
	NodeD *parent; //pointer to the parent.
	NodeD *next; 	//pointer to the next 
	int ii;	  		//state id. 
	double P;	  	//probability of the state sequence to this point. 
};
//double linked list of NodeDs 
void AddNodeToHeadNodeD(NodeD **HeadNode, NodeD *node);
void DeleteLinkNodeD(NodeD **HeadNode);
void DeleteNodeD(NodeD **HeadNode, NodeD *node);
int LengthLinkNodeD(NodeD **HeadNode);

//free memory pointed by pt double, called from Python using CDLL. 
void freeArray(double *pt);
//free memory pointed by pt int, called from Python using CDLL.
void freeArrayInt(int *pt);

typedef struct {
    int N;
    int* S;
    double* P;
    int* StateNumVis;
} ThreeArrays;

/*
	Construct a POMM equivalent to n-gram. ng=1 is the Markov model.
	Note that pointers S and P must be freed after returning Python. Use freeArray.

	Innputs

	nSeq 	- number sequences
	osIn	- sequences, expecting numerical sequences.
	ng		- order of the Markov model. 
	S		- state vector
	P		- probability
	 
	Returns 

	Structure ThreeArrays containing S, P, StateNumVis, N.
 
*/ 
ThreeArrays* constructNGramPOMMC(int nSeq, int *osIn, int ng);


#endif
