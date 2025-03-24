HEADERS = 
LIBS = -lm

all: libPOMM.so 

libPOMM.so: libPOMM.o
	gcc -shared -o libPOMM.so libPOMM.o $(LIBS)

libPOMM.o: libPOMM.c libPOMM.h
	gcc -fPIC -c libPOMM.c $(HEADERS)

test: test.c libPOMM.o
	gcc -fPIC -c test.c $(HEADERS)
	gcc -o test test.o libPOMM.o $(LIBS)
