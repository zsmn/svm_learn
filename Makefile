CC = g++
CFLAGS = -g -Wall
SRCS = svm.cpp
PROG = svm

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)