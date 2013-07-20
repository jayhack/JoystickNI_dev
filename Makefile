CC = g++
OBJS = get_fist.o fist.o
DEBUG = -g
LFLAGS = -Wall -c $(DEBUG)

OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`


get_fist: $(OBJS)
	$(CC) -o get_fist $(OBJS) $(OPENCV_CFLAGS) $(OPENCV_LIBS)
get_fist.o: get_fist.cpp fist.h
	$(CC) $(LFLAGS) get_fist.cpp $(OPENCV_CFLAGS) $(OPENCV_LIBS)
fist.o: fist.cpp fist.h
	$(CC) $(LFLAGS) fist.cpp $(OPENCV_CFLAGS) $(OPENCV_LIBS)