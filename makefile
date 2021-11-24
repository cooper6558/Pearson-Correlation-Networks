correlation: main.o correlation.o
	nvcc -arch=sm_30 -g main.o correlation.o -o correlation

correlation.o: correlation.cu
	nvcc -arch=sm_30 correlation.cu -c -o correlation.o

main.o: main.c correlation.h
	g++ -Wall main.c -c -o main.o

clean:
	rm correlation.o
	rm main.o
	rm correlation
