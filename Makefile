nvcc = nvcc
nvccflag = -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT
heappath = ./heap
timedebugpath = ./


#nvcc -O3 main.cu time_debug.cu -DHEAP_SORT -I./ -I./heap -o nj_run.run


all: nj_gpu

nj_gpu: main.cu
	echo $<
	$(nvcc) $(nvccflag) $< time_debug.cu  -I$(heappath)/ -I$(timedebugpath)/ -o $@.run

clean:
	rm -rf nj_gpu.run
