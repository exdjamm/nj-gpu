nvcc = nvcc
nvcc_colab = /usr/local/cuda-12.4/bin/nvcc
nvccflag = -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT
heappath = ./heap
timedebugpath = ./

#nvcc -O3 main.cu time_debug.cu -DHEAP_SORT -I./ -I./heap -o nj_run.run

all: nj_gpu nj_gpu_no_time

nj_gpu: main.cu
# echo $<
	$(nvcc) $(nvccflag) $< time_debug.cu  -I$(heappath)/ -I$(timedebugpath)/ -o $@.run

nj_gpu_colab: main.cu
# echo $<
	$(nvcc) $(nvccflag) $< time_debug.cu  -I$(heappath)/ -I$(timedebugpath)/ -o $@.run

nj_gpu_no_time: main.cu
# echo $<
	$(nvcc) $(nvccflag) $< time_debug.cu -DNO_TIME -I$(heappath)/ -I$(timedebugpath)/ -o $@.run

clean:
	rm -rf nj_gpu.run
