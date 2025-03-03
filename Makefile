cuda_version = $(nvidia-smi | egrep CUDA | cut -f3 -d: | sed 's/|*//g')
nvcc = /usr/local/cuda-$(cuda_version)/bin/nvcc
nvccflag = -std=c++11 -O3 -arch=native -DHEAP_SORT
heappath = ./heap
timedebugpath = ./

ifeq "$(cuda_version)" ""
	echo "Not able to find  device cuda version, using default."
	nvcc = nvcc
endif

#nvcc -O3 main.cu time_debug.cu -DHEAP_SORT -I./ -I./heap -o nj_run.run

all: nj_gpu nj_gpu_time

nj_gpu: main.cu
	$(nvcc) $(nvccflag) $< time_debug.cu -DNO_TIME -I$(heappath)/ -I$(timedebugpath)/ -o $@.run

nj_gpu_time: main.cu
	$(nvcc) $(nvccflag) $< time_debug.cu -I$(heappath)/ -I$(timedebugpath)/ -o $@.run
	

clean:
	rm -rf nj_gpu.run nj_gpu_time.run
