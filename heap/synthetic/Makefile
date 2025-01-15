nvcc = nvcc
nvccflag = -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT
heappath = ..

all: uheap_synthetic

uheap_synthetic: sort.cu $(heappath)/uheap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)/ $< -o $@

clean:
	rm -rf uheap_synthetic
