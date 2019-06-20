NVCC        = nvcc

NVCC_FLAGS  = --ptxas-options=-v -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = scan_largearray
OBJ	        = scan_largearray_cu.o scan_largearray_cpp.o

default: $(EXE)

scan_largearray_cu.o: scan_largearray.cu scan_largearray_kernel.cu
	$(NVCC) -c -o $@ scan_largearray.cu $(NVCC_FLAGS)

scan_largearray_cpp.o: scan_gold.cpp
	$(NVCC) -c -o $@ scan_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
