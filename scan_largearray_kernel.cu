#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
// You can use any other block size you wish.
#define BLOCK_SIZE 1024
#define TILE_SIZE 2048

unsigned int *sumArray, *sumArray2, *sumArray3;
int blocksL1, blocksL2, blocksL3;

// Host Helper Functions (allocate your own data structure...)
void preallocBlockSums(int numElements){
    blocksL1 = (int)ceil(numElements/(float)TILE_SIZE);
    cudaMalloc((void**) &sumArray, sizeof(unsigned int)*blocksL1);
    blocksL2 = (int)ceil(blocksL1/(float)TILE_SIZE);
    cudaMalloc((void**) &sumArray2, sizeof(unsigned int)*blocksL2);
    blocksL3 = (int)ceil(blocksL2/(float)TILE_SIZE);
    cudaMalloc((void**) &sumArray3, sizeof(unsigned int)*blocksL3);
}

void deallocBlockSums(){
    cudaFree(sumArray);
    cudaFree(sumArray2);
    cudaFree(sumArray3);
}

// Device Functions



// Kernel Functions

__global__ void scanArray(unsigned int *outArray, unsigned int *inArray, unsigned int *sumArray, int numElements){
    __shared__ unsigned int tileArray[TILE_SIZE];
    int index = blockIdx.x*TILE_SIZE + threadIdx.x;
    if(index < numElements && (threadIdx.x!=0 || blockIdx.x!=0))
        tileArray[threadIdx.x] = inArray[index-1];
    else
        tileArray[threadIdx.x] = 0;
    if(index+BLOCK_SIZE < numElements)
        tileArray[threadIdx.x + BLOCK_SIZE] = inArray[index-1 + BLOCK_SIZE];
    else
        tileArray[threadIdx.x + BLOCK_SIZE] = 0;
    unsigned int id, stride;
    for(stride=1;stride<TILE_SIZE;stride *= 2){
        __syncthreads();
        id = (threadIdx.x+1) * 2 * stride - 1;
        if(id<TILE_SIZE)
            tileArray[id] += tileArray[id-stride];
    }

    for(stride=TILE_SIZE/4;stride>0;stride /= 2){
        __syncthreads();
        id = (threadIdx.x+1) * 2 * stride - 1;
        if(id + stride < TILE_SIZE)
            tileArray[id+stride] += tileArray[id];
    }
    
    __syncthreads();
    if(threadIdx.x==0)
        sumArray[blockIdx.x] = tileArray[TILE_SIZE-1];
    if(index < numElements)
        outArray[index] = tileArray[threadIdx.x];
    if(index + BLOCK_SIZE < numElements)
        outArray[index+BLOCK_SIZE] = tileArray[threadIdx.x+BLOCK_SIZE]; 
}

__global__ void vectorAddition(unsigned int *vector, unsigned int *sumVector, int numElements){
    int index = blockIdx.x*TILE_SIZE + threadIdx.x;
    if(index < numElements){
        vector[index] += sumVector[blockIdx.x];
    }
    if(index + BLOCK_SIZE < numElements){
        vector[index + BLOCK_SIZE] += sumVector[blockIdx.x];
    }
}

// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
    scanArray<<<blocksL1, BLOCK_SIZE>>>(outArray, inArray, sumArray, numElements);
    if(blocksL1 > 1){
        // execute level 2 if more than one block in level 1
        scanArray<<<blocksL2, BLOCK_SIZE>>>(sumArray, sumArray, sumArray2, blocksL1);
        if(blocksL2 > 1){
            // execute level 3 if more than one block in level 2 
            // this should ideally have just one block
            scanArray<<<blocksL3, BLOCK_SIZE>>>(sumArray2, sumArray2, sumArray3, blocksL2);
            vectorAddition<<<blocksL2, BLOCK_SIZE>>>(sumArray, sumArray2, blocksL1);
        }
        vectorAddition<<<blocksL1, BLOCK_SIZE>>>(outArray, sumArray, numElements);
    }
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
