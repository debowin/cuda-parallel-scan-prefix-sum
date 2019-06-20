#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
// includes, kernels
#include "scan_largearray_kernel.cu"

#define DEFAULT_NUM_ELEMENTS 16777216
#define MAX_RAND 3


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" 
unsigned int compare( const unsigned int* reference, const unsigned int* data, 
                     const unsigned int len);
extern "C" 
void computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len);
bool CompareArrays(unsigned int *A, unsigned int *B, int size);
void WriteFile(unsigned int* arr, char* file_name, int num_elements);
int ReadParamsFile(int* params, char* file_name, int num_params);
int ReadFile(unsigned int* arr, char* file_name, int num_elements);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_read = 0;
    srand(time(NULL));
    int* size = (int*)malloc(1 * sizeof(int));
    unsigned int data2read = 1;
    int num_elements = 0; // Must support large, non-power-of-2 arrays

    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( unsigned int) * num_elements;
    unsigned int* h_data = (unsigned int*) malloc( mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Randomly generate input data and write the result to
    //   file name specified by first argument
    // * Two arguments: Read the first argument which indicate the size of the array,
    //   randomly generate input data and write the input data
    //   to the second argument. (for generating random input data)
    // * Three arguments: Read the first file which indicate the size of the array,
    //   then input data from the file name specified by 2nd argument and write the
    //   SCAN output to file name specified by the 3rd argument.
    switch(argc-1)
    {      
        case 2: 
            // Determine size of array
            data2read = ReadParamsFile(size, argv[1], data2read);
            if(data2read != 1){
                printf("Error reading parameter file\n");
                exit(1);
            }

            num_elements = size[0];

            // allocate host memory to store the input data
            mem_size = sizeof( unsigned int) * num_elements;
            h_data = (unsigned int*) malloc( mem_size);

            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = (int)(rand() % MAX_RAND);
            }
            WriteFile(h_data, argv[2], num_elements);
        break;
    
        case 3:  // Three Arguments
            data2read = ReadParamsFile(size, argv[1], data2read);
            if(data2read != 1){
                printf("Error reading parameter file\n");
                exit(1);
            }

            num_elements = size[0];
            
            // allocate host memory to store the input data
            mem_size = sizeof( unsigned int) * num_elements;
            h_data = (unsigned int*) malloc( mem_size);

            num_read = ReadFile(h_data, argv[2], num_elements);
            if(num_read != num_elements)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            // Use DEFAULT_NUM_ELEMENTS num_elements
            num_elements = DEFAULT_NUM_ELEMENTS;
            
            // allocate host memory to store the input data
            mem_size = sizeof( unsigned int) * num_elements;
            h_data = (unsigned int*) malloc( mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND);
            }
        break;  
    }    
   
    // compute reference solution
    unsigned int* reference = (unsigned int*) malloc( mem_size); 
    struct timeval start_time, end_time;
    gettimeofday(&start_time,NULL);
    computeGold( reference, h_data, num_elements);
    gettimeofday(&end_time,NULL);
    printf("Processing %d elements...\n", num_elements);
    double start_count = (double) start_time.tv_sec
        + 1.e-6 * (double) start_time.tv_usec;
    double end_count = (double) end_time.tv_sec +
        1.e-6 * (double) end_time.tv_usec;
    double host_ms = (double)( (end_count - start_count) * 1000);
    printf("CPU Processing time: %lf (ms)\n", host_ms);

    // allocate device memory input and output arrays
    unsigned int* d_idata = NULL;
    unsigned int* d_odata = NULL;

    int padded_num_elements = TILE_SIZE*((num_elements+TILE_SIZE-1)/TILE_SIZE);
    int padded_mem_size = padded_num_elements *sizeof(unsigned int);

    // Make a padded copy of the input data
    unsigned int* padded_hdata = (unsigned int*) malloc(padded_mem_size);
    memcpy(padded_hdata, h_data, mem_size);
    memset(padded_hdata+num_elements, 0, padded_mem_size - mem_size);

    cudaMalloc( (void**) &d_idata, padded_mem_size);
    cudaMalloc( (void**) &d_odata, padded_mem_size);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, padded_hdata, padded_mem_size, cudaMemcpyHostToDevice);
    // initialize all the other device arrays to be safe
    cudaMemcpy( d_odata, padded_hdata, padded_mem_size, cudaMemcpyHostToDevice);

    free(padded_hdata);
    padded_hdata = NULL;
    // **===--------------- Allocate data structure here --------------===**
    preallocBlockSums(num_elements);
    // **===-----------------------------------------------------------===**

    // Run just once to remove startup overhead for more accurate performance 
    // measurement
    prescanArray(d_odata, d_idata, TILE_SIZE);

    // Run the prescan
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    cudaEventRecord(start); 
    // **===-------------- Modify the body of this function -----------===**
    prescanArray(d_odata, d_idata, padded_num_elements);
    // **===-----------------------------------------------------------===**
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float device_ms = 0;
    cudaEventElapsedTime(&device_ms, start, stop);

    printf("GPU Processing time: %f (ms)\n", device_ms);
    printf("Speedup: %fX\n", host_ms/device_ms);

    // **===--------------- Deallocate data structure here ------------===**
    deallocBlockSums();
    // **===-----------------------------------------------------------===**


    // copy result from device to host
    cudaMemcpy( h_data, d_odata, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost);

    if ((argc - 1) == 3)  // Three Arguments, write result to file
    {
        WriteFile(h_data, argv[3], num_elements);
    }
    else if ((argc - 1) == 1)  // One Argument, write result to file
    {
        WriteFile(h_data, argv[1], num_elements);
    }


    // Check if the result is equivalent to the expected soluion
    unsigned int result_regtest = CompareArrays( reference, h_data, num_elements);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

    // cleanup memory
    free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
}


// Read a floating point array in from file
int ReadFile(unsigned int* arr, char* file_name, int num_elements)
{
    FILE* input = fopen(file_name, "r");
    if (input == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }
    for (unsigned i = 0; i < num_elements; i++) 
        fscanf(input, "%d", &(arr[i]));
    return num_elements;
}

// Read params of input matrices
int ReadParamsFile(int* params, char* file_name, int num_params)
{
    FILE* input = fopen(file_name, "r");
    if (input == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }
    for (unsigned i = 0; i < num_params; i++) 
        fscanf(input, "%d", &(params[i]));
    return num_params;
}

// Write a 16x16 floating point matrix to file
void WriteFile(unsigned int* arr, char* file_name, int num_elements)
{
    FILE* output = fopen(file_name, "w");
    if (output == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }
    for (unsigned i = 0; i < num_elements; i++) {
        fprintf(output, "%d ", arr[i]);
    }
}

// returns true iff A and B have same elements in same order
bool CompareArrays(unsigned int *A, unsigned int *B, int size) {
    for (unsigned i = 0; i < size; i++)
        if (fabs(A[i] - B[i]) > 0.001f)
            return false;
    return true;
}
