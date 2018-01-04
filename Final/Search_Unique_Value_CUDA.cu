// C++ Libraries.
#include <iostream>

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"

// Define max number of concurrent threads.
#define MAX_BLOCKSIZE 512

// Define optimal number of search inquries per thread.
#define OPTIMAL_INQUIRES 12





////////////////////////////////////////////////////////////////////////////////
/// Unrolled Coalesced N Element Search ///
////////////////////////////////////////////////////////////////////////////////

/**
 * Searches dev_Array for the given unique value by having each thread search adjacent to each other in
 * a coalesced fashion, followed by searching adjacent to the other threads again but offset by the total
 * number of threads. All for loops are unrolled with #pragma.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param numToCheck Number of elements each thread will check.
 * @param numOfThreads Total number of threads searching the array.
 * @param arraySize Number of elements in the given array to search.
 * @param dev_foundIndex Output index of the found unique value.
 */
__global__ void dev_Unrolled_Coalesced_N_Search(int *dev_Array, int uniqueValue, int numToCheck, int numOfThreads, int arraySize, int *dev_foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize currentValue and actualIndex to register memory.
    int currentValue, actualIndex;

    // Iterate through offset N number of adjacent elements.

#pragma unroll
    for (int N = 0; N < numToCheck; N++){

        // Calculate actual array index.
        actualIndex = numOfThreads * N + tid;

        // Ensure thread is not out of bounds.
        if ( actualIndex < arraySize ) {

            // Retrieve current value from global memory to be checked.
            currentValue = dev_Array[actualIndex];

            // Check if current value is the unique value.
            if ( currentValue == uniqueValue ) {
                // Unique value found, store its index in the foundIndex global memory variable.
                *dev_foundIndex = actualIndex;
            }
        }
    }
}



/**
 * Wrapper function to call the CUDA kernel device function dev_Unrolled_Coalesced_N_Search.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param numToCheck Number of elements each thread will check.
 * @param arraySize Number of elements in the given array to search.
 * @return Return the index of the unique value.
 */
int Unrolled_Coalesced_N_Search(int *dev_Array, int uniqueValue, int numToCheck, int arraySize) {
    // Initialize foundIndex integer.
    int foundIndex = -1;
    // Initialize foundIndex device pointer.
    int *dev_foundIndex;
    // Allocate memory on device for foundIndex.
    cudaMalloc((void**)&dev_foundIndex, sizeof(int));
    // Copy foundIndex initialized value to device.
    cudaMemcpy(dev_foundIndex, &foundIndex, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of threads expected.
    int numOfThreads = arraySize / numToCheck + 1;
    // Initialize blocksize as the number of threads.
    dim3 blockSize(MAX_BLOCKSIZE, 1, 1);
    dim3 gridSize(numOfThreads / MAX_BLOCKSIZE + 1, 1);

    // Launch device Strided_Offset_N_Search kernel routine.
    dev_Unrolled_Coalesced_N_Search<<<gridSize, blockSize>>>(dev_Array, uniqueValue, numToCheck, numOfThreads, arraySize, dev_foundIndex);

    // Copy d_foundIndex device value back to host memory.
    cudaMemcpy(&foundIndex, dev_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);

    // Return found index.
    return foundIndex;
}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////









/**
 * Run a test case for unrolled coalesced array search.
 */
int main(){

    // Define unique value to search for.
    const int uniqueValue = 5;
    // Define random index the unique value will be for constructing the searchable array.
    const int randomIndex = 68;
    // Define the size of our array.
    const int arraySize = 500000;

    // Initialize test array that we will search.
    int testArray[arraySize];
    // Set array to all zeros.
    for (int i = 0; i < arraySize; i++){
        testArray[i] = 0;
    }
    // Set random index to value to search for.
    testArray[randomIndex] = uniqueValue;


    // CUDA ALLOCATIONS //

    // Initialize device pointers.
    int *d_testArray, d_foundIndex;

    // Allocate memory for local variables on the GPU device.
    cudaMalloc((void**)&d_testArray,  arraySize * sizeof(int));
    cudaMalloc((void**)&d_foundIndex, sizeof(int));

    // Transfer test array from local host memory to device.
    cudaMemcpy(d_testArray, testArray, arraySize * sizeof(int), cudaMemcpyHostToDevice);




    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Each thread searches through N elements in a coalesced fashion where each thread begins its search
    // adjacent to the previous and following threads starting positions. From there, the threads search the
    // value which is the total number of threads offset from the current position, so that all threads are
    // still making coalesced memory calls. For loop is unroll with #pragma.
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // Initialize output index as invalid index.
    int foundIndex = -1;

    // Search for unique value using CUDA.
    foundIndex = Unrolled_Coalesced_N_Search(d_testArray, uniqueValue, OPTIMAL_INQUIRES, arraySize);

    // Print out index of found unique value.
    std::cout << "Located unique value at index = " << foundIndex << std::endl;

}







