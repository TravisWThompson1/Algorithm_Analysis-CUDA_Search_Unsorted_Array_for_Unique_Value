// C++ Libraries.
#include <iostream>

// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"


// Define max number of concurrent threads
#define MAX_BLOCKSIZE 512



////////////////////////////////////////////////////////////////////////////////
/// 1. Strided Offset N Search ///
////////////////////////////////////////////////////////////////////////////////


/**
 * Searches dev_Array for the given unique value by having each thread search 'offset' number of elements from
 * the previous thread's search (strided offset).
 * @param dev_Array Array to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param offset Number of elements each thread will search, and the separation between each thread's starting index.
 * @param arraySize Number of elements in the given array to search.
 * @param dev_foundIndex Output index of the found unique value.
 */
__global__ void dev_Strided_Offset_N_Search(int *dev_Array, int uniqueValue, int offset, int arraySize, int *dev_foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize currentValue and actualIndex to register memory.
    int currentValue, actualIndex;

    // Iterate through offset N number of adjacent elements.
    for (int N = 0; N < offset; N++){

        // Calculate actual array index.
        actualIndex = tid * offset + N;

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
 * Wrapper function to call the CUDA kernel device function dev_Strided_Offset_N_Search.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param offset Number of elements each thread will search, and the separation between each thread's starting index.
 * @param arraySize Number of elements in the given array to search.
 * @return Return the index of the unique value.
 */
int Strided_Offset_N_Search(int *dev_Array, int uniqueValue, int offset, int arraySize){
    // Initialize foundIndex integer.
    int foundIndex = -1;
    // Initialize foundIndex device pointer.
    int *dev_foundIndex;
    // Allocate memory on device for foundIndex.
    cudaMalloc((void**)&dev_foundIndex, sizeof(int));
    // Copy foundIndex initialized value to device.
    cudaMemcpy(dev_foundIndex, &foundIndex, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of threads expected.
    int numOfThreads = arraySize / offset + 1;

    // Initiaize CUDA event timers.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize blocksize as the number of threads.
    dim3 blockSize(MAX_BLOCKSIZE, 1, 1);
    dim3 gridSize(numOfThreads / MAX_BLOCKSIZE + 1, 1);

    // Launch device Strided_Offset_N_Search kernel routine and start and stop event timers.
    cudaEventRecord(start);
    dev_Strided_Offset_N_Search<<<gridSize, blockSize>>>(dev_Array, uniqueValue, offset, arraySize, dev_foundIndex);
    cudaEventRecord(stop);

    // Retrieve kernel timing.
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Print event timing.
    std::cout << "CUDA kernel :: Strided_Offset_N_Search :: " << milliseconds << "ms elapsed." << std::endl;

    // Copy d_foundIndex device value back to host memory.
    cudaMemcpy(&foundIndex, dev_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);

    // Return found index.
    return foundIndex;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
/// 2. Coalesced N Element Search ///
////////////////////////////////////////////////////////////////////////////////

/**
 * Searches dev_Array for the given unique value by having each thread search adjacent to each other in
 * a coalesced fashion, followed by searching adjacent to the other threads again but offset by the total
 * number of threads.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param numToCheck Number of elements each thread will check.
 * @param numOfThreads Total number of threads searching the array.
 * @param arraySize Number of elements in the given array to search.
 * @param dev_foundIndex Output index of the found unique value.
 */
__global__ void dev_Coalesced_N_Search(int *dev_Array, int uniqueValue, int numToCheck, int numOfThreads, int arraySize, int *dev_foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize currentValue and actualIndex to register memory.
    int currentValue, actualIndex;

    // Iterate through offset N number of adjacent elements.
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
 * Wrapper function to call the CUDA kernel device function dev_Coalesced_N_Search.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param numToCheck Number of elements each thread will check.
 * @param arraySize Number of elements in the given array to search.
 * @return Return the index of the unique value.
 */
int Coalesced_N_Search(int *dev_Array, int uniqueValue, int numToCheck, int arraySize) {
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

    // Initiaize CUDA event timers.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize blocksize as the number of threads.
    dim3 blockSize(MAX_BLOCKSIZE, 1, 1);
    dim3 gridSize(numOfThreads / MAX_BLOCKSIZE + 1, 1);

    // Launch device Strided_Offset_N_Search kernel routine and start and stop event timers.
    cudaEventRecord(start);
    dev_Coalesced_N_Search<<<gridSize, blockSize>>>(dev_Array, uniqueValue, numToCheck, numOfThreads, arraySize, dev_foundIndex);
    cudaEventRecord(stop);

    // Retrieve kernel timing.
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Print event timing.
    std::cout << "CUDA kernel :: Coalesced_N_Search :: " << milliseconds << "ms elapsed." << std::endl;

    // Copy d_foundIndex device value back to host memory.
    cudaMemcpy(&foundIndex, dev_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);

    // Return found index.
    return foundIndex;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////







////////////////////////////////////////////////////////////////////////////////
/// 3. Full Coalesced Element Search ///
////////////////////////////////////////////////////////////////////////////////

/**
 * Searches dev_Array for the given unique value by having each thread search adjacent to each other in
 * a coalesced fashion.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param arraySize Number of elements in the given array to search.
 * @param dev_foundIndex Output index of the found unique value.
 */
__global__ void dev_Full_Coalesced_Search(int *dev_Array, int uniqueValue, int arraySize, int *dev_foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Retrieve current value from global memory to be checked.
    int currentValue = dev_Array[tid];

    // Check if current value is the unique value.
    if ( currentValue == uniqueValue ) {
        // Unique value found, store its index in the foundIndex global memory variable.
        *dev_foundIndex = tid;
    }
}



/**
 * Wrapper function to call the CUDA kernel device function dev_Coalesced_N_Search.
 * @param dev_Array Array in device memory to be searched.
 * @param uniqueValue Unique value to be searched for.
 * @param arraySize Number of elements in the given array to search.
 * @return Return the index of the unique value.
 */
int Full_Coalesced_Search(int *dev_Array, int uniqueValue, int arraySize) {
    // Initialize foundIndex integer.
    int foundIndex = -1;
    // Initialize foundIndex device pointer.
    int *dev_foundIndex;
    // Allocate memory on device for foundIndex.
    cudaMalloc((void**)&dev_foundIndex, sizeof(int));
    // Copy foundIndex initialized value to device.
    cudaMemcpy(dev_foundIndex, &foundIndex, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of threads expected.
    int numOfThreads = arraySize;

    // Initiaize CUDA event timers.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize blocksize as the number of threads.
    dim3 blockSize(MAX_BLOCKSIZE, 1, 1);
    dim3 gridSize(numOfThreads / MAX_BLOCKSIZE + 1, 1);

    // Launch device Strided_Offset_N_Search kernel routine and start and stop event timers.
    cudaEventRecord(start);
    dev_Full_Coalesced_Search<<<gridSize, blockSize>>>(dev_Array, uniqueValue, arraySize, dev_foundIndex);
    cudaEventRecord(stop);

    // Retrieve kernel timing.
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Print event timing.
    std::cout << "CUDA kernel :: Coalesced_N_Search :: " << milliseconds << "ms elapsed." << std::endl;

    // Copy d_foundIndex device value back to host memory.
    cudaMemcpy(&foundIndex, dev_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);

    // Return found index.
    return foundIndex;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////







int main(){

    // Define unique value to search for.
    const int uniqueValue = 5;
    // Define random index the unique value will be for constructing the searchable array.
    const int randomIndex = 68;
    // Define the size of our array.
    const int arraySize = 150000;

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



    // Find unique values //
    int foundIndex = -1;


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Each thread searches through N adjacent elements where each thread begins its search N elements
    // from the previous thread's starting position. If a thread successfully locates the unique value, it
    // write the index of the element to memory.
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // Test multiple offset sizes.
    std::cout << "-- Strided Offset N Search --" << std::endl;
    int offset = 1;
    for (int offset = 1; offset < 50; offset+=1) {
        foundIndex = Strided_Offset_N_Search(d_testArray, uniqueValue, offset, arraySize);
    }
    // Print out index of found unique value.
    std::cout << "Located unique value at index = " << foundIndex << std::endl;




    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // 2. Each thread searches through N elements in a coalesced fashion where each thread begins its search
    // adjacent to the previous and following threads starting positions. From there, the threads search the
    // value which is the total number of threads offset from the current position, so that all threads are
    // still making coalesced memory calls.
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // Test multiple values of N, where N is the number of elements each thread will check.
    std::cout << "-- Coalesced N Search --" << std::endl;
    int numToCheck = 1;
    for (int numToCheck = 1; numToCheck < 50; numToCheck+=1) {
        foundIndex = Coalesced_N_Search(d_testArray, uniqueValue, numToCheck, arraySize);
    }
    // Print out index of found unique value.
    std::cout << "Located unique value at index = " << foundIndex << std::endl;





    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // 3. Each thread searches a single elements in a coalesced fashion where each thread begins its search
    // adjacent to the previous and following threads starting positions.
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    // Test multiple values of N, where N is the number of elements each thread will check.
    std::cout << "-- Full Coalesced Search --" << std::endl;
    foundIndex = Full_Coalesced_Search(d_testArray, uniqueValue, arraySize);
    // Print out index of found unique value.
    std::cout << "Located unique value at index = " << foundIndex << std::endl;


}







