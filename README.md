# Analysis of CUDA algorithms - 
## Search Unsorted Array for Unique Value

The focus of this post is to determine the optimal algorithm to search an unsorted array of unique elements for a particular element, or a random index of one of the non-unique elements matching the searched for element in an unsorted array. 

In standard programming on the CPU, the typical algorithm to search an unsorted array is to sequentially check every index of an array until the particular value of interest is found. This operation is of _O(n)_ complexity and on average takes _n/2_ inquiries to locate the unique value. While linear time complexities are already efficient, we can still be limited by the sequential nature of a search on the CPU. 

One option is to sort the array of elements so that searching can be performed by an algorithm with a more efficient complexity. Once sorted, searching for a unique value can be done with an algorithm such as binary search, which has a very efficient _O(logn)_ time complexity. However, this may not be preferable due to a host of reasons including memory size limitations or data immutability.

Another option is that we can leverage the massively parallelized capability of CUDA-enabled GPU's to speed up our searches. In fact, this may be the only option in cases where the array of elements is already contained in the GPU device's memory. And in this case, we can search for our element without any data transfer between host and device or changing any data. However, for arrays that live on the host memory and not the GPU device, we must pay the cost to transfer the array to the device before searching, but this may still be faster than the sequential search of the CPU.

### Strided N Search

The first way we can try to design the searching algorithm is to use a _strided_ approach. By this, we mean that if we have _N_ elements to search, we will deploy some _m_ threads (where _m_ < _N_) so that each thread searches _k_ elements before returning. However, there are multiple ways to do this. 

The most obvious way is to have each thread look at its _k_ elements using strided memory accesses. This means that the first thread will begin its search at index _0_ and end at index _k-1_, the second thread will begin at index _k_ and end at index _2k-1_, and so on. Visually, we can see this type of memory access in the following diagram where each thread will search a total of _k=3_ elements.

```
Iteration 1:
thread 0          thread 1          thread 2
   |                 |                 |
   |                 |                 |
   v                 v                 v
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  | ... |  N-1  |
```

```
Iteration 2:
       thread 0         thread 1          thread 2
         |                 |                 |
         |                 |                 |
         v                 v                 v
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  | ... |  N-1  |
```

```
Iteration 3:
            thread 0          thread 1          thread 2
               |                 |                 |
               |                 |                 |
               v                 v                 v
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  | ... |  N-1  |
```

The next question to ask is: how many elements should each thread search through? Surely, there must be an optimal number. That question is not always an easy one to ask, but what we can do is try multiple values of _k_ to see if there is a trend.

The CUDA kernel for this function is given as follows:

```
__global__ void dev_Strided_Offset_N_Search(int *array, int uniqueValue, int offset, int arraySize, int *foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize currentValue and actualIndex to register memory.
    int currentValue, actualIndex;

    // Iterate through offset N number of adjacent elements.
    for (int k = 0; k < offset; k++){

        // Calculate actual array index.
        actualIndex = tid * offset + k;

        // Ensure thread is not out of bounds.
        if ( actualIndex < arraySize ) {

            // Retrieve current value from global memory to be checked.
            currentValue = array[actualIndex];

            // Check if current value is the unique value.
            if ( currentValue == uniqueValue ) {
                // Unique value found, store its index in the foundIndex global memory variable.
                *foundIndex = actualIndex;
            }
        }
    }
}
```

This function takes in a pointer to an array of elements to be searched (in this case an array of integers), the unique value you are searching for, the number of elements each thread should search (also called the offset), the number of elements in the array, and finally a pointer to an integer that represents the index of the located elements. The first few lines are standard CUDA and C commands such as determining the thread's id and initializing integers for later in the function. Next, the function will iterate its search for _k_ elements (_offset_). In the iteration stage, the function determines the index of the array to search next (_actualIndex_), ensure the index is within the bounds of the array, and retrieve the value stored at the current index. If this value matches the element that is being searched for, that thread will write the index of that element to the input parameter pointer _foundIndex_. However, this pointer is accessible to all threads, so it may be overwritten if another instance of the element is found.

While this method works, we can still improve our algorithm. Using strided memory access calls, as illustrated above, is not the most efficient way to access global memory. Instead, we can reorganize what we have above to make our global memory accesses more efficient using _coalesced_ memory accesses.



### Coalesced N Search

A simplified definition for a coalesced memory access is when all threads in a warp access adjacent memory locations at once. In the example of the strided N search, the threads in each warp (sets of 32 threads) were not accessing adjacent data. Rather, they were accessing data that was separated by _k_ elements. It can be shown that the bandwidth of the of the GPU (effective speed) is slowed significantly by increasing the separation of the accessed memory elements.

Therefore, in order to make our algorithm faster, we want all of our threads to be accessing memory locations that are adjacent to each other in each iteration. In the first iteration, the first thread will search index 0, the second thread will search index 1, and so on. In the next iteration, the first thread will search at index _m_ (where _m_ is the total number of threads), the second thread will search at _m+1_, and so on. As before, we illustrate the first two iterations of this visually where there are a total of _m=4_ threads:

```
Iteration 1:
  tid0  tid1  tid2  tid3
   |     |     |     |
   |     |     |     |
   v     v     v     v
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  | ... |  N-1  |
```

```
Iteration 2:
                          tid0  tid1  tid2  tid3
                           |     |     |     |
                           |     |     |     |
                           v     v     v     v
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  | ... |  N-1  |
```
This improvement is implemented in the following CUDA code,

```__global__ void dev_Coalesced_N_Search(int *array, int uniqueValue, int numToCheck, int numOfThreads, int arraySize, int *foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize currentValue and actualIndex to register memory.
    int currentValue, actualIndex;

    // Iterate through offset N number of adjacent elements.
    for (int k = 0; k < numToCheck; k++){

        // Calculate actual array index.
        actualIndex = numOfThreads * k + tid;

        // Ensure thread is not out of bounds.
        if ( actualIndex < arraySize ) {

            // Retrieve current value from global memory to be checked.
            currentValue = array[actualIndex];

            // Check if current value is the unique value.
            if ( currentValue == uniqueValue ) {
                // Unique value found, store its index in the foundIndex global memory variable.
                *foundIndex = actualIndex;
            }
        }
    }
}
```
where the only major difference the _actualIndex_ variable which controls the index that the thread will search in the current iteration.

By having all of our threads in a warp access adjacent memory locations, the speed of our code will improve. But again, we are faced with the question of how many threads should we launch to tackle this problem (or asking the same question: how many elements does each thread search?). Again, we can run this code with a set of varying _k_ or _m_ values and find out.

### Unrolled Coalesced N Search

From here, we can make another small improvement. When we iterate through the for loop in the coalesced example above, we can actually _unroll_ this loop. What that means is instead of writing a small set of instructions and looping over the set, we can instead simply write out all of the instructions at once, which actually gives us a small performance benefit. Although we could do this by hand, it is much easier to add a single line _#pragma unroll_, which will do this for us.

We add this #pragma command to the above coalesced N search example above:

```
__global__ void dev_Unrolled_Coalesced_N_Search(int *array, int uniqueValue, int numToCheck, int numOfThreads, int arraySize, int *foundIndex){
    // Calculate thread id.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize currentValue and actualIndex to register memory.
    int currentValue, actualIndex;

    // Iterate through offset N number of adjacent elements.
    for (int k = 0; k < numToCheck; k++){

        // Calculate actual array index.
        actualIndex = numOfThreads * k + tid;

        // Ensure thread is not out of bounds.
        if ( actualIndex < arraySize ) {
        
            // Retrieve current value from global memory to be checked.
            currentValue = array[actualIndex];
            
            // Check if current value is the unique value.
            if ( currentValue == uniqueValue ) {
                // Unique value found, store its index in the foundIndex global memory variable.
                *foundIndex = actualIndex;
            }
        }
    }
}
```

### Full Coalesced Search







<img src="https://github.com/TravisWThompson1/Algorithm_Analysis-CUDA_Search_Unsorted_Array_for_Unique_Value/blob/master/Data/Algorithm_Runtime_Analysis.png" width="600">



<img src="https://github.com/TravisWThompson1/Algorithm_Analysis-CUDA_Search_Unsorted_Array_for_Unique_Value/blob/master/Data/Analysis_Optimal_Unrolled_Full.png" width="600">


