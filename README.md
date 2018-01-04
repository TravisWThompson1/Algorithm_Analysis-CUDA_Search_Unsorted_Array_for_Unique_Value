# Analysis of CUDA algorithms - 
## Search Unsorted Array for Unique Value

The focus of this post is to determine the optimal algorithm to search an unsorted array of unique elements for a particular element, or a random index of one of the non-unique elements matching the searched for element in an unsorted array. 

In standard programming on the CPU, the typical algorithm to search an unsorted array is to sequentially check every index of an array until the particular value of interest is found. This operation is of _O(n)_ complexity and on average takes _n/2_ inquiries to locate the unique value. While linear time complexities are already efficient, we can still be limited by the sequential nature of a search on the CPU. 

One option is to sort the array of elements so that searching can be performed by an algorithm with a more efficient complexity. However, this may not be preferable due to a host of reasons including memory size limitations or data immutability.

Another option is that we can leverage the massively parallelized capability of CUDA-enabled GPU's to speed up our searches. In fact, this may be the only option in cases where the array of elements is already contained in the GPU device's memory. And in this case, we can search for our element without any data transfer between host and device. However, for arrays that live on the host memory and not the GPU device, we must pay the cost to transfer the array to the device before searching, but this may still be faster than the sequential search of the CPU.








