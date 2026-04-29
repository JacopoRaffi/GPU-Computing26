# Reduction exercises

Sum the elements of a given array by using warp-reduction kernel. The exercise is to implement kernel functions, includes:
    1) Execute the first reduction on the large array by using more than one block
    2) Implement reduction per block by calling the reduction per warp.
    3) Perform partial reductoin per warp

Check the correctness by comparing with CPU result. Compare the performance against the provided kernel solutions.

Note: 
    1) The template defines the format of input parameter: argv[1] is to set the array size, argv[2] is to set debug flag, e.g., argv[2] > 1, then print temp_varaible results.








