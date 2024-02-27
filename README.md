# Distributed-Memory Particle Simulation with MPI

Project for the course CS 267 (Applications of Parallel Computers) at UC Berkeley in Spring 2023. Bin parallelization of particle simulation akin molecular dynamics with 1D row and 2D box parallelization schemes. Impelementation includes cross-thread communication of ghost particles as well thread-to-thread particle transfer as they might cross bin boundaries. Simulations and benchmarking of up to 6 million particles. 1D parallelization scripts does not achieve complete correctectness, with small numerical deviation from reference trajectory. Final submission with further parallelized 2D scheme. 

Authors: Pedro G. Martins, Danush Reddyrand and  Akul Arora. 
