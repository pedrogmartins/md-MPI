# Distributed-Memory Particle Simulation with MPI

Project for the course CS 267 (Applications of Parallel Computers) at UC Berkeley in Spring 2023. Bin parallelization of particle simulation akin molecular dynamics with 1D row and 2D box parallelization schemes. Impelementation includes cross-thread communication of ghost particles as well thread-to-thread particle transfer as they might cross bin boundaries. Simulations and benchmarking of up to 6 million particles. 1D parallelization scripts does not achieve complete correctectness, withsmall numerical deviation from reference trajectory.  

Authors: Pedro G. Martins (mpi-1d.cpp), Danush Reddyrand (mpi-2d.cpp) and  Akul Arora (mpi-2d.cpp). 
