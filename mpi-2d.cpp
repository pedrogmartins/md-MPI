#include "common.h"
#include <mpi.h>
#include <set>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <stdio.h>



// Put any static global variables here that you will use throughout the simulation.

double size;
int num_bins_1d;
int num_bins_2d;
int num_processes;
int rows_per_process;
int num_processes_w_leftover_rows;
std::vector<int> rank_bins_ID; //Collects the index of all bins in this rank
typedef std::vector<particle_t*> bin_t;
bin_t* bins;
int* misplaced_array;
int* displacement;
double bin_width;

// We retrieve the functions from the previous serial implementation

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

   //Here we will implement square bins, with dimensions equal to the cutoff value
   num_bins_1d = (int) ceil(size/(cutoff));
   bin_width = size / num_bins_1d;
   num_bins_2d = num_bins_1d * num_bins_1d;

   //Create the bins object locally, so each MPI process will have a bins like object
   //   but only its assigned bins will be populated, but still possible to add more
   //   particles and ghost particles from nearby processors if needed.
   bins = new bin_t[num_bins_1d * num_bins_1d];

   //Each mpi rank will manage at least 'rows_per_process' rows, and
   //   'num_processes_w_leftover_rows' mpi ranks will manage 'rows_per_process' + 1 rows
   num_processes = std::min(num_procs, num_bins_1d);
   rows_per_process =  (int) floor(num_bins_1d / num_processes);
   num_processes_w_leftover_rows = num_bins_1d % num_processes;


   //Assign bins to each of the MPI ranks
   if (rank < num_processes) {

      //First, we collect the bins that shall be managed by this rank
      //   starting with the mpi ranks with 'rows_per_process' + 1 rows

      if (rank < num_processes_w_leftover_rows) {
         //Loop over the number of rows per process
         for (int i = 0; i < rows_per_process + 1; i++) {
            int row = rank * (rows_per_process + 1) + i;
            //Loop over all the bins in each row
            for (int j = 0; j < num_bins_1d; j++) {
               //Push bin ID # into the rank list rank_bins_ID
               rank_bins_ID.push_back(row*num_bins_1d + j);

            }
         }

      //Collecting bin ids for mpi ranks with 'rows_per_process' rows
      } else {
         for (int i = 0; i < rows_per_process; i++) {
         //Now for finding the row number, we need to consider all
         //  the previously assigned ones
            int row = ( num_processes_w_leftover_rows * (rows_per_process + 1)
                       + (rank - num_processes_w_leftover_rows) * rows_per_process + i);
            for (int j = 0; j < num_bins_1d; j++) {
               rank_bins_ID.push_back(row*num_bins_1d + j);
            }
         }
      }
   }

   //Now we need to actually assign all particles to each bin, each rank process should
   //   assign its particles to the bins object

   for (int i = 0; i < num_parts; i++) {

      if (rank == 0) {
          //std::cout << "PROCESSOR = " << rank << " PARTICLE ID = " << parts[i].id << std::endl;
      }

      int x = (int) floor(parts[i].x / bin_width);
      int y = (int) floor(parts[i].y / bin_width);
      int bin_id = y*num_bins_1d + x;

      if (rank == 0 && parts[i].id == 25) {
         //std::cout << "BIN WIDTH = " << bin_width << " SIZE = " << size << " X = " << x  << " Y = " << y << " Bin ID = " << bin_id << std::endl;
      }
      //Need to get now the MPI rank which corresponds to this bin

      int row_number = (int) floor(bin_id / num_bins_1d);

      int mpi_rank = 1000; //Define placeholder so we can access this variable outside if statement

      // For rows which belong to processors with rows_per_process + 1 rows
      if (row_number < num_processes_w_leftover_rows * (rows_per_process + 1)) {
         mpi_rank = (int) floor(row_number / (rows_per_process + 1));

      // For rows which belong to processors with rows_per_process rows
      } else {
         mpi_rank = (int) floor( (row_number - num_processes_w_leftover_rows * (rows_per_process + 1)) / rows_per_process
                         + num_processes_w_leftover_rows);
      }

      if (rank == 0) {
          //std::cout << "PROCESSOR = " << mpi_rank << " PARTICLE ID = " << parts[i].id << std::endl;
      }

      // Finally, if mpi_rank matches the current rank, we add the particle to the correct bin
      if (mpi_rank == rank) {
         bins[bin_id].push_back(&parts[i]);
         if (parts[i].id == 25) {
           // std::cout << "TOTAL # ROWS " << num_bins_1d << " # PROCESSES = " << num_processes << " ROWS PER PROCESSOR " << rows_per_process << " PROCS W 3 ROWS " << num_processes_w_leftover_rows << std::endl;
            //std::cout << "PROCESSOR = " << mpi_rank << " PARTICLE ID = " << parts[i].id << std::endl;
            //std::cout << "BIN ID = " << bin_id << " WHICH ROW = " << row_number  << std::endl;
            //std::cout << "PARTICLE X = " << parts[i].x << " PARTICLE Y = " << parts[i].y << std::endl;
         }
      }
   }

   //Allocate some space in memory for the arrays related to rebinning
   misplaced_array = (int*) malloc(num_procs * sizeof(int));
   displacement = (int*) malloc(num_procs * sizeof(int));

}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

   //The first step here would be to compute forces. Nonetheless, we need to make sure that
   //   each MPI process also has access to the particles sittin in nearby rows above and
   //   below, so forces can be computer in bins that sit nearby as well.

   //We will use MPI_Rec and MPI_Send to allow access to  bins from below and above rows
   //   in the current MPI Rank. Define first a list of request and a vector of vectors
   //   with particles to be sent/received per request

   std::vector <MPI_Request*> requests; //Vector to collect all requests so we can delete them later
   std::vector<std::vector<particle_t>*> send_buffers;

   // SEND PARTICLES

   //Check whether this current processor has rows above or below

   if (rank > 0 && rank <  num_processes) {
      //If true, this means this means we are not dealing with the first set of rows,
      //   but we haven't exceeded the number of processes as well

      //We have to now collect the particles in the bins in the top  row
      //  into a int vector so they can be sent to the other processor

      std::vector<int>* list_top = new std::vector<int>();
      int top_row;

      //Find top row index rows processors with 'rows_per_process' +1 rows
      if (rank < num_processes_w_leftover_rows) {
         top_row = rank * (rows_per_process + 1);
      } else {
         top_row = ( num_processes_w_leftover_rows * (rows_per_process + 1)
                + (rank - num_processes_w_leftover_rows) * rows_per_process);
      }

      //Collect bin IDs in the row above
      for (int i = 0; i < num_bins_1d; i++) {
         list_top->push_back(top_row + num_bins_1d * i);
      }

      //Collect particles from top row to send, create vector of particles
      std::vector<particle_t>* send_par_list = new std::vector<particle_t>();
      for (auto bin_id : *list_top) {
         for (auto part : bins[bin_id]) {
            send_par_list->push_back(*part);
         }
      }

      //Identify the rank of the processor managing particles above
      int process_above_rank = rank - 1;

      //Store particles send requests to that specific process in the send_buffed and send particles
      MPI_Request* request_send_up = new MPI_Request();
      MPI_Isend(&(*send_par_list)[0], send_par_list->size(), PARTICLE, process_above_rank, 0, MPI_COMM_WORLD, request_send_up);

      //Collect all particle list to send as well as all requests
      send_buffers.push_back(send_par_list);
      requests.push_back(request_send_up); //USED DOT OPERATION BECAUSE WE ARE NOT WORKING WITH THE
                                           //POINTER LIKE IN THE FUNCTION VERSION OF THIS IMPLEMENTATION
   }

   //Repeat now with the rows below, first check if it has rows below, rank
   //   indexation starts at 0

   if (rank < num_processes - 1) {

      std::vector<int>* list_bottom = new std::vector<int>();
      int bottom_row;

      //Find bottom row index rows processors with 'rows_per_process' +1 rows
      if (rank < num_processes_w_leftover_rows) {
         bottom_row = rank * (rows_per_process + 1) + rows_per_process;
      } else {
         bottom_row = ( num_processes_w_leftover_rows * (rows_per_process + 1)
                      + (rank - num_processes_w_leftover_rows) * rows_per_process
                      + rows_per_process - 1);
      }

      //Collect bin IDs in the row below
      for (int i = 0; i < num_bins_1d; i++) {
         list_bottom->push_back(bottom_row + num_bins_1d * i);
      }

      //Collect particles from bottom row to send, create vector of particles
      std::vector<particle_t>* send_par_list = new std::vector<particle_t>();
      for (auto bin_id : *list_bottom) {
         for (auto part : bins[bin_id]) {
            send_par_list->push_back(*part);
         }
      }

      //Identify the rank of the processor managing particles above
      int process_below_rank = rank + 1;

      //Send particles to that specific process
      MPI_Request* request_send_below = new MPI_Request();
      MPI_Isend(&(*send_par_list)[0], send_par_list->size(), PARTICLE, process_below_rank, 0, MPI_COMM_WORLD, request_send_below);

      //Collect all particle list to send as well as all requests
      send_buffers.push_back(send_par_list);
      requests.push_back(request_send_below);

   }

   //RECEIVE PARTICLES

   //Create buffer object to receive all the particles from other processors with enough space for alocating particles
   std::vector<particle_t*> receive_buffers;
   MPI_Status status; //struct associated with MPI processes status needed as argument for MPI_Recv
   std::set<int> surrounding_bin_ids; //List to keep track of bins which were modified

   //Receive particles from both the processor above and below if they exist

   //Go through all but the first MPI rank, should receive from the MPI process with rank = rank_current - 1
   if (rank > 0 && rank < num_processes) {

      particle_t* receive_buffer = new particle_t[99999];

      int process_above_rank = rank - 1;

      MPI_Recv(receive_buffer,
               num_parts, ///Does this need to be with this size????
               PARTICLE,
               process_above_rank,
               0,
               MPI_COMM_WORLD,
               &status
             );

      //Now, we need to assign the received particles to the correct bins
      int number_of_particles;
      MPI_Get_count(&status, PARTICLE, &number_of_particles);

      for (int i = 0; i < number_of_particles; i++) {
         int x = (int) (receive_buffer[i].x / bin_width);
         int y = (int) (receive_buffer[i].y / bin_width);
         int bin_id = y*num_bins_1d + x;
         bins[bin_id].push_back(&receive_buffer[i]);
         //Keep track of the bins to that were populated so they can be emptied after we finish all the calculations
         surrounding_bin_ids.insert(bin_id);// MADE THIS INSERT BECAUSE WE NO LONGER PASS ON THE POINTER TO A FUNCTION

      }

      receive_buffers.push_back(receive_buffer);

   }


   //Go through all but the last  MPI rank, should receive from the MPI process with rank = rank_current + 1
   if (rank < num_processes - 1) {

      particle_t* receive_buffer = new particle_t[99999];

      int process_below_rank = rank + 1;

      MPI_Recv(receive_buffer,
               num_parts, ///Does this need to be with this size????
               PARTICLE,
               process_below_rank,
               0,
               MPI_COMM_WORLD,
               &status
             );

      //Assign the received particles to the correct bins
      int number_of_particles;
      MPI_Get_count(&status, PARTICLE, &number_of_particles);

      for (int i = 0; i < number_of_particles; i++) {
         int x = (int) (receive_buffer[i].x / bin_width);
         int y = (int) (receive_buffer[i].y / bin_width);
         int bin_id = y*num_bins_1d + x;
         bins[bin_id].push_back(&receive_buffer[i]);
         //Keep track of the bins to that were populated so they can be emptied after we finish all the calculations
         surrounding_bin_ids.insert(bin_id);// MADE THIS INSERT BECAUSE WE NO LONGER PASS ON THE POINTER TO A FUNCTION

      }

      receive_buffers.push_back(receive_buffer);

   }

   //Perform force calculations, we now need to make sure to go through bins surrounding
   //   every bin in the rows assigned to this processor. We will need to check
   //   for every bin if we have particles assigned to nearby bin indices and
   //   loop over all particles in there.

   for (auto bin_id : rank_bins_ID) {
       for (auto particle : bins[bin_id]) {

            // Zero forces in each partcile
            particle->ax = 0;
            particle->ay = 0;

            //Need to make sure we are looping over all bins surrounding this bins.
            //  Unless the current bin is sitting at the borders of the system,
            //  the neighbor particles in side bins will be looped over.

            //Check if there is a bin immediately above
            if (bin_id - num_bins_1d > -1) {
               for (particle_t* part : bins[bin_id - num_bins_1d]) {
                  apply_force(*particle, *part);
                }
            }

            //Check if there is a bin immediately below
            if (bin_id + num_bins_1d < num_bins_2d) {
               for (particle_t* part : bins[bin_id + num_bins_1d]) {
                  apply_force(*particle, *part);
               }
            }


            //Check if there is a bin immediately to the left
            if (bin_id % num_bins_1d != 0) {
               for (particle_t* part : bins[bin_id - 1]) {
                  apply_force(*particle, *part);
               }
            }

            //Check if there is a bin immediately to the right
            if (bin_id % num_bins_1d != num_bins_1d - 1) {
               for (particle_t* part : bins[bin_id + 1]) {
                  apply_force(*particle, *part);
               }
            }

            //Deal with diagonally placed bins

            //Check if there is a bin northeast
            if (bin_id - num_bins_1d > -1 && bin_id % num_bins_1d != num_bins_1d - 1) {
               for (particle_t* part : bins[bin_id - num_bins_1d + 1]) {
                  apply_force(*particle, *part);
               }
            }

            //Check if there is a bin southeast
            if (bin_id + num_bins_1d < num_bins_2d && bin_id % num_bins_1d != num_bins_1d - 1) {
               for (particle_t* part : bins[bin_id + num_bins_1d + 1]) {
                  apply_force(*particle, *part);
               }
            }

            //Check if there is a bin southwest
            if (bin_id + num_bins_1d < num_bins_2d && bin_id % num_bins_1d != 0) {
               for (particle_t* part : bins[bin_id + num_bins_1d - 1]) {
                  apply_force(*particle, *part);
               }
            }

            //Check if there is a bin northwest
            if (bin_id - num_bins_1d > -1 && bin_id % num_bins_1d != 0) {
               for (particle_t* part : bins[bin_id - num_bins_1d - 1]) {
                  apply_force(*particle, *part);
               }
            }

            //Get interactions with other particles in this same bin
            for (particle_t* part : bins[bin_id]) {
               if (particle != part) {
                  apply_force(*particle, *part);
               }
            }

       }
   }

   //Make sure all the communication processes are completed and delete the requests and buffered data

    MPI_Status array_of_statuses[requests.size()];
    for (auto request : requests) {
        MPI_Status status;
        MPI_Wait(request, &status);
        delete request;
    }

    for (int i = 0; i < send_buffers.size(); i++) {
        delete send_buffers[i];
    }

    for (auto bin_id : surrounding_bin_ids) {
        bins[bin_id].clear();
    }

    for (int i = 0; i < receive_buffers.size(); i++) {
        delete[] receive_buffers[i];
    }


   //Move all the particles in bins managed by this thread

   for (auto bin_id : rank_bins_ID) {
      for (auto particle : bins[bin_id]) {
         move(*particle, size);
      }
   }

   //Add barrier to make sure all the threads are synchronized ready to move to the next step
   MPI_Barrier(MPI_COMM_WORLD);

   //The last task in this function is to now update the particles that might have moved
   //   to another bin in the same or even another processor. Loop over partciles and
   //   identify their new bin. Define first a vector of particles misplaced

   std::vector<particle_t> particles_misplaced;

   //Loop over all bins in this processor, and then over all particles in each bin

   for (int bin_ID : rank_bins_ID) {
      for(auto iteration = bins[bin_ID].begin() ; iteration != bins[bin_ID].end(); iteration ++) {

         particle_t* particle = *iteration;
         int new_x = (int) (particle->x / bin_width);
         int new_y = (int) (particle->y / bin_width);
         int new_bin_ID = new_y*num_bins_1d + new_x;

         //Now, we need to identify to which processor this new bins belongs to

         int row_number = (int) floor(new_bin_ID / num_bins_1d);

         int mpi_rank = 1000; //Define placeholder so we can access this variable outside if statement

         // For rows which belong to processors with rows_per_process + 1 rows
         if (row_number < num_processes_w_leftover_rows * (rows_per_process + 1)) {
            mpi_rank = (int) floor(row_number / (rows_per_process + 1));

         // For rows which belong to processors with rows_per_process rows
         } else {
            mpi_rank = (int) floor( (row_number - num_processes_w_leftover_rows * (rows_per_process + 1)) / rows_per_process
                         + num_processes_w_leftover_rows);
         }

         if (particle->id == 25) {
            std::cout << "PROCESSOR = " << mpi_rank << " PARTICLE ID = " << particle->id << std::endl;
            std::cout << "BIN ID = " << new_bin_ID << " WHICH ROW = " << row_number  << std::endl;
            std::cout << "PARTICLE X = " << particle->x << " PARTICLE Y = " << particle->y << std::endl;
            std::cout << " " << std::endl;
         }


         //Check if the rank corresponding to particle is different, and if bin is changede
         if (mpi_rank != rank) {
            particles_misplaced.push_back(**iteration);
            bins[bin_ID].erase(iteration--);
         } else if (new_bin_ID != bin_ID) {
            //Particle still in same processor, but now in a different bin
            bins[new_bin_ID].push_back(*iteration);
            bins[bin_ID].erase(iteration--);
         }
      }
   }

   //All particles that need to be moved accross processors have been identified
   //   We are left with sharing them among all processors and executing the new
   //   assignments.

   //Make sure all threads are at this point before we proceed.
   MPI_Barrier(MPI_COMM_WORLD);

   //Gathering the number of particles to be exchanged in each processor into a single vector
   int num_particles_misplaced = particles_misplaced.size();
   MPI_Allgather(&num_particles_misplaced, 1, MPI_INT, misplaced_array, 1, MPI_INT, MPI_COMM_WORLD);

   //Compute total number of misplaced particles for further indexing
   int total_num_misplaced = 0;
   for (int i = 0; i < num_procs; i++) {
      total_num_misplaced += misplaced_array[i];
   }

   //As we collect all the particles in another gather, we need to know
   //   how many units of displacement to place data from each processor
   //   so nothing gets overwritten. First gather is placed at index 0

   displacement[0] = 0;
   for (int i = 0; i < num_procs; i++) {
      displacement[i] = displacement[i-1] + misplaced_array[i-1];
   }

   //Create a new array for all the misplaced partciles to be acommodated, large enoguh
   particle_t* particles = new particle_t[num_parts];


   //Gather all the particles starting at the first entry in the misplaced particles list
   //  and space writes by the correct displacement per processor to avoid overwrites

   MPI_Allgatherv(&particles_misplaced[0],
                 num_particles_misplaced,
                 PARTICLE,
		 particles,
		 misplaced_array,
		 displacement,
		 PARTICLE,
		 MPI_COMM_WORLD
		);

    //Loop over all the misplaced particles and assign them to the correct processors
    for (int i = 0; i < total_num_misplaced; i++) {

       //First we get the correct bin assignment for this particle
       particle_t particle = particles[i];
       int new_x = (int) (particle.x / bin_width);
       int new_y = (int) (particle.y / bin_width);
       int new_bin_ID = new_y*num_bins_1d + new_x;

       //Get the corresponding processor to this new bin_ID
       int row_number = (int) floor(new_bin_ID / num_bins_1d);

       int mpi_rank = 1000; //Define placeholder so we can access this variable outside if statement

       // For rows which belong to processors with rows_per_process + 1 rows
       if (row_number < num_processes_w_leftover_rows * (rows_per_process + 1)) {
          mpi_rank = (int) floor(row_number / (rows_per_process + 1));

       // For rows which belong to processors with rows_per_process rows
       } else {
          mpi_rank = (int) floor( (row_number - num_processes_w_leftover_rows * (rows_per_process + 1)) / rows_per_process
                         + num_processes_w_leftover_rows);
       }

       //Now, once these match, we copy the particle into this processor at the appropriate bin
       if (mpi_rank == rank) {
          particle_t part_cp = particles[i];
          bins[new_bin_ID].push_back(&parts[part_cp.id - 1]);
          parts[part_cp.id - 1].x = part_cp.x;
          parts[part_cp.id - 1].y = part_cp.y;
          parts[part_cp.id - 1].vx = part_cp.vx;
          parts[part_cp.id - 1].vy = part_cp.vy;
          parts[part_cp.id - 1].ax = part_cp.ax;
          parts[part_cp.id - 1].ay = part_cp.ay;

       }

    }

    //Delete array with the particles misplaced
    delete[] particles;


}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    std::vector<particle_t> local_particles;
    for (int binID: rank_bins_ID) {
        for (particle_t* p: bins[binID]) {
            local_particles.push_back(*p);
        }
    }
    int* gather_particles_size = new int[num_procs];
    int* gather_disp_size = new int[num_procs];

    int error_code = 0;

    int local_particles_size = local_particles.size();
    error_code = MPI_Gather(&local_particles_size, 1, MPI_INT, gather_particles_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != 0) {
        std::cout << "MPI_Gather error!!" << std::endl;
    }
    gather_disp_size[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        gather_disp_size[i] = gather_disp_size[i-1] + gather_particles_size[i-1];
    }
    
    //particle_t recv_buf[num_parts];
    particle_t* recv_buf = new particle_t[num_parts];
    error_code = MPI_Gatherv(&local_particles[0], local_particles.size(), PARTICLE, recv_buf,
            gather_particles_size, gather_disp_size, PARTICLE, 0, MPI_COMM_WORLD);

    if (error_code != 0) {
        std::cout << "MPI_Gatherv error!!" << std::endl;
    }

    if (rank == 0) {
        for (int i = 0; i < num_parts; i++) {;
            particle_t p = recv_buf[i];
            parts[p.id-1].x = p.x;
            parts[p.id-1].y = p.y;
            parts[p.id-1].ax = p.ax;
            parts[p.id-1].ay = p.ay;
            parts[p.id-1].vx = p.vx;
            parts[p.id-1].vy = p.vy;
        }
    }
    delete[] recv_buf;
    delete[] gather_particles_size;
    delete[] gather_disp_size;





}
