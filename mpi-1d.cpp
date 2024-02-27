#include "common.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <vector>

// Apply the force from neighbor to particle
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

// Integrate the ODE
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

// Put any static global variables here that you will use throughout the simulation.
int num_bins;
int sqrt_procs;
int sides_per_proc;
int sides_per_proc_ghosts;
static std::vector<std::vector<particle_t>> bins;
static std::vector<std::vector<particle_t*>> serial_bins;

// Variables for sending and receiving particles
std::vector<particle_t> parts_sent;
particle_t* parts_recv;
// For MPI calls need to send counts of particles to send and receive
int* counts_sent;
int* counts_recv;
int* cum_counts_sent;
int* cum_counts_recv;

int get_proc(int proc_x, int proc_y) {
    // Gets rank based on x,y for process grid
    return proc_x * sqrt_procs + proc_y;
}

int get_proc_x(int rank) {
    // Gets x in process grid based on rank
    return (int)(rank / sqrt_procs);
}

int get_proc_y(int rank) {
    // Gets y in process grid based on rank
    return rank % sqrt_procs;
}

int get_proc_bin(int x, int y) {
    // Gets bin based on x,y for bin in process's bin grid
    return x * sides_per_proc_ghosts + y;
}

int* is_ghost(int global_bin_x, int global_bin_y, int rank) {
    // Checks if bin is a ghost bin
    int proc_x = get_proc_x(rank);
    int proc_y = get_proc_y(rank);

    // Checks that we're in the range of the process + ghost edges on (x,y)
    // E.g. for sides_per_proc=4 and (proc_x, proc_y)=(5, 6), we want to check that
    // x is in the range of [19,24] and y is in the range of [23, 28]
    if (global_bin_x < proc_x * sides_per_proc - 1 ||
        global_bin_x >= (proc_x + 1) * sides_per_proc + 1)
        return NULL;
    if (global_bin_y < proc_y * sides_per_proc - 1 ||
        global_bin_y >= (proc_y + 1) * sides_per_proc + 1)
        return NULL;

    // Get bin relative to top left corner of process's bins (0,0) not including ghost edges
    // For prev. example, this is global_bin (20,24)
    int proc_bin_x = global_bin_x - proc_x * sides_per_proc;
    int proc_bin_y = global_bin_y - proc_y * sides_per_proc;

    int* ret = new int[2];
    ret[0] = proc_bin_x;
    ret[1] = proc_bin_y;
    return ret;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    num_bins = (int)ceil(size / cutoff);

    if (num_parts < 1000) {
        serial_bins.resize(num_bins * num_bins);

        for (int i = 0; i < num_parts; i++) {
            int x = (int)(parts[i].x / cutoff);
            int y = (int)(parts[i].y / cutoff);
            serial_bins[x * num_bins + y].push_back(&parts[i]);
        }
        return;
    }

    // Ensure we are working with a square number
    sqrt_procs = (int)sqrt(num_procs);

    sides_per_proc = (int)ceil((double)num_bins / sqrt_procs);
    sides_per_proc_ghosts = sides_per_proc + 2;

    // Initialize stuff for sending/receiving
    counts_sent = new int[num_procs];
    counts_recv = new int[num_procs];
    cum_counts_sent = new int[num_procs];
    cum_counts_recv = new int[num_procs];
    parts_recv = parts;

    // Ghost bins are added to the edges but empty for now
    bins.resize(sides_per_proc_ghosts * sides_per_proc_ghosts);

    // Add all particles to bins in appropriate process
    for (int i = 0; i < num_parts; i++) {
        int global_bin_x = (int)(parts[i].x / cutoff);
        int global_bin_y = (int)(parts[i].y / cutoff);

        int proc_x = (int)(global_bin_x / sides_per_proc);
        int proc_y = (int)(global_bin_y / sides_per_proc);

        if (get_proc(proc_x, proc_y) == rank) {
            // Offset by 1 bc we have ghost bins on the edges
            int proc_bin_x = global_bin_x % (sides_per_proc) + 1;
            int proc_bin_y = global_bin_y % (sides_per_proc) + 1;
            bins[get_proc_bin(proc_bin_x, proc_bin_y)].push_back(parts[i]);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function

    /* Reorganize and move. */
    // Get particles to send and move particles between bins
    // Also clear ghost bins

    /* Cleanup */

    if (num_parts < 1000) {
        int num_bins = (int)ceil(size / cutoff);

        for (int x = 0; x < num_bins; x++) {
            for (int y = 0; y < num_bins; y++) {
                int ind = x * num_bins + y;

                for (auto it = serial_bins[ind].begin(); it != serial_bins[ind].end();) {
                    particle_t* particle = *it;
                    int new_x = (int)(particle->x / cutoff);
                    int new_y = (int)(particle->y / cutoff);

                    if (new_x != x || new_y != y) {
                        int new_ind = new_x * num_bins + new_y;
                        serial_bins[new_ind].push_back(particle);
                        it = serial_bins[ind].erase(it);
                    } else {
                        it++;
                    }
                }
            }
        }

        for (int i = 0; i < num_bins; i++) {
            for (int j = 0; j < num_bins; j++) {
                int ind = i * num_bins + j;

                for (particle_t* particle : serial_bins[ind]) {
                    particle->ax = particle->ay = 0;

                    for (int x = i - 1; x < i + 2; x++) {
                        for (int y = j - 1; y < j + 2; y++) {
                            if (x < num_bins && y < num_bins && x >= 0 && y >= 0) {
                                int new_ind = x * num_bins + y;
                                for (particle_t* part : serial_bins[new_ind]) {
                                    if (particle != part) {
                                        apply_force(*particle, *part);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Move Particles
        for (int i = 0; i < num_parts; ++i) {
            move(parts[i], size);
        }
        return;
    }

    parts_sent.clear();
    memset(counts_sent, 0, num_procs * sizeof(int));
    memset(counts_recv, 0, num_procs * sizeof(int));
    memset(cum_counts_sent, 0, num_procs * sizeof(int));
    memset(cum_counts_recv, 0, num_procs * sizeof(int));

    if (rank < sqrt_procs * sqrt_procs) {
        for (int x = 0; x < sides_per_proc_ghosts; x++) {
            for (int y = 0; y < sides_per_proc_ghosts; y++) {
                int ind = get_proc_bin(x, y);

                // If ghost bin, clear it
                if (x == 0 || y == 0 || x == sides_per_proc_ghosts - 1 ||
                    y == sides_per_proc_ghosts - 1) {
                    bins[ind].clear();
                    continue;
                }

                // If not ghost bin, find particles to send and rebin
                for (auto it = bins[ind].begin(); it != bins[ind].end(); it++) {
                    particle_t& particle = *it;
                    int global_bin_x = (int)(particle.x / cutoff);
                    int global_bin_y = (int)(particle.y / cutoff);

                    int proc_x = (int)(global_bin_x / sides_per_proc);
                    int proc_y = (int)(global_bin_y / sides_per_proc);

                    int new_rank = get_proc(proc_x, proc_y);

                    // If clause -> this particle is still in this process
                    // If the particle is on the edge of the process (ghost), we treat it as not
                    // part of the process (else case)
                    if (new_rank == rank) {
                        int proc_bin_x = global_bin_x % (sides_per_proc) + 1;
                        int proc_bin_y = global_bin_y % (sides_per_proc) + 1;
                        int new_ind = get_proc_bin(proc_bin_x, proc_bin_y);

                        if (proc_bin_x == 1 || proc_bin_y == 1 || proc_bin_x == sides_per_proc ||
                            proc_bin_y == sides_per_proc) {
                            parts_sent.push_back(particle);
                        }

                        if (new_ind != ind) {
                            bins[new_ind].push_back(particle);
                            it = bins[ind].erase(it);
                            it--;
                        }
                    } else {
                        parts_sent.push_back(particle);
                        it = bins[ind].erase(it);
                        it--;
                    }
                }
            }
        }
    }

    // Make all processes aware of how many particles they will be receiving
    int num_parts_sent = parts_sent.size();
    MPI_Allgather(&num_parts_sent, 1, MPI_INT, counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

    // Create cum_counts_recv
    for (int i = 0; i < num_procs; i++) {
        cum_counts_recv[i] = counts_recv[i - 1] + (i == 0 ? 0 : cum_counts_recv[i - 1]);
    }

    MPI_Allgatherv(&parts_sent[0], parts_sent.size(), PARTICLE, parts_recv, counts_recv,
                   cum_counts_recv, PARTICLE, MPI_COMM_WORLD);

    if (rank < sqrt_procs * sqrt_procs) {
        int total_parts_recv = cum_counts_recv[num_procs - 1] + counts_recv[num_procs - 1];
        for (int i = 0; i < total_parts_recv; i++) {

            particle_t& particle = parts_recv[i];
            int global_bin_x = (int)(particle.x / cutoff);
            int global_bin_y = (int)(particle.y / cutoff);

            int proc_x = (int)(global_bin_x / sides_per_proc);
            int proc_y = (int)(global_bin_y / sides_per_proc);

            int new_rank = get_proc(proc_x, proc_y);

            // Check if this particle belongs to this process (not ghost, moved to this process)
            if (new_rank == rank) {
                int proc_bin_x = global_bin_x % (sides_per_proc) + 1;
                int proc_bin_y = global_bin_y % (sides_per_proc) + 1;
                int new_ind = get_proc_bin(proc_bin_x, proc_bin_y);

                bool is_new = true;
                for (particle_t& p : bins[new_ind]) {
                    if (p.id == particle.id) {
                        is_new = false;
                        break;
                    }
                }
                if (is_new) {
                    bins[new_ind].push_back(particle);
                }
            }

            // Check if ghost
            int* ret = is_ghost(global_bin_x, global_bin_y, rank);
            if (ret != NULL) {
                int proc_bin_x = ret[0] + 1;
                int proc_bin_y = ret[1] + 1;
                delete[] ret;
                int new_ind = get_proc_bin(proc_bin_x, proc_bin_y);
                bool is_new = true;
                for (particle_t& p : bins[new_ind]) {
                    if (p.id == particle.id) {
                        is_new = false;
                        break;
                    }
                }
                if (is_new) {
                    bins[new_ind].push_back(particle);
                }
            }
        }
    }

    // Compute forces
    // Go thru all bins but ghost bins on edges to compute on
    if (rank < sqrt_procs * sqrt_procs) {
        for (int i = 1; i < sides_per_proc_ghosts - 1; i++) {
            for (int j = 1; j < sides_per_proc_ghosts - 1; j++) {
                int ind = get_proc_bin(i, j);

                for (particle_t& particle : bins[ind]) {
                    particle.ax = particle.ay = 0;

                    for (int x = i - 1; x <= i + 1; x++) {
                        for (int y = j - 1; y <= j + 1; y++) {
                            if (x < num_bins && y < num_bins && x >= 0 && y >= 0) {
                                int apply_ind = x * sides_per_proc_ghosts + y;
                                for (particle_t& part : bins[apply_ind]) {
                                    if (particle.id != part.id) {
                                        apply_force(particle, part);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Move Particles
        for (int x = 1; x < sides_per_proc_ghosts - 1; x++) {
            for (int y = 1; y < sides_per_proc_ghosts - 1; y++) {
                int ind = get_proc_bin(x, y);

                for (auto it = bins[ind].begin(); it != bins[ind].end(); it++) {
                    particle_t& particle = *it;
                    move(particle, size);
                }
            }
        }
    }
}

void gather_for_save(particle_t* parts, const int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    if (num_parts < 1000) {
        return;
    }

    // Cleanup
    parts_sent.clear();
    memset(counts_recv, 0, num_procs * sizeof(int));
    memset(cum_counts_recv, 0, num_procs * sizeof(int));
    memset(counts_sent, 0, num_procs * sizeof(int));
    memset(cum_counts_sent, 0, num_procs * sizeof(int));

    parts_recv = new particle_t[num_parts];

    for (int x = 1; x < sides_per_proc_ghosts - 1; x++) {
        for (int y = 1; y < sides_per_proc_ghosts - 1; y++) {
            int ind = get_proc_bin(x, y);

            for (auto it = bins[ind].begin(); it != bins[ind].end(); it++) {
                particle_t& particle = *it;
                parts_sent.push_back(particle);
            }
        }
    }

    int num_parts_sent = parts_sent.size();
    MPI_Gather(&num_parts_sent, 1, MPI_INT, counts_recv, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            cum_counts_recv[i] = counts_recv[i - 1] + (i == 0 ? 0 : cum_counts_recv[i - 1]);
        }
    }

    MPI_Gatherv(&parts_sent[0], parts_sent.size(), PARTICLE, parts_recv, counts_recv,
                cum_counts_recv, PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_parts_recv = cum_counts_recv[num_procs - 1] + counts_recv[num_procs - 1];
        for (int i = 0; i < total_parts_recv; i++) {
            particle_t* particle = &parts_recv[i];
            parts[particle->id - 1] = *particle;
        }
    }
}
