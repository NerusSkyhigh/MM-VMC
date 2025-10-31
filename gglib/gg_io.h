// gg_io.h
// Created by gu on 31/10/25.
//

#pragma once

/**
 * @struct LammpsDat
 * @brief Descriptor for a LAMMPS-compatible data or trajectory file.
 *
 * This struct stores the information needed to write particle coordinates
 * to a LAMMPS-style data file. It keeps track of the filename, the number
 * of frames written, the number of particles, and the cubic box length.
 *
 * Usage:
 * 1. Initialize with `initLammpsData`.
 * 2. Write frames with `writeLammpsDatFrame`.
 * 3. Free resources with `freeLammpsData`.
 */
typedef struct LAMMPS_DAT_FILE {
    char *filename;
    size_t n_frames;

    size_t n_particles;
    double boxL;

} LammpsDat;

/**
 * @brief Initialize a LAMMPS data file descriptor.
 *
 * Allocates memory for the filename and sets up the number of particles, box length,
 * and frame counter. This should be called before writing any frames.
 *
 * @param ld Pointer to the LammpsDat struct to initialize.
 * @param filename Name of the LAMMPS data file to create/append to.
 * @param n_particles Number of particles that will be stored in the file.
 * @param boxL Length of the cubic box (assumes box from -boxL/2 to +boxL/2 in each dimension).
 */
void initLammpsData(LammpsDat *ld, const char *filename, size_t n_particles, double boxL);


/**
 * @brief Free resources associated with a LammpsDat struct.
 *
 * Frees the memory allocated for the filename and resets the fields
 * of the struct. Should be called when the LammpsDat struct is no longer needed.
 *
 * @param ld Pointer to the LammpsDat struct to clean up.
 */
void freeLammpsData(LammpsDat *ld);


/**
 * @brief Write a frame of particle coordinates to a LAMMPS-compatible file.
 *
 * The function appends a frame to the file. If it is the first frame, the file is created.
 * Coordinates are expected as a flat array [x0,y0,z0, x1,y1,z1, ...]. The output format
 * uses LAMMPS "dump" style with TIMESTEP, NUMBER OF ATOMS, BOX BOUNDS, and ATOMS sections.
 *
 * @param ld Pointer to the initialized LammpsDat struct.
 * @param coordinates Flat array of particle coordinates of size 3 * n_particles.
 */
void writeLammpsDatFrame(LammpsDat *ld, float *coordinates);

