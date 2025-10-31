/**
 * @file gg_io.c
 * @brief Functions to create and append frames to a LAMMPS-compatible data/trajectory file.
 *
 * Provides utilities to initialize a LAMMPS data file descriptor and write
 * particle coordinates frame by frame in a format suitable for visualization in VMD.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gg_io.h"

void initLammpsData(LammpsDat *ld, const char *filename, size_t n_particles, double boxL) {
    if (!ld || !filename) return;

    // Allocate memory for the filename string (+1 for null terminator)
    ld->filename = malloc(strlen(filename) + 1);
    if (!ld->filename) {
        fprintf(stderr, "Failed to allocate memory for filename: %s", filename);
        return;
    }
    strcpy(ld->filename, filename);
    ld->n_frames = 0;
    ld->n_particles = n_particles;
    ld->boxL = boxL;
}


void writeLammpsDatFrame(LammpsDat *ld, float *coordinates) {
    if (!ld || !coordinates) return;

    FILE *fp;
    if (ld->n_frames == 0) {
        // First frame: create file
        fp = fopen(ld->filename, "w");
        if (!fp) {
            perror("Failed to open file for writing");
            return;
        }
    } else {
        // Append mode for subsequent frames
        fp = fopen(ld->filename, "a");
        if (!fp) {
            perror("Failed to open file for appending");
            return;
        }
    }

    fprintf(fp, "ITEM: TIMESTEP\n");
    fprintf(fp,"%zu\n", ld->n_frames);

    fprintf(fp, "ITEM: NUMBER OF ATOMS\n");
    fprintf(fp, "%zu\n", ld->n_particles);

    fprintf(fp, "ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(fp, "%e %e\n", -0.5*ld->boxL, 0.5*ld->boxL);
    fprintf(fp, "%e %e\n", -0.5*ld->boxL, 0.5*ld->boxL);
    fprintf(fp, "%e %e\n", -0.5*ld->boxL, 0.5*ld->boxL);

    fprintf(fp, "ITEM: ATOMS id mol type xu yu zu\n");

    for (size_t i = 0; i < ld->n_particles; ++i) {
        size_t id = i+1;
        size_t molId = 1;
        size_t type = 1;

        double xu = coordinates[3*i+0];
        double yu = coordinates[3*i+1];
        double zu = coordinates[3*i+2];

        fprintf(fp, "%zu %zu %zu %f %f %f\n", id, molId, type, xu, yu, zu);
    }

    fclose(fp);
    ld->n_frames += 1;
}


void freeLammpsData(LammpsDat *ld) {
    if (!ld) return;

    if (ld->filename) {
        free(ld->filename);
        ld->filename = NULL;
    }

    ld->n_frames = 0;
    ld->n_particles = 0;
    ld->boxL = 0.0;
}
