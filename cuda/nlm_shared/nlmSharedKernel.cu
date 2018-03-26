#include <math.h>
#include <stdio.h>

// Array access macros
#define INPUT(i,j) input_grid[(j) + (i)*(N)]
#define TEMP(i,j) temp_grid[(j) + (i)*(N)]

#define WINDOW_SIZE (7)
#define NEIGHBOR_SIZE (3)
#define BLOCK_SIZE (512)

#define FILTER_SIZE ((WINDOW_SIZE) + (NEIGHBOR_SIZE) - 1)
#define FILTER_RADIUS (((FILTER_SIZE) - 1) / 2)

__global__ void nlmSimple(int N, double const *input_grid, double *output_grid, float filtSigma)
{
  __shared__ double temp_grid[BLOCK_SIZE * FILTER_SIZE];

  // Define global and local indices of current pixel
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + FILTER_RADIUS * blockDim.x;

  int pix_ix, pix_iy, pix_jx, pix_jy;

  double neighbor_j,
         neighbor_i,
         output = 0,
         sum_weights = 0;

  // Read input elements into shared memory
  for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++)
  {
    if ((int)blockIdx.x + i >= 0 && (int)blockIdx.x + i < N)
    {
      temp_grid[lindex + i * (int)blockDim.x] = input_grid[gindex + i * (int)blockDim.x];
    }
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  pix_iy = lindex % N;
  pix_ix = (lindex - pix_iy) / N;

  if (pix_ix < FILTER_SIZE && pix_iy < N)
  { 
    int window_radius = (WINDOW_SIZE - 1) / 2;
    int neighbor_radius = (NEIGHBOR_SIZE - 1) / 2; 

    // Iterate through window
    for (int k = -window_radius; k <= window_radius; k++)
      for (int l = -window_radius; l <= window_radius; l++)
      {
        double weight = 0;
        double distance = 0;

        pix_jx = pix_ix + k; 
        pix_jy = pix_iy + l;

        if (pix_jx < 0 || pix_jx >= FILTER_SIZE ||
            pix_jy < 0 || pix_jy >= N)
          continue;

        // Iterate through every pix_j neighbors
        for (int p = -neighbor_radius; p <= neighbor_radius; p++)
          for (int q = -neighbor_radius; q <= neighbor_radius; q++)
          {
            if (pix_jx + p < 0 || pix_jx + p >= FILTER_SIZE ||
                pix_jy + q < 0 || pix_jy + q >= N ||
                pix_ix + p < 0 || pix_ix + p >= FILTER_SIZE ||
                pix_iy + q < 0 || pix_iy + q >= N)
              continue;

            neighbor_j = TEMP(pix_jx + p, pix_jy + q);
            neighbor_i = TEMP(pix_ix + p, pix_iy + q);
            distance += (neighbor_i - neighbor_j) * (neighbor_i - neighbor_j);
          }

        // Derive weight for pixels i and j
        weight = __expf(-(distance / filtSigma + 
              (k*k + l*l) * (1.0f)/(float)(WINDOW_SIZE* WINDOW_SIZE)));

        sum_weights += weight;

        // Sum for every pixel in the window
        output += TEMP(pix_jx, pix_jy) * weight;			
      }

    // Normalize
    sum_weights = (double)(1 / sum_weights);
    output *= sum_weights;

    // Write output to global memory
    output_grid[gindex] = output;
  }
}
