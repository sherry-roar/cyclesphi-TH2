#include <mpi.h>
#include <iostream>

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double* a=nullptr;
  a = (double*)malloc(sizeof(double) * 10);

  if (world_rank == 0){
    //master node init the data 
    for (int i=0; i<10; ++i){
      a[i] = i+0.1;
    }
  }

  if (world_rank > 0){
    // before MPI_Bcast
    // world_rank > 0 print 0.0
    printf(" before rank:%d ", world_rank);
    for (int i=0; i<10; ++i){
      printf(" %0.2f ", a[i]);
    }
    printf("\n");
  }

  MPI_Bcast(a, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // After MPI_Bcast, all node got data
  printf("after rank:%d ", world_rank);
  for (int i=0; i<10; ++i){
    printf("%0.2f ", a[i]);
  }
  printf("\n");

  free(a);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}