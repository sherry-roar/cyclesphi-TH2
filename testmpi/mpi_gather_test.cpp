#include <mpi.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void initRow(double *a, int cols, double initNum){
  for (int i=0; i<cols; ++i){
    *(a+i) = initNum;
  }
}

int main(){

  MPI_Init(NULL, NULL);
  int world_rank=0;
  int world_size =0 ;
  const int rows = 10;
  const int cols = 5;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  {
        // int i=1;
        int i=0;
        while (0 == i)
            sleep(5);
  }
  const int numsPerNode = (rows * cols) / world_size;
  const int rowsPerNode = rows / world_size;

  double* a=nullptr;
  a = (double*)malloc(sizeof(double) * rows * cols);
  assert(a != nullptr);

  double* recv_a = nullptr;
  recv_a = (double*)malloc(sizeof(double) * numsPerNode);
  assert(recv_a != nullptr);

  double* res_perNode = nullptr;
  res_perNode = (double*)malloc(sizeof(double) * rowsPerNode);
  assert(res_perNode != nullptr);

  double* res = nullptr;
  res = (double*)malloc(sizeof(double) * rowsPerNode*world_size);
  assert(res != nullptr);

  if (world_rank == 0){
    // master node init matrix A
    for(int row = 0; row < rows; ++row){
      initRow(a+row*cols, cols, row+0.1);
    }
        for (int i=0; i<rows; ++i){
      printf("%0.2f ", res[i]);
    }
    printf("\n");
  }

  MPI_Scatter(a, numsPerNode, MPI_DOUBLE, recv_a, numsPerNode, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double temp = 0.0;
  for (int i=0; i<rowsPerNode; ++i){
    temp = 0;
    for (int j=0; j<cols; ++j){
      printf("%0.2f %d\n", recv_a[i*cols + j], world_rank);
      temp += recv_a[i*cols + j];
    }
    res_perNode[i] = temp;
  }

  MPI_Gather(res_perNode, rowsPerNode, MPI_DOUBLE, res, rowsPerNode, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank == 0){
    for (int i=0; i<rows; ++i){
      printf("%0.2f ", res[i]);
    }
    printf("\n");
  }



  free(a);
  free(recv_a);
  free(res);
  free(res_perNode);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}