# Modified CyclesPhi to run on TH-2 with MPI

#### MODIFIED VERSION OF CYCLES PRODUCTION RENDERER FOR EFFICIENT RENDERING ON HPC CLUSTER

We have modified the kernel of the Blender Cycles rendering engine and then extended its capabilities to support the HPC environment.
We call this version CyclesPhi.
This application has been developed at [IT4Innovations National Supercomputing Center](https://www.it4i.cz/).

---
# CyclesPhi v3 Release Information
---

In this release we have extended CyclesX (includes Optix Denoiser). It supports multi-GPU rendering on HPC cluster and it contains the method to analyze the memory access pattern of the path tracer and to effectively distribude the scene data so massive scenes can be rendered on multiple GPUs with only minimal effect on the performance.

---
# Content
---

1. [Supported technologies](https://code.it4i.cz/raas/cyclesphi#supported-technologies)
2. [CyclesPhi working principle](https://code.it4i.cz/raas/cyclesphi#cyclesphi-working-principle)
3. [Compatibility](https://code.it4i.cz/raas/cyclesphi#compatibility)
4. [Building process](https://code.it4i.cz/raas/cyclesphi#building-process)
5. [Running CyclesPhi example](https://code.it4i.cz/raas/cyclesphi#running-cyclesphi-example)
6. [License](https://code.it4i.cz/raas/cyclesphi#license)
7. [Acknowledgement](https://code.it4i.cz/raas/cyclesphi#acknowledgement)
8. [References](https://code.it4i.cz/raas/cyclesphi#references)

---
# Supported technologies
---

To make CyclesPhi effectively utilize HPC resources it has been modified to support following technologies:

* OpenMP
* MPI
* NVIDIA GPU (CUDA, Optix)
* CUDA Unified Memory mechanism
* And their combinations

---
# CyclesPhi working principle
---

CyclesPhi application works closely with Blender, since CyclesPhi is modified version of Blender's renderer.
CyclesPhi is capable of rendering Blender scenes while distributing the rendering task on multiple cluster nodes using MPI.
Communication between Blender and CyclesPhi is via file transfer.
Overview of working principle is depicted in pictures below.

![](https://code.it4i.cz/raas/cyclesphi/raw/master/doc/cyclesphi/CyclesPhi_principle.png)

---
# Compatibility
---

Although CyclesPhi is compatible only with modified version of Blender 3.3, scenes from standard Blender 3.3 are of course supported.
Modification of Blender is necessary for sufficien support of MPI technology on an HPC cluster.
The build of modified Blender version is part of CyclesPhi building procedure provided below.  

---
# Building process
---

### Building CyclesPhi and modified Blender version on HPC Cluster

Supplied shell script *download_blender.sh* downloads the source codes from git repository and builds CyclesPhi and Blender into selected directory on HPC cluster.

First, the CyclesPhi is built then Blender is built.

Second, the modified version of Blender is built.

The whole process is described below.

```
ssh karolina.it4i.cz
it4i-get-project-dir OPEN-XX-XX
mkdir -p /mnt/projX/open-XX-XX/project-cyclesphi
cd /mnt/projX/open-XX-XX/project-cyclesphi

wget https://code.it4i.cz/raas/cyclesphi/raw/master/client/scripts/download_lib.sh
chmod +x download_lib.sh
./download_lib.sh

wget https://code.it4i.cz/raas/cyclesphi/raw/master/client/scripts/download_blender.sh
chmod +x download_blender.sh
./download_blender.sh

wget https://code.it4i.cz/raas/cyclesphi/raw/master/client/scripts/download_benchmarks.sh
chmod +x download_benchmarks.sh
./download_benchmarks.sh

qsub -q qexp -I
cd /mnt/projX/open-XX-XX/project-cyclesphi
./scripts/build_blender33.sh
./scripts/build_cyclesphi33_cpu.sh
./scripts/build_cyclesphi33_gpu.sh
./scripts/build_cyclesphi33_gpu_unimem_rr.sh
./scripts/build_cyclesphi33_gpu_unimem_credit.sh

exit
```
./scripts/build_cyclesphi33_gpu.sh need to change
```
CUDA_ROOT=/usr/local/cuda-12.2

# export CC='gcc'
# export CXX='g++'
export CC='mpicc'
export CXX='mpicxx'


# make_d="${make_d} -DMPI_CXX_HEADER_DIR=/apps/all/impi/2021.7.1-intel-compilers-2022.2.1/mpi/2021.7.1/include"
# make_d="${make_d} -DMPI_LIBRARIES=/apps/all/impi/2021.7.1-intel-compilers-2022.2.1/mpi/2021.7.1/lib/release/libmpi.so"
make_d="${make_d} -DMPI_CXX_HEADER_DIR=/usr/lib/x86_64-linux-gnu/openmpi/include"
make_d="${make_d} -DMPI_LIBRARIES=/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so"
```
## be ware that samples needs to be change in `src/cyclesphi/client/cycles/kernel_omp.cpp:line 1065` and `intern/cycles/integrator/path_trace_work_cpu.cpp:line 119`, same in gpu and other samplers

in `kernel_omp.cpp`:
```
add in line 1059: g_num_samples = end_sample - start_sample;
wtile.num_samples = g_num_samples;
```
in `path_trace_work_cpu.cpp`:
```
// work_tile.num_samples = 1;
    work_tile.num_samples = samples_num;
```
in `src/cyclesphi/client/cycles/cycles_client.cpp:line 4238`, we can add below codes to set samples into evn samples:

```
# if 1
// set num samples into env samples
    const char *env_samples = getenv("DEBUG_SAMPLES");
    int num_samples = data.client_path_trace_data.num_samples;
    if (env_samples != NULL) {
      num_samples = atoi(env_samples);
    }
    data.client_path_trace_data.num_samples = num_samples;
# endif
```
`defined WITH_CLIENT_MPI_FILE_1` is mpi parallel method. it copy all data to all core and render, then composit when each tile have been done.

1. `build_xx.sh` add `make_d="${make_d} -DWITH_CLIENT_MPI_FILE_1=ON"`
2. `CMakeLists.txt` in `./cyclesphi/client/cycles:line588` and `./cyclesphi/client/blender_client:line67` add 
```
if(WITH_CLIENT_MPI_FILE_1)
	add_definitions(-DWITH_CLIENT_MPI_FILE_1)
endif()
```
3. in `blender_client.cpp:line236`add `|| defined(WITH_CLIENT_MPI_FILE_1)`
4. in `cycles_client.cpp:line146`add
```
#if defined(WITH_CLIENT_MPI_FILE_1)
  void bcast(void *data, size_t size)
  {
  #if defined(WITH_CLIENT_FILE)
    if (g_world_rank == 0)
    {
      NET_NAMESPACE::read_data_kernelglobal((char *)data, size);
      const size_t unit_giga = 1024L * 1024L * 128L;
      size_t size_sended = 0;
      while (size - size_sended > unit_giga) {
        MPI_Bcast((char *)data + size_sended, unit_giga, MPI_BYTE, 0, MPI_COMM_WORLD);
        size_sended += unit_giga;
      }
      printf("[%d] bcast %zu bytes\n", g_world_rank, size);
      MPI_Bcast((char *)data + size_sended, size - size_sended, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else
    {
      printf("[%d] bcast %zu bytes\n", g_world_rank, size);
      MPI_Bcast((char *)data, size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  #endif
  }
#else
...
#endif
```
5. in `line:4226` add
```
#     if !defined(WITH_CLIENT_MPI_FILE_1)
  DEV_NAMESPACE::mem_copy_from(
      CLIENT_DEVICE_ID,
      DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
      &buffer[0],
      offsetBuf,
      buffer.size(),
      (char *)&pass_stride_float);
#     else
  DEV_NAMESPACE::mem_copy_from(
      CLIENT_DEVICE_ID,
      g_path_trace_buffer_received.data(),
      &buffer[0],
      offsetBuf,
      buffer.size(),
      (char *)&pass_stride_float);
#     endif
```
6. in `line:2577` change to follow:
```
#ifdef WITH_CLIENT_MPI_FILE_1
    std::vector<char> g_path_trace_buffer_received;
    // std::vector<char> g_path_trace_pixels_send;
    // std::vector<char> g_path_trace_pixels_received;
    DEVICE_PTR g_path_trace_buffer = NULL;
    // DEVICE_PTR g_path_trace_pixels = NULL;

    std::vector<int> g_displsPix;
    std::vector<int> g_recvcountsPix;

    void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                      char *buffer_temp,
                                      size_t buffer_size)
    {

      int dev_node = data.world_rank;           //         -1;
      int devices_size_node = data.world_size;  // -1;

      // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
      // ceil
      int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                    (float)devices_size_node);

      int tile_last_node = data.client_path_trace_data.tile_h -
                          (devices_size_node - 1) * tile_step_node;

      if (tile_last_node < 1) {
        tile_step_node = (int)((float)data.client_path_trace_data.tile_h / (float)(devices_size_node));

        tile_last_node = data.client_path_trace_data.tile_h - (devices_size_node - 1) * tile_step_node;
      }

      int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
      int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

      size_t buff_size_node = tile_h_node * data.client_path_trace_data.tile_w *
                              data.client_path_trace_data.pass_stride * sizeof(float);

    #  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
          (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      buff_size_node *= 2;  // left+right

      size_t buff_offset = (data.client_path_trace_data.tile_x +
                            tile_y_node * data.client_path_trace_data.stride * 2) *
                          data.client_path_trace_data.pass_stride * sizeof(float);
    #  else
      size_t buff_offset = (data.client_path_trace_data.tile_x +
                            tile_y_node * data.client_path_trace_data.stride) *
                          data.client_path_trace_data.pass_stride * sizeof(float);
    #  endif

      if (g_displsPix.size() != devices_size_node) {
        g_displsPix.resize(devices_size_node, 0);
        g_recvcountsPix.resize(devices_size_node, 0);
      }

      for (int dev = 0; dev < devices_size_node; dev++) {
        int tile_y2 = data.client_path_trace_data.tile_y + tile_step_node * dev;
        int tile_h2 = (devices_size_node - 1 == dev) ? tile_last_node : tile_step_node;
        if (tile_h2 == 0)
          continue;

        g_displsPix[dev] = (data.client_path_trace_data.offset + data.client_path_trace_data.tile_x +
                            tile_y2 * data.client_path_trace_data.stride) *
                          data.client_path_trace_data.pass_stride * sizeof(float);
        g_recvcountsPix[dev] = data.client_path_trace_data.tile_w * tile_h2 *
                              data.client_path_trace_data.pass_stride * sizeof(float);
      }

      if (g_path_trace_buffer_received.size() != buffer_size)
        g_path_trace_buffer_received.resize(buffer_size);
      
      
      MPI_Gatherv((char *)buffer_temp + buff_offset,
                  buff_size_node,
                  MPI_BYTE,
                  g_path_trace_buffer_received.data(),
                  &g_recvcountsPix[0],
                  &g_displsPix[0],
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);

      
      if (data.world_rank == 0)
          DEV_NAMESPACE::mem_copy_to(
          CLIENT_DEVICE_ID, 
          g_path_trace_buffer_received.data(), 
          // DEV_NAMESPACE::get_ptr_map(g_path_trace_buffer),
          DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
          buffer_size, 
          NULL);
        // cyclesphi::client::save_buffer(data);
      //   NET_NAMESPACE::send_data_data(g_path_trace_buffer_received.data(), buffer_size);
    }
    
  #else

    void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                      char *buffer_temp,
                                      size_t buffer_size)
    {
    }

  #endif
```
---
# Running CyclesPhi example
---

After building CyclesPhi and Blender it is possible to run the rendering task either on CPU or GPU nodes.
Description for both types is provided below.
To supply your own Blender scene for rendering you have to modify the script *run_blender_pre.sh* on line 16 and modify the text *$ROOT/src/lib/benchmarks/cycles/classroom/classroom.blend*.

## Running on GPUs

### Fully duplicated scene

```
pre=$(qsub -q qexp -l select=1 ./scripts/run_blender_pre.sh) 
rend=$(qsub -A OPEN-0-0 -W depend=afterok:$pre -q qnvidia -l select=2:mpiprocs=1:ompthreads=128:ncpus=128 ./scripts/run_cyclesphi33_gpu.sh)
post=$(qsub -W depend=afterok:$rend -q qexp -l select=1 ./scripts/run_blender_post.sh)
echo $pre '->' $rend '->' $post
```

### Fully distributed scene using 2MB chunks and round robin method

```
pre=$(qsub -q qexp -l select=1 ./scripts/run_blender_pre.sh) 
rend=$(qsub -A OPEN-0-0 -W depend=afterok:$pre -q qnvidia -l select=2:mpiprocs=1:ompthreads=128:ncpus=128 ./scripts/run_cyclesphi33_gpu_unimem_rr.sh)
post=$(qsub -W depend=afterok:$rend -q qexp -l select=1 ./scripts/run_blender_post.sh)
echo $pre '->' $rend '->' $post
```

### 1% duplicated and 99% distributed scene using 2MB chunks and credit method

```
pre=$(qsub -q qexp -l select=1 ./scripts/run_blender_pre.sh) 
rend=$(qsub -A OPEN-0-0 -W depend=afterok:$pre -q qnvidia -l select=2:mpiprocs=1:ompthreads=128:ncpus=128 ./scripts/run_cyclesphi33_gpu_unimem_credit.sh)
post=$(qsub -W depend=afterok:$rend -q qexp -l select=1 ./scripts/run_blender_post.sh)
echo $pre '->' $rend '->' $post
```

## Running on CPUs

```
pre=$(qsub -q qexp -l select=1 ./scripts/run_blender_pre.sh) 
rend=$(qsub -A OPEN-0-0 -W depend=afterok:$pre -q qnvidia -l select=2:mpiprocs=1:ompthreads=128:ncpus=128 ./scripts/run_cyclesphi33_cpu.sh)
post=$(qsub -W depend=afterok:$rend -q qexp -l select=1 ./scripts/run_blender_post.sh)
echo $pre '->' $rend '->' $post
```

### Result

After running the example, you can find the rendering results in folder data/*. 
Example contains a classroom demo - The popular classroom scene by Christophe SEUX (CC0, 
christopheseux@free.fr), see below.

![](https://code.it4i.cz/raas/cyclesphi/raw/master/client/data/classroom.png)

#### Strong scalability test

The strong scalability test was run on 1 to 8 nodes (8 to 64 gpus).

![](https://code.it4i.cz/raas/cyclesphi/raw/master/doc/cyclesphi/CyclesPhi_classroom_scalability.png)

### Strong scalability test using CPUs

The strong scalability test was run on 1 to 8 nodes.

![](https://code.it4i.cz/raas/cyclesphi/raw/master/doc/cyclesphi/CyclesPhi_classroom_scalability_CPU.png)

---
# License
---
This software is licensed under the terms of the [GNU General Public License](https://code.it4i.cz/raas/cyclesphi/blob/master/COPYING).


---
# Acknowledgement
---
This work was supported by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140).


---
# References
---

Milan Jaroš, Lubomír Říha, Petr Strakoš, and Matěj Špeťko. 2021. GPU Accelerated Path Tracing of Massive Scenes. ACM Trans. Graph. 40, 2, Article 16 (June 2021), 17 pages. [DOI](https://doi.org/10.1145/3447807)
