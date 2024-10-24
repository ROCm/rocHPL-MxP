/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hplmxp.hpp"

template <typename T>
int HPLMXP_bcast_1rinM(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*Root immediately sends to ROOT+1*/
  if(rank == ROOT) {
    MPI_Send(SBUF, SCOUNT, T2MPI<T>::type, MModAdd1(ROOT, size), ROOT, COMM);
  } else if(rank == MModAdd1(ROOT, size)) {
    MPI_Recv(SBUF, SCOUNT, T2MPI<T>::type, ROOT, ROOT, COMM, MPI_STATUS_IGNORE);
    return MPI_SUCCESS;
  }

  if(size == 2) return (MPI_SUCCESS);

  /*One ring exchange to rule them all*/
  int chunk_size = (1 << 21) / sizeof(T); // 2MB
  // int chunk_size = (1<<18)/sizeof(T); // 256KB

  chunk_size = std::min(chunk_size, SCOUNT);

  MPI_Request request[2];

  request[0] = MPI_REQUEST_NULL;
  request[1] = MPI_REQUEST_NULL;

  const int Nchunks = (SCOUNT + chunk_size - 1) / chunk_size;

  const int tag = rank;
  const int next =
      (rank == ROOT) ? MModAdd(ROOT, 2, size) : MModAdd1(rank, size);
  const int prev =
      (rank == MModAdd(ROOT, 2, size)) ? ROOT : MModSub1(rank, size);

  T* RBUF = SBUF;

  /*Shift to ROOT=0*/
  rank = MModSub(rank, ROOT, size);

  int Nsend = (rank == size - 1) ? 0 : SCOUNT;
  int Nrecv = (rank == 0) ? 0 : SCOUNT;

  /*Recv from left*/
  int Nr = std::min(Nrecv, chunk_size);
  if(Nr > 0) {
    MPI_Irecv(RBUF, Nr, T2MPI<T>::type, prev, prev, COMM, request + 0);
  }

  /*Send to right if there is data present to send*/
  int Ns = std::min(Nsend - Nrecv, chunk_size);
  if(Ns > 0) {
    MPI_Isend(SBUF, Ns, T2MPI<T>::type, next, tag, COMM, request + 1);
  }

  while(Nsend > 0 || Nrecv > 0) {
    int index = -1;
    MPI_Waitany(2, request, &index, MPI_STATUSES_IGNORE);

    if(index == 0) { /*Recv'd from left*/
      /*If we're waiting on this recv in order to send, send now*/
      if(Nrecv == Nsend) {
        Ns = Nr;
        MPI_Isend(SBUF, Ns, T2MPI<T>::type, next, tag, COMM, request + 1);
      }

      /*Count the recv'd amounts*/
      Nrecv -= Nr;
      RBUF += Nr;

      /*Post next recv if needed*/
      Nr = std::min(Nrecv, chunk_size);
      if(Nr > 0) {
        MPI_Irecv(RBUF, Nr, T2MPI<T>::type, prev, prev, COMM, request + 0);
      } else {
        request[0] = MPI_REQUEST_NULL;
      }

    } else if(index == 1) { /*Sent to right */
      Nsend -= Ns;
      SBUF += Ns;

      /*Send to right if there is data present to send*/
      Ns = std::min(Nsend - Nrecv, chunk_size);
      if(Ns > 0) {
        MPI_Isend(SBUF, Ns, T2MPI<T>::type, next, tag, COMM, request + 1);
      } else {
        request[1] = MPI_REQUEST_NULL;
      }
    }
  }

  return MPI_SUCCESS;
}

template int HPLMXP_bcast_1rinM(double*  SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);

template int HPLMXP_bcast_1rinM(float*   SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);

template int HPLMXP_bcast_1rinM(__half*  SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);

template int HPLMXP_bcast_1rinM(hipblaslt_f8_fnuz*  SBUF,
                                int      SCOUNT,
                                int      ROOT,
                                MPI_Comm COMM);
