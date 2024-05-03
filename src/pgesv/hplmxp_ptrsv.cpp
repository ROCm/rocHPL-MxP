
#include "hplmxp.hpp"

void HPLMXP_ptrsvL(HPLMXP_T_grid&         grid,
                   HPLMXP_T_pmat<fp32_t>& A,
                   fp64_t*                x,
                   fp64_t*                work) {
  // x is a column vector
  // only the diagonal part need to be valid
  // all the values will be modified after the computation
  // w1 and w2 are working space which has same size with x.
  // w3 has length b
  fp32_t*   Ap      = A.A;
  int const b       = A.nb;
  int const nblocks = A.n / b;
  int const nbrow   = A.nbrow;
  int const nbcol   = A.nbcol;
  int const lda     = A.ld;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  bool const single_col = grid.npcol == 1;
  int const  left       = single_col
                       ? MPI_PROC_NULL
                       : (grid.mycol == 0 ? grid.npcol - 1 : grid.mycol - 1);
  int const right = single_col
                        ? MPI_PROC_NULL
                        : (grid.mycol == grid.npcol - 1 ? 0 : grid.mycol + 1);
  int const top    = (grid.myrow == 0 ? grid.nprow - 1 : grid.myrow - 1);
  int const bottom = (grid.myrow == grid.nprow - 1 ? 0 : grid.myrow + 1);

  fp64_t* W  = work;
  fp64_t* w1 = W + b * b;
  fp64_t* w2 = w1 + b * nbrow;
  fp64_t* w3 = w2 + b * nbrow;

  /* set value */
  HPLMXP_set(b * nbrow, 0.0, w1);
  HIP_CHECK(hipDeviceSynchronize());

  for(int pj = 0; pj < nbcol; ++pj) {
    MPI_Request req_recv_pivv = MPI_REQUEST_NULL, req_recv_v = MPI_REQUEST_NULL;
    int         gj     = mycol + pj * npcol;
    int         cleft  = gj == 0 ? MPI_PROC_NULL : left;
    int         pivot  = gj % nprow;
    int         istart = (gj < myrow ? 0 : (gj - myrow + nprow - 1) / nprow);
    int         ii     = myrow + istart * nprow;

    if(ii >= nblocks) break;

    bool impivot   = pivot == myrow;
    bool no_bottom = grid.nprow == 1 || ii + 1 >= nblocks;

    if(impivot) {
      MPI_Irecv(w1 + istart * b,
                b,
                T2MPI<fp64_t>::type,
                cleft,
                200,
                grid.row_comm,
                &req_recv_pivv);

      if(istart + 1 < nbrow) {
        MPI_Irecv(w1 + b * (istart + 1),
                  b * (nbrow - istart - 1),
                  T2MPI<fp64_t>::type,
                  cleft,
                  200,
                  grid.row_comm,
                  &req_recv_v);
      }

      HPLMXP_lacpyL(b, b, Mptr(Ap, istart * b, pj * b, lda), lda, W, b);

      MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);

      // compute the pivot first
      HPLMXP_axpy(b, -1.0, w1 + b * istart, x + b * istart);

      HPLMXP_trsvL(b, W, b, x + b * istart);

      HIP_CHECK(hipEventRecord(piv, computeStream));

      if(istart + 1 < nbrow) {
        HPLMXP_gemv(b,
                    b,
                    1.0,
                    Mptr(Ap, (istart + 1) * b, pj * b, lda),
                    lda,
                    x + b * istart,
                    0.0,
                    w2 + b * (istart + 1));
      }

      if(!no_bottom) {
        /* sync */
        HIP_CHECK(hipEventSynchronize(piv));
        MPI_Send(
            x + b * istart, b, T2MPI<fp64_t>::type, bottom, 100, grid.col_comm);
      }

      if(istart + 1 < nbrow) {
        MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);

        HPLMXP_axpy(b, 1.0, w2 + b * (istart + 1), w1 + b * (istart + 1));

        /* sync */
        HIP_CHECK(hipStreamSynchronize(computeStream));

        // compute others last
        if(istart + 2 < nbrow) {
          HPLMXP_gemv(b * (nbrow - istart - 2),
                      b,
                      1.0,
                      Mptr(Ap, (istart + 2) * b, pj * b, lda),
                      lda,
                      x + b * istart,
                      0.0,
                      w2 + b * (istart + 2));

          // Fuse this with above^?
          HPLMXP_axpy(b * (nbrow - istart - 2),
                      1.0,
                      w2 + b * (istart + 2),
                      w1 + b * (istart + 2));
        }

        MPI_Send(w1 + b * (istart + 1),
                 b,
                 T2MPI<fp64_t>::type,
                 right,
                 200,
                 grid.row_comm);

        if(istart + 2 < nbrow) {
          /* sync */
          HIP_CHECK(hipStreamSynchronize(computeStream));

          MPI_Send(w1 + b * (istart + 2),
                   b * (nbrow - istart - 2),
                   T2MPI<fp64_t>::type,
                   right,
                   200,
                   grid.row_comm);
        }
      }

    } else {

      bool bottom_is_pivot = pivot == bottom;

      MPI_Irecv(w1 + istart * b,
                b,
                T2MPI<fp64_t>::type,
                cleft,
                200,
                grid.row_comm,
                &req_recv_pivv);

      if(istart + 1 < nbrow) {
        MPI_Irecv(w1 + (istart + 1) * b,
                  b * (nbrow - istart - 1),
                  T2MPI<fp64_t>::type,
                  cleft,
                  200,
                  grid.row_comm,
                  &req_recv_v);
      }

      MPI_Recv(w3,
               b,
               T2MPI<fp64_t>::type,
               top,
               100,
               grid.col_comm,
               MPI_STATUS_IGNORE);

      // compute the critical-path first
      HPLMXP_gemv(b,
                  b,
                  1.0,
                  Mptr(Ap, istart * b, pj * b, lda),
                  lda,
                  w3,
                  0.0,
                  w2 + b * istart);

      if(!bottom_is_pivot && !no_bottom) {
        MPI_Send(w3, b, T2MPI<fp64_t>::type, bottom, 100, grid.col_comm);
      }

      MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);

      HPLMXP_axpy(b, 1.0, w2 + b * istart, w1 + b * istart);

      /* sync */
      HIP_CHECK(hipStreamSynchronize(computeStream));

      // compute others
      if(istart + 1 < nbrow) {
        HPLMXP_gemv(b * (nbrow - istart - 1),
                    b,
                    1.0,
                    Mptr(Ap, (istart + 1) * b, pj * b, lda),
                    lda,
                    w3,
                    0.0,
                    w2 + b * (istart + 1));
      }

      MPI_Send(
          w1 + b * istart, b, T2MPI<fp64_t>::type, right, 200, grid.row_comm);

      if(istart + 1 < nbrow) {
        MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);

        HPLMXP_axpy(b * (nbrow - istart - 1),
                    1.0,
                    w2 + b * (istart + 1),
                    w1 + b * (istart + 1));

        /* sync */
        HIP_CHECK(hipStreamSynchronize(computeStream));

        MPI_Send(w1 + b * (istart + 1),
                 b * (nbrow - istart - 1),
                 T2MPI<fp64_t>::type,
                 right,
                 200,
                 grid.row_comm);
      }
    }
  }
}

void HPLMXP_ptrsvU(HPLMXP_T_grid&         grid,
                   HPLMXP_T_pmat<fp32_t>& A,
                   fp64_t*                x,
                   fp64_t*                work) {
  // x is a column vector
  // only the diagonal part need to be valid
  // all the values will be modified after the computation
  // w1 and w2 are working space which has same size with x.
  // w3 has length b
  fp32_t*   Ap      = A.A;
  int const b       = A.nb;
  int const nblocks = A.n / b;
  int const nbrow   = A.nbrow;
  int const nbcol   = A.nbcol;
  int const lda     = A.ld;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  bool const single_col = grid.npcol == 1;
  int const  left       = single_col
                       ? MPI_PROC_NULL
                       : (grid.mycol == 0 ? grid.npcol - 1 : grid.mycol - 1);
  int const right = single_col
                        ? MPI_PROC_NULL
                        : (grid.mycol == grid.npcol - 1 ? 0 : grid.mycol + 1);
  int const top    = (grid.myrow == 0 ? grid.nprow - 1 : grid.myrow - 1);
  int const bottom = (grid.myrow == grid.nprow - 1 ? 0 : grid.myrow + 1);

  fp64_t* W  = work;
  fp64_t* w1 = W + b * b;
  fp64_t* w2 = w1 + b * nbrow;
  fp64_t* w3 = w2 + b * nbrow;

  HPLMXP_set(b * nbrow, 0.0, w1);
  HIP_CHECK(hipDeviceSynchronize());

  for(int pj = nbcol - 1; pj >= 0; --pj) {
    MPI_Request req_recv_pivv = MPI_REQUEST_NULL, req_recv_v = MPI_REQUEST_NULL;
    int         gj = mycol + pj * npcol;

    if(gj < myrow) break;

    int  pivot   = gj % nprow;
    bool impivot = pivot == myrow;
    int  iend    = gj / nprow + (myrow <= pivot ? 1 : 0);

    if(impivot) {
      bool no_top = (gj == 0) || (grid.nprow == 1);

      if(gj != nblocks - 1) {
        MPI_Irecv(w1 + b * (iend - 1),
                  b,
                  T2MPI<fp64_t>::type,
                  right,
                  200,
                  grid.row_comm,
                  &req_recv_pivv);

        if(iend > 1) {
          MPI_Irecv(w1,
                    b * (iend - 1),
                    T2MPI<fp64_t>::type,
                    right,
                    200,
                    grid.row_comm,
                    &req_recv_v);
        }
      }

      HPLMXP_lacpyU(b, b, Mptr(Ap, (iend - 1) * b, pj * b, lda), lda, W, b);

      MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);

      HPLMXP_axpy(b, -1.0, w1 + b * (iend - 1), x + b * (iend - 1));

      // compute the pivot first
      HPLMXP_trsvU(b, W, b, x + b * (iend - 1));

      HIP_CHECK(hipEventRecord(piv, computeStream));

      if(iend > 1) {
        HPLMXP_gemv(b,
                    b,
                    1.0,
                    Mptr(Ap, (iend - 2) * b, pj * b, lda),
                    lda,
                    x + b * (iend - 1),
                    0.0,
                    w2 + b * (iend - 2));
      }

      if(!no_top) {
        /* sync */
        HIP_CHECK(hipEventSynchronize(piv));

        MPI_Send(x + b * (iend - 1),
                 b,
                 T2MPI<fp64_t>::type,
                 top,
                 100,
                 grid.col_comm);
      }

      if(iend > 1) {
        MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);

        HPLMXP_axpy(b, 1.0, w2 + b * (iend - 2), w1 + b * (iend - 2));

        /* sync */
        HIP_CHECK(hipStreamSynchronize(computeStream));

        if(iend > 2) {
          HPLMXP_gemv(b * (iend - 2),
                      b,
                      1.0,
                      Mptr(Ap, 0, pj * b, lda),
                      lda,
                      x + b * (iend - 1),
                      0.0,
                      w2);

          HPLMXP_axpy(b * (iend - 2), 1.0, w2, w1);
        }

        MPI_Send(w1 + b * (iend - 2),
                 b,
                 T2MPI<fp64_t>::type,
                 left,
                 200,
                 grid.row_comm);

        if(iend > 2) {
          /* sync */
          HIP_CHECK(hipStreamSynchronize(computeStream));

          MPI_Send(w1,
                   b * (iend - 2),
                   T2MPI<fp64_t>::type,
                   left,
                   200,
                   grid.row_comm);
        }
      }

    } else {

      bool stop_bcast = pivot == top || (gj < nprow && myrow == 0);

      if(gj != nblocks - 1) {
        MPI_Irecv(w1 + b * (iend - 1),
                  b,
                  T2MPI<fp64_t>::type,
                  right,
                  200,
                  grid.row_comm,
                  &req_recv_pivv);

        if(iend > 1) {
          MPI_Irecv(w1,
                    b * (iend - 1),
                    T2MPI<fp64_t>::type,
                    right,
                    200,
                    grid.row_comm,
                    &req_recv_v);
        }
      }

      MPI_Recv(w3,
               b,
               T2MPI<fp64_t>::type,
               bottom,
               100,
               grid.col_comm,
               MPI_STATUS_IGNORE);

      // compute the critical-path first
      HPLMXP_gemv(b,
                  b,
                  1.0,
                  Mptr(Ap, (iend - 1) * b, pj * b, lda),
                  lda,
                  w3,
                  0.0,
                  w2 + b * (iend - 1));

      if(!stop_bcast) {
        MPI_Send(w3, b, T2MPI<fp64_t>::type, top, 100, grid.col_comm);
      }

      MPI_Wait(&req_recv_pivv, MPI_STATUS_IGNORE);

      HPLMXP_axpy(b, 1.0, w2 + b * (iend - 1), w1 + b * (iend - 1));

      /* sync */
      HIP_CHECK(hipStreamSynchronize(computeStream));

      // compute others last
      if(iend > 1) {
        HPLMXP_gemv(
            b * (iend - 1), b, 1.0, Mptr(Ap, 0, pj * b, lda), lda, w3, 0.0, w2);
      }

      MPI_Send(w1 + b * (iend - 1),
               b,
               T2MPI<fp64_t>::type,
               left,
               200,
               grid.row_comm);

      if(iend > 1) {
        MPI_Wait(&req_recv_v, MPI_STATUS_IGNORE);

        HPLMXP_axpy(b * (iend - 1), 1.0, w2, w1);

        /* sync */
        HIP_CHECK(hipStreamSynchronize(computeStream));

        MPI_Send(
            w1, b * (iend - 1), T2MPI<fp64_t>::type, left, 200, grid.row_comm);
      }
    }
  }
}
