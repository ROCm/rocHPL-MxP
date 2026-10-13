#include "hplmxp.hpp"

/*
 * The solves move three kinds of messages, each on its own communicator
 * (upstream used tags 100/200 on the column/row communicators):
 *
 *   piv  row ring   the partial sum for the next diagonal block (b values),
 *                   on the critical path from one pivot to the next
 *   vec  row ring   the partial sums for the remaining blocks
 *   col  column     the solved block of x, passed along the process column
 *
 * piv and vec are both left-to-right (L) or right-to-left (U) and FIFO
 * would keep them apart on one communicator, but with RCCL the operations
 * of a communicator execute in order on its stream: a non-pivot process
 * must forward piv before vec from its left neighbour has arrived, so
 * they need separate communicators to overlap. They are duplicates of
 * grid.row_comm/col_comm created on the first solve (every rank calls the
 * solves in the same order, so the collective splits are safe) and kept
 * until the backend finalizes; ptrsv is their only user.
 *
 * Deadlock freedom. Label every operation of the solve for global block
 * column gj: the col messages of column gj get label gj, the piv/vec
 * messages from the owner of column gj to its neighbour get gj + 1/2 in L
 * (gj - 1/2 in U, where they flow right to left). On every process the
 * operations are issued, and therefore queued on every stream, in
 * monotone label order, and a matched pair always has equal labels. Any
 * wait is then on an operation with a smaller label, or on one with the
 * same label inside an acyclic pattern (a chain down the column, or the
 * two-message piv/vec handoff between neighbours), so no cycle of waits
 * exists even if a process executes all its operations one at a time.
 * Hence no group is needed in the solve. Ring connections are made up
 * front (ptrsv_connect) so that the first message to a peer does not run
 * RCCL's host-blocking connection handshake in the middle of the solve.
 */
static struct {
  HPLMXP_Comm row_src = HPLMXP_COMM_NULL; /* grid comms the dups belong to */
  HPLMXP_Comm col_src = HPLMXP_COMM_NULL;
  HPLMXP_Comm piv     = HPLMXP_COMM_NULL;
  HPLMXP_Comm vec     = HPLMXP_COMM_NULL;
  HPLMXP_Comm col     = HPLMXP_COMM_NULL;
} pc;

static HPLMXP_Comm ptrsv_dup(HPLMXP_Comm comm) {
  HPLMXP_Comm dup = HPLMXP_COMM_NULL;
  HPLMXP_comm_split(comm, 0, HPLMXP_comm_rank(comm), &dup);
  return dup;
}

static void ptrsv_connect(HPLMXP_Comm comm, fp64_t* buf) {
  int const rank = HPLMXP_comm_rank(comm);
  int const size = HPLMXP_comm_size(comm);
  int const next = (rank + 1) % size;
  int const prev = (rank + size - 1) % size;

  HPLMXP_group_start();
  HPLMXP_send(buf, 1, next, comm, computeStream);
  HPLMXP_send(buf + 1, 1, prev, comm, computeStream);
  HPLMXP_recv(buf + 2, 1, prev, comm, computeStream);
  HPLMXP_recv(buf + 3, 1, next, comm, computeStream);
  HPLMXP_group_end();
}

static void ptrsv_comms(const HPLMXP_T_grid& grid) {
  if(pc.row_src == grid.row_comm && pc.col_src == grid.col_comm) return;

  HPLMXP_comm_free(&pc.piv);
  HPLMXP_comm_free(&pc.vec);
  HPLMXP_comm_free(&pc.col);

  fp64_t* buf;
  HIP_CHECK(hipMalloc(&buf, 4 * sizeof(fp64_t)));
  HIP_CHECK(hipMemset(buf, 0, 4 * sizeof(fp64_t)));
  if(grid.npcol > 1) {
    pc.piv = ptrsv_dup(grid.row_comm);
    pc.vec = ptrsv_dup(grid.row_comm);
    ptrsv_connect(pc.piv, buf);
    ptrsv_connect(pc.vec, buf);
  }
  if(grid.nprow > 1) {
    pc.col = ptrsv_dup(grid.col_comm);
    ptrsv_connect(pc.col, buf);
  }
  HIP_CHECK(hipFree(buf));

  pc.row_src = grid.row_comm;
  pc.col_src = grid.col_comm;
}

/*
 * The sends of a column are completed (in stream order) before the next
 * column reuses w1/w3, and before the solve returns.
 */
static void ptrsv_wait_sends(HPLMXP_Request* sends) {
  for(int i = 0; i < 3; ++i) HPLMXP_wait(&sends[i], computeStream);
}

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
  int const  left =
      single_col ? -1 : (grid.mycol == 0 ? grid.npcol - 1 : grid.mycol - 1);
  int const right =
      single_col ? -1 : (grid.mycol == grid.npcol - 1 ? 0 : grid.mycol + 1);
  int const top    = (grid.myrow == 0 ? grid.nprow - 1 : grid.myrow - 1);
  int const bottom = (grid.myrow == grid.nprow - 1 ? 0 : grid.myrow + 1);

  fp64_t* W  = work;
  fp64_t* w1 = W + b * b;
  fp64_t* w2 = w1 + b * nbrow;
  fp64_t* w3 = w2 + b * nbrow;

  ptrsv_comms(grid);

  /* set value */
  HPLMXP_set(b * nbrow, 0.0, w1);
  HPLMXP_comm_sync(nullptr, "the device wait at HPLMXP_ptrsvL entry");

  // sends of the previous column: x down, piv and vec to the right
  HPLMXP_Request sends[3] = {
      HPLMXP_REQUEST_NULL, HPLMXP_REQUEST_NULL, HPLMXP_REQUEST_NULL};
  HPLMXP_Request& send_x   = sends[0];
  HPLMXP_Request& send_piv = sends[1];
  HPLMXP_Request& send_v   = sends[2];

  for(int pj = 0; pj < nbcol; ++pj) {
    HPLMXP_Request recv_piv = HPLMXP_REQUEST_NULL, recv_v = HPLMXP_REQUEST_NULL;
    HPLMXP_Request recv_x = HPLMXP_REQUEST_NULL;
    int            gj     = mycol + pj * npcol;
    int            cleft  = gj == 0 ? -1 : left;
    int            pivot  = gj % nprow;
    int            istart = (gj < myrow ? 0 : (gj - myrow + nprow - 1) / nprow);
    int            ii     = myrow + istart * nprow;

    if(ii >= nblocks) break;

    bool impivot   = pivot == myrow;
    bool no_bottom = grid.nprow == 1 || ii + 1 >= nblocks;

    ptrsv_wait_sends(sends);

    if(impivot) {
      HPLMXP_irecv(w1 + istart * b, b, cleft, pc.piv, computeStream, &recv_piv);

      if(istart + 1 < nbrow) {
        HPLMXP_irecv(w1 + b * (istart + 1),
                     b * (nbrow - istart - 1),
                     cleft,
                     pc.vec,
                     computeStream,
                     &recv_v);
      }

      HPLMXP_lacpyL(b, b, Mptr(Ap, istart * b, pj * b, lda), lda, W, b);

      HPLMXP_wait(&recv_piv, computeStream);

      // compute the pivot first
      HPLMXP_axpy(b, -1.0, w1 + b * istart, x + b * istart);

      HPLMXP_trsvL(b, W, b, x + b * istart);

      if(!no_bottom) {
        HPLMXP_isend(x + b * istart, b, bottom, pc.col, computeStream, &send_x);
      }

      if(istart + 1 < nbrow) {
        HPLMXP_gemv(b,
                    b,
                    1.0,
                    Mptr(Ap, (istart + 1) * b, pj * b, lda),
                    lda,
                    x + b * istart,
                    0.0,
                    w2 + b * (istart + 1));

        HPLMXP_wait(&recv_v, computeStream);

        HPLMXP_axpy(b, 1.0, w2 + b * (istart + 1), w1 + b * (istart + 1));

        HPLMXP_isend(
            w1 + b * (istart + 1), b, right, pc.piv, computeStream, &send_piv);

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

          HPLMXP_isend(w1 + b * (istart + 2),
                       b * (nbrow - istart - 2),
                       right,
                       pc.vec,
                       computeStream,
                       &send_v);
        }
      }

    } else {

      bool bottom_is_pivot = pivot == bottom;

      HPLMXP_irecv(w1 + istart * b, b, cleft, pc.piv, computeStream, &recv_piv);

      if(istart + 1 < nbrow) {
        HPLMXP_irecv(w1 + (istart + 1) * b,
                     b * (nbrow - istart - 1),
                     cleft,
                     pc.vec,
                     computeStream,
                     &recv_v);
      }

      HPLMXP_irecv(w3, b, top, pc.col, computeStream, &recv_x);

      HPLMXP_wait(&recv_x, computeStream);

      // w3 came in on pc.col: forward it without waiting for computeStream
      if(!bottom_is_pivot && !no_bottom) {
        HPLMXP_isend(w3, b, bottom, pc.col, nullptr, &send_x);
      }

      // compute the critical-path first
      HPLMXP_gemv(b,
                  b,
                  1.0,
                  Mptr(Ap, istart * b, pj * b, lda),
                  lda,
                  w3,
                  0.0,
                  w2 + b * istart);

      HPLMXP_wait(&recv_piv, computeStream);

      HPLMXP_axpy(b, 1.0, w2 + b * istart, w1 + b * istart);

      HPLMXP_isend(w1 + b * istart, b, right, pc.piv, computeStream, &send_piv);

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

        HPLMXP_wait(&recv_v, computeStream);

        HPLMXP_axpy(b * (nbrow - istart - 1),
                    1.0,
                    w2 + b * (istart + 1),
                    w1 + b * (istart + 1));

        HPLMXP_isend(w1 + b * (istart + 1),
                     b * (nbrow - istart - 1),
                     right,
                     pc.vec,
                     computeStream,
                     &send_v);
      }
    }
  }

  ptrsv_wait_sends(sends);
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
  int const  left =
      single_col ? -1 : (grid.mycol == 0 ? grid.npcol - 1 : grid.mycol - 1);
  int const right =
      single_col ? -1 : (grid.mycol == grid.npcol - 1 ? 0 : grid.mycol + 1);
  int const top    = (grid.myrow == 0 ? grid.nprow - 1 : grid.myrow - 1);
  int const bottom = (grid.myrow == grid.nprow - 1 ? 0 : grid.myrow + 1);

  fp64_t* W  = work;
  fp64_t* w1 = W + b * b;
  fp64_t* w2 = w1 + b * nbrow;
  fp64_t* w3 = w2 + b * nbrow;

  ptrsv_comms(grid);

  HPLMXP_set(b * nbrow, 0.0, w1);
  HPLMXP_comm_sync(
      nullptr,
      "the device wait at HPLMXP_ptrsvU entry (covers the ptrsvL p2p)");

  // sends of the previous column: x up, piv and vec to the left
  HPLMXP_Request sends[3] = {
      HPLMXP_REQUEST_NULL, HPLMXP_REQUEST_NULL, HPLMXP_REQUEST_NULL};
  HPLMXP_Request& send_x   = sends[0];
  HPLMXP_Request& send_piv = sends[1];
  HPLMXP_Request& send_v   = sends[2];

  for(int pj = nbcol - 1; pj >= 0; --pj) {
    HPLMXP_Request recv_piv = HPLMXP_REQUEST_NULL, recv_v = HPLMXP_REQUEST_NULL;
    HPLMXP_Request recv_x = HPLMXP_REQUEST_NULL;
    int            gj     = mycol + pj * npcol;

    if(gj < myrow) break;

    int  pivot   = gj % nprow;
    bool impivot = pivot == myrow;
    int  iend    = gj / nprow + (myrow <= pivot ? 1 : 0);

    ptrsv_wait_sends(sends);

    if(impivot) {
      bool no_top = (gj == 0) || (grid.nprow == 1);

      if(gj != nblocks - 1) {
        HPLMXP_irecv(
            w1 + b * (iend - 1), b, right, pc.piv, computeStream, &recv_piv);

        if(iend > 1) {
          HPLMXP_irecv(
              w1, b * (iend - 1), right, pc.vec, computeStream, &recv_v);
        }
      }

      HPLMXP_lacpyU(b, b, Mptr(Ap, (iend - 1) * b, pj * b, lda), lda, W, b);

      HPLMXP_wait(&recv_piv, computeStream);

      HPLMXP_axpy(b, -1.0, w1 + b * (iend - 1), x + b * (iend - 1));

      // compute the pivot first
      HPLMXP_trsvU(b, W, b, x + b * (iend - 1));

      if(!no_top) {
        HPLMXP_isend(
            x + b * (iend - 1), b, top, pc.col, computeStream, &send_x);
      }

      if(iend > 1) {
        HPLMXP_gemv(b,
                    b,
                    1.0,
                    Mptr(Ap, (iend - 2) * b, pj * b, lda),
                    lda,
                    x + b * (iend - 1),
                    0.0,
                    w2 + b * (iend - 2));

        HPLMXP_wait(&recv_v, computeStream);

        HPLMXP_axpy(b, 1.0, w2 + b * (iend - 2), w1 + b * (iend - 2));

        HPLMXP_isend(
            w1 + b * (iend - 2), b, left, pc.piv, computeStream, &send_piv);

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

          HPLMXP_isend(
              w1, b * (iend - 2), left, pc.vec, computeStream, &send_v);
        }
      }

    } else {

      bool stop_bcast = pivot == top || (gj < nprow && myrow == 0);

      if(gj != nblocks - 1) {
        HPLMXP_irecv(
            w1 + b * (iend - 1), b, right, pc.piv, computeStream, &recv_piv);

        if(iend > 1) {
          HPLMXP_irecv(
              w1, b * (iend - 1), right, pc.vec, computeStream, &recv_v);
        }
      }

      HPLMXP_irecv(w3, b, bottom, pc.col, computeStream, &recv_x);

      HPLMXP_wait(&recv_x, computeStream);

      // w3 came in on pc.col: forward it without waiting for computeStream
      if(!stop_bcast) { HPLMXP_isend(w3, b, top, pc.col, nullptr, &send_x); }

      // compute the critical-path first
      HPLMXP_gemv(b,
                  b,
                  1.0,
                  Mptr(Ap, (iend - 1) * b, pj * b, lda),
                  lda,
                  w3,
                  0.0,
                  w2 + b * (iend - 1));

      HPLMXP_wait(&recv_piv, computeStream);

      HPLMXP_axpy(b, 1.0, w2 + b * (iend - 1), w1 + b * (iend - 1));

      HPLMXP_isend(
          w1 + b * (iend - 1), b, left, pc.piv, computeStream, &send_piv);

      // compute others last
      if(iend > 1) {
        HPLMXP_gemv(
            b * (iend - 1), b, 1.0, Mptr(Ap, 0, pj * b, lda), lda, w3, 0.0, w2);

        HPLMXP_wait(&recv_v, computeStream);

        HPLMXP_axpy(b * (iend - 1), 1.0, w2, w1);

        HPLMXP_isend(w1, b * (iend - 1), left, pc.vec, computeStream, &send_v);
      }
    }
  }

  ptrsv_wait_sends(sends);
}
