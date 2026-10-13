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
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>

/* Strict integer: the whole string must be a number */
static bool HPLMXP_pinfo_int(const char* s, int* v) {
  char* end;
  errno  = 0;
  long x = strtol(s, &end, 10);
  if(end == s || *end != '\0' || errno != 0 || x < INT_MIN || x > INT_MAX)
    return false;
  *v = (int)x;
  return true;
}

/*
 * The first COUNT whitespace-separated integers of LINE (anything after
 * them is a comment). False if there are fewer or one is malformed.
 */
static bool HPLMXP_pinfo_ints(const char* line, int count, int* v) {
  const char* s = line;
  for(int i = 0; i < count; ++i) {
    char* end;
    errno  = 0;
    long x = strtol(s, &end, 10);
    if(end == s || errno != 0 || x < INT_MIN || x > INT_MAX) return false;
    if(*end != '\0' && !isspace((unsigned char)*end)) return false;
    v[i] = (int)x;
    s    = end;
  }
  return true;
}

void HPLMXP_pinfo(HPLMXP_Comm     COMM,
                  int             ARGC,
                  char**          ARGV,
                  HPLMXP_T_test*  TEST,
                  int*            NS,
                  int*            N,
                  int*            NBS,
                  int*            NB,
                  HPLMXP_T_ORDER* PMAPPIN,
                  int*            P,
                  int*            Q,
                  int*            p,
                  int*            q,
                  int*            NTPS,
                  HPLMXP_T_TOP*   TP,
                  int*            ITS) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pinfo reads  the  startup  information for the various tests and
   * transmits it to all processes.
   *
   * Arguments
   * =========
   *
   * COMM    (global input)                HPLMXP_Comm
   *         On entry, COMM is the world communicator.
   *
   * TEST    (global output)               HPLMXP_T_test *
   *         On entry, TEST  points to a testing data structure.  On exit,
   *         the fields of this data structure are initialized as follows:
   *         TEST->outfp  specifies the output file where the results will
   *         be printed.  It is only defined and used by  the process 0 of
   *         the grid.  TEST->thrsh specifies the threshhold value for the
   *         test ratio.  TEST->epsil is the relative machine precision of
   *         the distributed computer.  Finally  the test counters, kfail,
   *         kpass, kskip, ktest are initialized to zero.
   *
   * NS      (global output)               int *
   *         On exit,  NS  specifies the number of different problem sizes
   *         to be tested. NS is less than or equal to HPLMXP_MAX_PARAM.
   *
   * N       (global output)               int *
   *         On entry, N is an array of dimension HPLMXP_MAX_PARAM.  On exit,
   *         the first NS entries of this array contain the  problem sizes
   *         to run the code with.
   *
   * NBS     (global output)               int *
   *         On exit,  NBS  specifies the number of different distribution
   *         blocking factors to be tested. NBS must be less than or equal
   *         to HPLMXP_MAX_PARAM.
   *
   * NB      (global output)               int *
   *         On exit,  PMAPPIN  specifies the process mapping onto the no-
   *         des of the  MPI machine configuration.  PMAPPIN  defaults  to
   *         row-major ordering.
   *
   * PMAPPIN (global output)               HPLMXP_T_ORDER *
   *         On entry, NB is an array of dimension HPLMXP_MAX_PARAM. On exit,
   *         the first NBS entries of this array contain the values of the
   *         various distribution blocking factors, to run the code with.
   *
   * P       (global output)               int *
   *         On exit, P specifies the number of rows in the MPI grid
   *
   * Q       (global output)               int *
   *         On exit, P specifies the number of columns in the MPI grid
   *
   * p       (global output)               int *
   *         On exit, p specifies the number of rows in the node-local MPI
   *         grid
   *
   * q       (global output)               int *
   *         On exit, q specifies the number of columns in the node-local
   *         MPI grid
   *
   * NTPS    (global output)               int *
   *         On exit, NTPS  specifies the  number of different values that
   *         can be used for the  broadcast topologies  to be tested. NTPS
   *         is less than or equal to HPLMXP_MAX_PARAM.
   *
   * TP      (global output)               HPLMXP_T_TOP *
   *         On entry, TP is an array of dimension HPLMXP_MAX_PARAM. On exit,
   *         the  first NTPS  entries of this  array  contain  the various
   *         broadcast (along rows) topologies to run the code with.
   *
   * ITS     (global output)               int *
   *         On exit,  ITS  specifies the number of iterations of each
   *         problem to run.
   *
   * ---------------------------------------------------------------------
   */

  char  file[HPLMXP_LINE_MAX], line[HPLMXP_LINE_MAX], auth[HPLMXP_LINE_MAX];
  FILE* infp   = NULL;
  int*  iwork  = NULL;
  int   lineno = 0;
  int   error  = 0, fid, i, j, lwork, nprocs, rank, size;

  rank = HPLMXP_comm_rank(COMM);
  size = HPLMXP_comm_size(COMM);
  /*
   * Initialize the TEST data structure with default values
   */
  TEST->outfp = stderr;
  TEST->epsil = 2.0e-16;
  TEST->thrsh = 16.0;
  TEST->kfail = TEST->kpass = TEST->kskip = TEST->ktest = 0;

  /*
   * Errors below are decided identically on every process (same command
   * line, or a value that has been reduced or broadcast), so every process
   * leaves through the same path. Process 0 reports.
   */
#define PINFO_FAIL(...)                                                        \
  do {                                                                         \
    if(rank == 0) HPLMXP_pwarn(stderr, __LINE__, "HPLMXP_pinfo", __VA_ARGS__); \
    HPLMXP_comm_finalize();                                                    \
    exit(1);                                                                   \
  } while(0)

  // parse settings
  int  _P = 1, _Q = 1, n = 61440, nb = 2560;
  int  _p = -1, _q = -1;
  int  _it = 1;
  bool cmdlinerun = false;
  bool inputfile  = false;

  std::string inputFileName = "HPL-MxP.dat";

  for(int i = 1; i < ARGC; i++) {
    const char* arg = ARGV[i];
    if(strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      if(rank == 0) {
        std::cout
            << "rocHPL-MxP client command line options:                      "
               "           \n"
               "-P  [ --ranksP ] arg (=1)           Specific MPI grid "
               "size: the number of      \n"
               "                                   rows in MPI grid.     "
               "                     \n"
               "-Q  [ --ranksQ ] arg (=1)           Specific MPI grid "
               "size: the number of      \n"
               "                                   columns in MPI grid.  "
               "                     \n"
               "-p  arg                            Node-local grid rows "
               "(default: 1)            \n"
               "-q  arg                            Node-local grid columns "
               "(default: ranks/node)\n"
               "-N  [ --sizeN ]  arg (=61440)       Specific matrix size: "
               "the number of rows   \n"
               "                                   /columns in global "
               "matrix.                 \n"
               "-NB [ --sizeNB ] arg (=2560)        Specific panel size: "
               "the number of rows    \n"
               "                                   /columns in panels.   "
               "                     \n"
               "-it arg (=1)                        Number of times to "
               "run each problem    \n"
               "-i  [ --input ]  arg (=HPL-MxP.dat) Input file. When set, "
               "all other commnand   \n"
               "                                    line parameters are "
               "ignored, and problem   \n"
               "                                    parameters are read "
               "from input file.       \n"
               "-h  [ --help ]                      Produces this help "
               "message                 \n"
               "--version                           Prints the version "
               "number                  \n";
      }
      HPLMXP_barrier(COMM);
      HPLMXP_comm_finalize();
      exit(0);
    }

    if(strcmp(arg, "--version") == 0) {
      if(rank == 0) {
        std::cout << "rocHPL-MxP version: " << __ROCHPLMXP_VER_MAJOR << "."
                  << __ROCHPLMXP_VER_MINOR << "." << __ROCHPLMXP_VER_PATCH
                  << std::endl;
      }
      HPLMXP_barrier(COMM);
      HPLMXP_comm_finalize();
      exit(0);
    }

    const bool isP  = !strcmp(arg, "-P") || !strcmp(arg, "--ranksP");
    const bool isQ  = !strcmp(arg, "-Q") || !strcmp(arg, "--ranksQ");
    const bool isp  = !strcmp(arg, "-p");
    const bool isq  = !strcmp(arg, "-q");
    const bool isN  = !strcmp(arg, "-N") || !strcmp(arg, "--sizeN");
    const bool isNB = !strcmp(arg, "-NB") || !strcmp(arg, "--sizeNB");
    const bool isIt = !strcmp(arg, "-it");
    const bool isIn = !strcmp(arg, "-i") || !strcmp(arg, "--input");
    if(!(isP || isQ || isp || isq || isN || isNB || isIt || isIn)) {
      if(rank == 0)
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "Ignoring unknown argument '%s'",
                     arg);
      continue;
    }
    if(i + 1 >= ARGC) PINFO_FAIL("Option %s needs a value. Exiting ...", arg);
    const char* val = ARGV[++i];
    if(isIn) {
      inputFileName = val;
      inputfile     = true;
      continue;
    }

    int v;
    if(!HPLMXP_pinfo_int(val, &v) || v < 1)
      PINFO_FAIL("Illegal value '%s' for %s, need a positive integer. "
                 "Exiting ...",
                 val,
                 arg);
    if(!isIt) cmdlinerun = true;
    if(isP) _P = v;
    if(isQ) _Q = v;
    if(isp) _p = v;
    if(isq) _q = v;
    if(isN) n = v;
    if(isNB) nb = v;
    if(isIt) _it = v;
  }

  *ITS = _it;

  if(inputfile == false && cmdlinerun == true) {
    // We were given run paramters via the cmd line so skip
    // trying to read from an input file and just fill a
    // TEST structure.

    /*
     * Problem size (>=0) (N)
     */
    *NS  = 1;
    N[0] = n;
    /*
     * Block size (>=1) (NB)
     */
    *NBS  = 1;
    NB[0] = nb;
    /*
     * Process grids, mapping, (>=1) (P, Q)
     */
    *PMAPPIN = HPLMXP_COLUMN_MAJOR;
    *P       = _P;
    *Q       = _Q;
    /*
     * Broadcast topology (TP) (0=rg, 1=2rg, 2=rgM, 3=2rgM, 4=L)
     */
    *NTPS = 1;
    TP[0] = HPLMXP_1RING;
  } else {
    /*
     * Process 0 reads the input data, broadcasts to other processes and
     * writes needed information to TEST->outfp.
     */
    const char* fname = inputFileName.c_str();
    if(rank == 0) {
      /*
       * Open file and skip data file header
       */
      if((infp = fopen(fname, "r")) == NULL) {
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "cannot open file %s: %s",
                     fname,
                     strerror(errno));
        error = 1;
        goto label_error;
      }

#define PINFO_LINE(what)                                  \
  if(fgets(line, HPLMXP_LINE_MAX - 2, infp) == NULL) {    \
    HPLMXP_pwarn(stderr,                                  \
                 __LINE__,                                \
                 "HPLMXP_pinfo",                          \
                 "%s: file ends at line %d, expected %s", \
                 fname,                                   \
                 lineno + 1,                              \
                 what);                                   \
    error = 1;                                            \
    goto label_error;                                     \
  }                                                       \
  lineno++
#define PINFO_BAD(what)            \
  do {                             \
    HPLMXP_pwarn(stderr,           \
                 __LINE__,         \
                 "HPLMXP_pinfo",   \
                 "%s line %d: %s", \
                 fname,            \
                 lineno,           \
                 what);            \
    error = 1;                     \
    goto label_error;              \
  } while(0)

      PINFO_LINE("the header");
      if(fgets(auth, HPLMXP_LINE_MAX - 2, infp) == NULL)
        PINFO_BAD("file too short");
      lineno++;
      /*
       * Read name and unit number for summary output file
       */
      PINFO_LINE("the output file name");
      if(sscanf(line, "%255s", file) != 1)
        PINFO_BAD("expected the output file name");
      PINFO_LINE("the output device");
      if(!HPLMXP_pinfo_ints(line, 1, &fid))
        PINFO_BAD("expected the output device (6=stdout, 7=stderr, file)");
      if(fid == 6)
        TEST->outfp = stdout;
      else if(fid == 7)
        TEST->outfp = stderr;
      else if((TEST->outfp = fopen(file, "w")) == NULL) {
        HPLMXP_pwarn(
            stderr, __LINE__, "HPLMXP_pinfo", "cannot open file %s.", file);
        error = 1;
        goto label_error;
      }
      /*
       * Read and check the parameter values for the tests.
       *
       * Problem size (>=0) (N)
       */
      PINFO_LINE("the number of problem sizes");
      if(!HPLMXP_pinfo_ints(line, 1, NS) || (*NS < 1) ||
         (*NS > HPLMXP_MAX_PARAM))
        PINFO_BAD("number of values of N is not in 1..20");
      PINFO_LINE("the problem sizes");
      if(!HPLMXP_pinfo_ints(line, *NS, N)) PINFO_BAD("expected NS values of N");
      for(i = 0; i < *NS; i++)
        if(N[i] < 0) PINFO_BAD("value of N less than 0");
      /*
       * Block size (>=1) (NB)
       */
      PINFO_LINE("the number of block sizes");
      if(!HPLMXP_pinfo_ints(line, 1, NBS) || (*NBS < 1) ||
         (*NBS > HPLMXP_MAX_PARAM))
        PINFO_BAD("number of values of NB is not in 1..20");
      PINFO_LINE("the block sizes");
      if(!HPLMXP_pinfo_ints(line, *NBS, NB))
        PINFO_BAD("expected NBS values of NB");
      for(i = 0; i < *NBS; i++)
        if(NB[i] < 1) PINFO_BAD("value of NB less than 1");
      /*
       * Process grids, mapping, (>=1) (P, Q)
       */
      PINFO_LINE("the process mapping");
      if(!HPLMXP_pinfo_ints(line, 1, &j))
        PINFO_BAD("expected the process mapping (0=row-, 1=column-major)");
      *PMAPPIN = (j == 1 ? HPLMXP_COLUMN_MAJOR : HPLMXP_ROW_MAJOR);

      PINFO_LINE("P");
      if(!HPLMXP_pinfo_ints(line, 1, P) || *P < 1)
        PINFO_BAD("value of P less than 1");
      PINFO_LINE("Q");
      if(!HPLMXP_pinfo_ints(line, 1, Q) || *Q < 1)
        PINFO_BAD("value of Q less than 1");
      /*
       * Checking threshold value (TEST->thrsh)
       */
      PINFO_LINE("the threshold");
      {
        char* end;
        TEST->thrsh = strtod(line, &end);
        if(end == line) PINFO_BAD("expected the residual threshold");
      }

      /*
       * Broadcast topology (TP) (0=rg, 1=2rg, 2=rgM, 3=2rgM, 4=L)
       */
      PINFO_LINE("the number of broadcast topologies");
      if(!HPLMXP_pinfo_ints(line, 1, NTPS) || (*NTPS < 1) ||
         (*NTPS > HPLMXP_MAX_PARAM))
        PINFO_BAD("number of values of BCAST is not in 1..20");
      PINFO_LINE("the broadcast topologies");
      {
        int tp[HPLMXP_MAX_PARAM];
        if(!HPLMXP_pinfo_ints(line, *NTPS, tp))
          PINFO_BAD("expected NTPS values of BCAST");
        for(i = 0; i < *NTPS; i++) {
          j = tp[i];
          if(j == 0)
            TP[i] = HPLMXP_1RING;
          else if(j == 1)
            TP[i] = HPLMXP_1RING_M;
          else if(j == 2)
            TP[i] = HPLMXP_2RING;
          else if(j == 3)
            TP[i] = HPLMXP_2RING_M;
          else if(j == 4)
            TP[i] = HPLMXP_BLONG;
          else // if(j == 5)
            TP[i] = HPLMXP_BLONG_M;
        }
      }
#undef PINFO_LINE
#undef PINFO_BAD

      /*
       * Close input file
       */
    label_error:
      if(infp != NULL) (void)fclose(infp);
    } else {
      TEST->outfp = NULL;
    }

    /*
     * Check for error on reading input file
     */
    HPLMXP_all_reduce_host(&error, 1, HPLMXP_MAX, COMM);
    if(error) PINFO_FAIL("Illegal input in file %s. Exiting ...", fname);
    /*
     * Pack information arrays and broadcast
     */
    HPLMXP_broadcast_host(&(TEST->thrsh), 1, 0, COMM);
    /*
     * Broadcast array sizes
     */
    iwork = (int*)malloc((size_t)(4) * sizeof(int));
    if(rank == 0) {
      iwork[0] = *NS;
      iwork[1] = *NBS;
      iwork[2] = (*PMAPPIN == HPLMXP_ROW_MAJOR ? 0 : 1);
      iwork[3] = *NTPS;
    }
    HPLMXP_broadcast_host(iwork, 4, 0, COMM);
    if(rank != 0) {
      *NS      = iwork[0];
      *NBS     = iwork[1];
      *PMAPPIN = (iwork[2] == 0 ? HPLMXP_ROW_MAJOR : HPLMXP_COLUMN_MAJOR);
      *NTPS    = iwork[3];
    }
    if(iwork) free(iwork);
    /*
     * Pack information arrays and broadcast
     */
    lwork = (*NS) + (*NBS) + 2 * 1 + (*NTPS) + 1;
    iwork = (int*)malloc((size_t)(lwork) * sizeof(int));
    if(rank == 0) {
      j = 0;
      for(i = 0; i < *NS; i++) {
        iwork[j] = N[i];
        j++;
      }
      for(i = 0; i < *NBS; i++) {
        iwork[j] = NB[i];
        j++;
      }
      iwork[j] = *P;
      j++;
      iwork[j] = *Q;
      j++;
      for(i = 0; i < *NTPS; i++) {
        if(TP[i] == HPLMXP_1RING)
          iwork[j] = 0;
        else if(TP[i] == HPLMXP_1RING_M)
          iwork[j] = 1;
        else if(TP[i] == HPLMXP_2RING)
          iwork[j] = 2;
        else if(TP[i] == HPLMXP_2RING_M)
          iwork[j] = 3;
        else if(TP[i] == HPLMXP_BLONG)
          iwork[j] = 4;
        else if(TP[i] == HPLMXP_BLONG_M)
          iwork[j] = 5;
        j++;
      }
      j++;
    }
    HPLMXP_broadcast_host(iwork, lwork, 0, COMM);
    if(rank != 0) {
      j = 0;
      for(i = 0; i < *NS; i++) {
        N[i] = iwork[j];
        j++;
      }
      for(i = 0; i < *NBS; i++) {
        NB[i] = iwork[j];
        j++;
      }
      *P = iwork[j];
      j++;
      *Q = iwork[j];
      j++;
      for(i = 0; i < *NTPS; i++) {
        if(iwork[j] == 0)
          TP[i] = HPLMXP_1RING;
        else if(iwork[j] == 1)
          TP[i] = HPLMXP_1RING_M;
        else if(iwork[j] == 2)
          TP[i] = HPLMXP_2RING;
        else if(iwork[j] == 3)
          TP[i] = HPLMXP_2RING_M;
        else if(iwork[j] == 4)
          TP[i] = HPLMXP_BLONG;
        else if(iwork[j] == 5)
          TP[i] = HPLMXP_BLONG_M;
        j++;
      }
      j++;
    }
    if(iwork) free(iwork);
  }

  /*
   * Check for enough processes in machine configuration
   */
  nprocs = (*P) * (*Q);
  if(nprocs > size)
    PINFO_FAIL("Need at least %d processes for a %d x %d grid, but there are "
               "%d. Exiting ...",
               nprocs,
               *P,
               *Q,
               size);

  /*Node-local grid: the same on every node*/
  int localSize = HPLMXP_comm_local_size();
  int localMin = localSize, localMax = localSize;
  HPLMXP_all_reduce_host(&localMin, 1, HPLMXP_MIN, COMM);
  HPLMXP_all_reduce_host(&localMax, 1, HPLMXP_MAX, COMM);
  if(localMin != localMax)
    PINFO_FAIL("Nodes run between %d and %d processes; every node needs the "
               "same number. Exiting ...",
               localMin,
               localMax);

  if(_p < 1 && _q < 1) { // Neither p nor q specified
    _q = localSize;      // Assume a 1xq node-local grid
    _p = 1;
  } else if(_p < 1) { // q specified
    if(localSize % _q != 0)
      PINFO_FAIL("Node-local MPI grid cannot be split into q=%d columns", _q);
    _p = localSize / _q;
  } else if(_q < 1) { // p specified
    if(localSize % _p != 0)
      PINFO_FAIL("Node-local MPI grid cannot be split into p=%d rows", _p);
    _q = localSize / _p;
  } else {
    if(localSize != _p * _q)
      PINFO_FAIL("Invalid Node-local MPI grid: p*q=%d*%d but %d processes per "
                 "node",
                 _p,
                 _q,
                 localSize);
  }

  /*Check grid can be distributed to nodes*/
  if(*Q % _q != 0 || *P % _p != 0)
    PINFO_FAIL("MPI grid is not uniformly distributed amoung nodes, "
               "(P,Q)=(%d,%d) and (p,q)=(%d,%d)",
               *P,
               *Q,
               _p,
               _q);
  /*
   * Node-local Process grids, mapping
   */
  *p = _p;
  *q = _q;

  /*
   * Compute and broadcast machine epsilon
   */
  TEST->epsil = HPLMXP_plamch<fp64_t>(COMM, HPLMXP_MACH_EPS);

  if(inputfile == false && cmdlinerun == true) {
    if(rank == 0) {
      if((TEST->outfp = fopen("HPL-MxP.out", "w")) == NULL) { error = 1; }
    }
    HPLMXP_all_reduce_host(&error, 1, HPLMXP_MAX, COMM);
    if(error) PINFO_FAIL("cannot open file HPL-MxP.out.");
  }
#undef PINFO_FAIL

  /*
   * regurgitate input
   */
  if(rank == 0) {
    HPLMXP_fprintf(TEST->outfp,
                   "%s%s\n",
                   "========================================",
                   "========================================");
    HPLMXP_fprintf(
        TEST->outfp,
        "%s%s\n",
        "HPLinpack 2.2  --  High-Performance Linpack benchmark  --  ",
        " February 24, 2016");
    HPLMXP_fprintf(TEST->outfp,
                   "%s%s\n",
                   "Written by A. Petitet and R. Clint Whaley,  ",
                   "Innovative Computing Laboratory, UTK");
    HPLMXP_fprintf(TEST->outfp,
                   "%s%s\n",
                   "Modified by Piotr Luszczek, ",
                   "Innovative Computing Laboratory, UTK");
    HPLMXP_fprintf(TEST->outfp,
                   "%s%s\n",
                   "Modified by Julien Langou, ",
                   "University of Colorado Denver");
    HPLMXP_fprintf(TEST->outfp,
                   "%s%s\n",
                   "========================================",
                   "========================================");

    HPLMXP_fprintf(TEST->outfp,
                   "\n%s\n",
                   "An explanation of the input/output parameters follows:");
    HPLMXP_fprintf(
        TEST->outfp, "%s\n", "T/V    : Wall time / encoded variant.");
    HPLMXP_fprintf(
        TEST->outfp, "%s\n", "N      : The order of the coefficient matrix A.");
    HPLMXP_fprintf(
        TEST->outfp, "%s\n", "NB     : The partitioning blocking factor.");
    HPLMXP_fprintf(TEST->outfp, "%s\n", "P      : The number of process rows.");
    HPLMXP_fprintf(
        TEST->outfp, "%s\n", "Q      : The number of process columns.");
    HPLMXP_fprintf(TEST->outfp,
                   "%s\n",
                   "Time   : Time in seconds to solve the linear system.");
    HPLMXP_fprintf(TEST->outfp,
                   "%s\n\n",
                   "Gflops : Rate of execution for solving the linear system.");
    HPLMXP_fprintf(
        TEST->outfp, "%s\n", "The following parameter values will be used:");
    /*
     * Problem size
     */
    HPLMXP_fprintf(TEST->outfp, "\nN      :");
    for(i = 0; i < Mmin(8, *NS); i++) HPLMXP_fprintf(TEST->outfp, "%8d ", N[i]);
    if(*NS > 8) {
      HPLMXP_fprintf(TEST->outfp, "\n        ");
      for(i = 8; i < Mmin(16, *NS); i++)
        HPLMXP_fprintf(TEST->outfp, "%8d ", N[i]);
      if(*NS > 16) {
        HPLMXP_fprintf(TEST->outfp, "\n        ");
        for(i = 16; i < *NS; i++) HPLMXP_fprintf(TEST->outfp, "%8d ", N[i]);
      }
    }
    /*
     * Distribution blocking factor
     */
    HPLMXP_fprintf(TEST->outfp, "\nNB     :");
    for(i = 0; i < Mmin(8, *NBS); i++)
      HPLMXP_fprintf(TEST->outfp, "%8d ", NB[i]);
    if(*NBS > 8) {
      HPLMXP_fprintf(TEST->outfp, "\n        ");
      for(i = 8; i < Mmin(16, *NBS); i++)
        HPLMXP_fprintf(TEST->outfp, "%8d ", NB[i]);
      if(*NBS > 16) {
        HPLMXP_fprintf(TEST->outfp, "\n        ");
        for(i = 16; i < *NBS; i++) HPLMXP_fprintf(TEST->outfp, "%8d ", NB[i]);
      }
    }
    /*
     * Process mapping
     */
    HPLMXP_fprintf(TEST->outfp, "\nPMAP   :");
    if(*PMAPPIN == HPLMXP_ROW_MAJOR)
      HPLMXP_fprintf(TEST->outfp, " Row-major process mapping");
    else if(*PMAPPIN == HPLMXP_COLUMN_MAJOR)
      HPLMXP_fprintf(TEST->outfp, " Column-major process mapping");
    /*
     * Process grid
     */
    HPLMXP_fprintf(TEST->outfp, "\nP      :");
    HPLMXP_fprintf(TEST->outfp, "%8d ", *P);
    HPLMXP_fprintf(TEST->outfp, "\nQ      :");
    HPLMXP_fprintf(TEST->outfp, "%8d ", *Q);

    /*
     * Broadcast topology
     */
    HPLMXP_fprintf(TEST->outfp, "\nBCAST  :");
    for(i = 0; i < Mmin(8, *NTPS); i++) {
      if(TP[i] == HPLMXP_1RING)
        HPLMXP_fprintf(TEST->outfp, "   1ring ");
      else if(TP[i] == HPLMXP_1RING_M)
        HPLMXP_fprintf(TEST->outfp, "  1ringM ");
      else if(TP[i] == HPLMXP_2RING)
        HPLMXP_fprintf(TEST->outfp, "   2ring ");
      else if(TP[i] == HPLMXP_2RING_M)
        HPLMXP_fprintf(TEST->outfp, "  2ringM ");
      else if(TP[i] == HPLMXP_BLONG)
        HPLMXP_fprintf(TEST->outfp, "   Blong ");
      else if(TP[i] == HPLMXP_BLONG_M)
        HPLMXP_fprintf(TEST->outfp, "  BlongM ");
    }
    if(*NTPS > 8) {
      HPLMXP_fprintf(TEST->outfp, "\n        ");
      for(i = 8; i < Mmin(16, *NTPS); i++) {
        if(TP[i] == HPLMXP_1RING)
          HPLMXP_fprintf(TEST->outfp, "   1ring ");
        else if(TP[i] == HPLMXP_1RING_M)
          HPLMXP_fprintf(TEST->outfp, "  1ringM ");
        else if(TP[i] == HPLMXP_2RING)
          HPLMXP_fprintf(TEST->outfp, "   2ring ");
        else if(TP[i] == HPLMXP_2RING_M)
          HPLMXP_fprintf(TEST->outfp, "  2ringM ");
        else if(TP[i] == HPLMXP_BLONG)
          HPLMXP_fprintf(TEST->outfp, "   Blong ");
        else if(TP[i] == HPLMXP_BLONG_M)
          HPLMXP_fprintf(TEST->outfp, "  BlongM ");
      }
      if(*NTPS > 16) {
        HPLMXP_fprintf(TEST->outfp, "\n        ");
        for(i = 16; i < *NTPS; i++) {
          if(TP[i] == HPLMXP_1RING)
            HPLMXP_fprintf(TEST->outfp, "   1ring ");
          else if(TP[i] == HPLMXP_1RING_M)
            HPLMXP_fprintf(TEST->outfp, "  1ringM ");
          else if(TP[i] == HPLMXP_2RING)
            HPLMXP_fprintf(TEST->outfp, "   2ring ");
          else if(TP[i] == HPLMXP_2RING_M)
            HPLMXP_fprintf(TEST->outfp, "  2ringM ");
          else if(TP[i] == HPLMXP_BLONG)
            HPLMXP_fprintf(TEST->outfp, "   Blong ");
          else if(TP[i] == HPLMXP_BLONG_M)
            HPLMXP_fprintf(TEST->outfp, "  BlongM ");
        }
      }
    }

    HPLMXP_fprintf(TEST->outfp, "\n\n");
    /*
     * For testing only
     */
    if(TEST->thrsh > HPLMXP_rzero) {
      HPLMXP_fprintf(TEST->outfp,
                     "%s%s\n\n",
                     "----------------------------------------",
                     "----------------------------------------");
      HPLMXP_fprintf(TEST->outfp,
                     "%s\n",
                     "- The matrix A is randomly generated for each test.");
      HPLMXP_fprintf(TEST->outfp,
                     "%s\n",
                     "- The following scaled residual check will be computed:");
      HPLMXP_fprintf(
          TEST->outfp,
          "%s\n",
          "      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || "
          "b ||_oo ) * N )");
      HPLMXP_fprintf(
          TEST->outfp,
          "%s %21.6e\n",
          "- The relative machine precision (eps) is taken to be     ",
          TEST->epsil);
      HPLMXP_fprintf(
          TEST->outfp,
          "%s   %11.1f\n\n",
          "- Computational tests pass if scaled residuals are less than      ",
          TEST->thrsh);
    }
  }
}
