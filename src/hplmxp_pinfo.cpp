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

void HPLMXP_pinfo(int             ARGC,
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

  char file[HPLMXP_LINE_MAX], line[HPLMXP_LINE_MAX], auth[HPLMXP_LINE_MAX],
      num[HPLMXP_LINE_MAX];
  FILE* infp;
  int*  iwork = NULL;
  char* lineptr;
  int   error = 0, fid, i, j, lwork, maxp, nprocs, rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /*
   * Initialize the TEST data structure with default values
   */
  TEST->outfp = stderr;
  TEST->epsil = 2.0e-16;
  TEST->thrsh = 16.0;
  TEST->kfail = TEST->kpass = TEST->kskip = TEST->ktest = 0;

  // parse settings
  int  _P = 1, _Q = 1, n = 61440, nb = 2560;
  int  _p = -1, _q = -1;
  int  _it = 1;
  bool cmdlinerun = false;
  bool inputfile  = false;

  std::string inputFileName = "HPL-MxP.dat";

  for(int i = 1; i < ARGC; i++) {
    if(strcmp(ARGV[i], "-h") == 0 || strcmp(ARGV[i], "--help") == 0) {
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
               "-N  [ --sizeN ]  arg (=61440)       Specific matrix size: "
               "the number of rows   \n"
               "                                   /columns in global "
               "matrix.                 \n"
               "-NB [ --sizeNB ] arg (=2560)        Specific panel size: "
               "the number of rows    \n"
               "                                   /columns in panels.   "
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
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);
    }

    if(strcmp(ARGV[i], "--version") == 0) {
      if(rank == 0) {
        std::cout << "rocHPL-MxP version: " << __ROCHPLMXP_VER_MAJOR << "."
                  << __ROCHPLMXP_VER_MINOR << "." << __ROCHPLMXP_VER_PATCH
                  << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      exit(0);
    }

    if(strcmp(ARGV[i], "-P") == 0 || strcmp(ARGV[i], "--ranksP") == 0) {
      _P         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(_P < 1) {
        if(rank == 0)
          HPLMXP_pwarn(stderr,
                       __LINE__,
                       "HPLMXP_pinfo",
                       "Illegal value for P. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-Q") == 0 || strcmp(ARGV[i], "--ranksQ") == 0) {
      _Q         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(_Q < 1) {
        if(rank == 0)
          HPLMXP_pwarn(stderr,
                       __LINE__,
                       "HPLMXP_pinfo",
                       "Illegal value for Q. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-p") == 0) {
      _p         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
    }
    if(strcmp(ARGV[i], "-q") == 0) {
      _q         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
    }

    if(strcmp(ARGV[i], "-N") == 0 || strcmp(ARGV[i], "--sizeN") == 0) {
      n          = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(n < 1) {
        if(rank == 0)
          HPLMXP_pwarn(stderr,
                       __LINE__,
                       "HPLMXP_pinfo",
                       "Illegal value for N. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-NB") == 0 || strcmp(ARGV[i], "--sizeNB") == 0) {
      nb         = atoi(ARGV[i + 1]);
      cmdlinerun = true;
      i++;
      if(nb < 1) {
        if(rank == 0)
          HPLMXP_pwarn(stderr,
                       __LINE__,
                       "HPLMXP_pinfo",
                       "Illegal value for NB. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-it") == 0) {
      _it = atoi(ARGV[i + 1]);
      i++;
      if(_it < 1) {
        if(rank == 0)
          HPLMXP_pwarn(stderr,
                       __LINE__,
                       "HPLMXP_pinfo",
                       "Invalid number of iterations. Exiting ...");
        MPI_Finalize();
        exit(1);
      }
    }
    if(strcmp(ARGV[i], "-i") == 0 || strcmp(ARGV[i], "--input") == 0) {
      inputFileName = ARGV[i + 1];
      inputfile     = true;
      i++;
    }
  }

  /*
   * Check for enough processes in machine configuration
   */
  maxp = _P * _Q;
  if(maxp > size) {
    if(rank == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pinfo",
                   "Need at least %d processes for these tests",
                   maxp);
    MPI_Finalize();
    exit(1);
  }

  /*Node-local grid*/
  MPI_Comm nodeComm;
  MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);

  int localRank;
  int localSize;
  MPI_Comm_rank(nodeComm, &localRank);
  MPI_Comm_size(nodeComm, &localSize);

  if(_p < 1 && _q < 1) { // Neither p nor q specified
    _q = localSize;      // Assume a 1xq node-local grid
    _p = 1;
  } else if(_p < 1) { // q specified
    if(localSize % _q != 0) {
      if(rank == 0)
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "Node-local MPI grid cannot be split into q=%d columns",
                     _q);
      MPI_Finalize();
      exit(1);
    }
    _p = localSize / _q;
  } else if(_q < 1) { // p specified
    if(localSize % _p != 0) {
      if(rank == 0)
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "Node-local MPI grid cannot be split into p=%d rows",
                     _p);
      MPI_Finalize();
      exit(1);
    }
    _q = localSize / _p;
  } else {
    if(localSize != _p * _q) {
      if(rank == 0)
        HPLMXP_pwarn(
            stderr, __LINE__, "HPLMXP_pinfo", "Invalid Node-local MPI grid");
      MPI_Finalize();
      exit(1);
    }
  }

  /*Check grid can be distributed to nodes*/
  if(_Q % _q != 0 || _P % _p != 0) {
    if(rank == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pinfo",
                   "MPI grid is not uniformly distributed amoung nodes, "
                   "(P,Q)=(%d,%d) and (p,q)=(%d,%d)",
                   _P,
                   _Q,
                   _p,
                   _q);
    MPI_Finalize();
    exit(1);
  }
  MPI_Comm_free(&nodeComm);
  /*
   * Node-local Process grids, mapping
   */
  *p = _p;
  *q = _q;

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

    /*
     * Compute and broadcast machine epsilon
     */
    TEST->epsil = HPLMXP_plamch<fp64_t>(MPI_COMM_WORLD, HPLMXP_MACH_EPS);

    if(rank == 0) {
      if((TEST->outfp = fopen("HPL-MxP.out", "w")) == NULL) { error = 1; }
    }
    HPLMXP_all_reduce(&error, 1, HPLMXP_MAX, MPI_COMM_WORLD);
    if(error) {
      if(rank == 0)
        HPLMXP_pwarn(
            stderr, __LINE__, "HPLMXP_pinfo", "cannot open file HPL-MxP.out.");
      MPI_Finalize();
      exit(1);
    }
  } else {
    /*
     * Process 0 reads the input data, broadcasts to other processes and
     * writes needed information to TEST->outfp.
     */
    char* status;
    if(rank == 0) {
      /*
       * Open file and skip data file header
       */
      if((infp = fopen(inputFileName.c_str(), "r")) == NULL) {
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "cannot open file %s",
                     inputFileName.c_str());
        error = 1;
        goto label_error;
      }

      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      status = fgets(auth, HPLMXP_LINE_MAX - 2, infp);
      /*
       * Read name and unit number for summary output file
       */
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", file);
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      fid = atoi(num);
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
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      *NS = atoi(num);
      if((*NS < 1) || (*NS > HPLMXP_MAX_PARAM)) {
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "%s %d",
                     "Number of values of N is less than 1 or greater than",
                     HPLMXP_MAX_PARAM);
        error = 1;
        goto label_error;
      }

      status  = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      lineptr = line;
      for(i = 0; i < *NS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        if((N[i] = atoi(num)) < 0) {
          HPLMXP_pwarn(
              stderr, __LINE__, "HPLMXP_pinfo", "Value of N less than 0");
          error = 1;
          goto label_error;
        }
      }
      /*
       * Block size (>=1) (NB)
       */
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      *NBS = atoi(num);
      if((*NBS < 1) || (*NBS > HPLMXP_MAX_PARAM)) {
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "%s %s %d",
                     "Number of values of NB is less than 1 or",
                     "greater than",
                     HPLMXP_MAX_PARAM);
        error = 1;
        goto label_error;
      }

      status  = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      lineptr = line;
      for(i = 0; i < *NBS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        if((NB[i] = atoi(num)) < 1) {
          HPLMXP_pwarn(
              stderr, __LINE__, "HPLMXP_pinfo", "Value of NB less than 1");
          error = 1;
          goto label_error;
        }
      }
      /*
       * Process grids, mapping, (>=1) (P, Q)
       */
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      *PMAPPIN = (atoi(num) == 1 ? HPLMXP_COLUMN_MAJOR : HPLMXP_ROW_MAJOR);

      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      if((*P = atoi(num)) < 1) {
        HPLMXP_pwarn(
            stderr, __LINE__, "HPLMXP_pinfo", "Value of P less than 1");
        error = 1;
        goto label_error;
      }

      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      if((*Q = atoi(num)) < 1) {
        HPLMXP_pwarn(
            stderr, __LINE__, "HPLMXP_pinfo", "Value of Q less than 1");
        error = 1;
        goto label_error;
      }

      /*
       * Check for enough processes in machine configuration
       */
      nprocs = (*P) * (*Q);
      if(nprocs > size) {
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "Need at least %d processes for these tests",
                     nprocs);
        error = 1;
        goto label_error;
      }
      /*
       * Checking threshold value (TEST->thrsh)
       */
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      TEST->thrsh = atof(num);

      /*
       * Broadcast topology (TP) (0=rg, 1=2rg, 2=rgM, 3=2rgM, 4=L)
       */
      status = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      (void)sscanf(line, "%s", num);
      *NTPS = atoi(num);
      if((*NTPS < 1) || (*NTPS > HPLMXP_MAX_PARAM)) {
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "%s %s %d",
                     "Number of values of BCAST",
                     "is less than 1 or greater than",
                     HPLMXP_MAX_PARAM);
        error = 1;
        goto label_error;
      }
      status  = fgets(line, HPLMXP_LINE_MAX - 2, infp);
      lineptr = line;
      for(i = 0; i < *NTPS; i++) {
        (void)sscanf(lineptr, "%s", num);
        lineptr += strlen(num) + 1;
        j = atoi(num);
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

      /*
       * Close input file
       */
    label_error:
      (void)fclose(infp);
    } else {
      TEST->outfp = NULL;
    }

    /*
     * Check for error on reading input file
     */
    HPLMXP_all_reduce(&error, 1, HPLMXP_MAX, MPI_COMM_WORLD);
    if(error) {
      if(rank == 0)
        HPLMXP_pwarn(stderr,
                     __LINE__,
                     "HPLMXP_pinfo",
                     "Illegal input in file HPL-MxP.dat. Exiting ...");
      MPI_Finalize();
      exit(1);
    }
    /*
     * Compute and broadcast machine epsilon
     */
    TEST->epsil = HPLMXP_plamch<fp64_t>(MPI_COMM_WORLD, HPLMXP_MACH_EPS);
    /*
     * Pack information arrays and broadcast
     */
    HPLMXP_broadcast(&(TEST->thrsh), 1, 0, MPI_COMM_WORLD);
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
    HPLMXP_broadcast(iwork, 4, 0, MPI_COMM_WORLD);
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
    HPLMXP_broadcast(iwork, lwork, 0, MPI_COMM_WORLD);
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
