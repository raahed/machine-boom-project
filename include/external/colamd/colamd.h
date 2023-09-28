/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/* ========================================================================== */
/* === colamd prototypes and definitions ==================================== */
/* ========================================================================== */

/*
    This is the colamd include file,

  http://www.cise.ufl.edu/~davis/colamd/colamd.h

    for use in the colamd.c, colamdmex.c, and symamdmex.c files located at

  http://www.cise.ufl.edu/~davis/colamd/

    See those files for a description of colamd and symamd, and for the
    copyright notice, which also applies to this file.

    August 3, 1998.  Version 1.0.
*/

#ifndef COLAMD_COLAMD_H
#define COLAMD_COLAMD_H

#ifdef __cplusplus
extern "C" {
#endif


#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( COLAMD_LIBRARY ) || defined( colamd_EXPORTS )
#    define COLAMD_EXPORT   __declspec(dllexport)
#  else
#    define COLAMD_EXPORT   __declspec(dllimport)
#  endif
#else
#  define COLAMD_EXPORT
#endif



/* ========================================================================== */
/* === Definitions ========================================================== */
/* ========================================================================== */

/* size of the knobs [ ] array.  Only knobs [0..1] are currently used. */
#define COLAMD_KNOBS 20

/* number of output statistics.  Only A [0..2] are currently used. */
#define COLAMD_STATS 20

/* knobs [0] and A [0]: dense row knob and output statistic. */
#define COLAMD_DENSE_ROW 0

/* knobs [1] and A [1]: dense column knob and output statistic. */
#define COLAMD_DENSE_COL 1

/* A [2]: memory defragmentation count output statistic */
#define COLAMD_DEFRAG_COUNT 2

/* A [3]: whether or not the input columns were jumbled or had duplicates */
#define COLAMD_JUMBLED_COLS 3

/* ========================================================================== */
/* === Prototypes of user-callable routines ================================= */
/* ========================================================================== */

COLAMD_EXPORT int agx_colamd_recommended    /* returns recommended value of Alen */
(
    int nnz,      /* nonzeros in A */
    int n_row,      /* number of rows in A */
    int n_col      /* number of columns in A */
) ;

COLAMD_EXPORT void agx_colamd_set_defaults  /* sets default parameters */
(        /* knobs argument is modified on output */
    double knobs [COLAMD_KNOBS]  /* parameter settings for colamd */
) ;

/**
 colamd,
 Renamed interface colamd vs agx_colamd to avoid interfering with newer versions
 of colamd.
*/
COLAMD_EXPORT int agx_colamd      /* returns TRUE if successful, FALSE otherwise*/
(        /* A and p arguments are modified on output */
    int n_row,      /* number of rows in A */
    int n_col,      /* number of columns in A */
    int Alen,      /* size of the array A */
    int A [],      /* row indices of A, of size Alen */
    int p [],      /* column pointers of A, of size n_col+1 */
    double knobs [COLAMD_KNOBS]  /* parameter settings for colamd */
) ;


/**
 symamd, interface to colamd for symmetric matrices.

 This modified version has a renamed interface (symamd vs symamd) to avoid
 interfering with newer versions of symamd which is not compatible due to stats
 changes.

 Input matrix will be overwritten and stats returned in A

 \return The return value from colamd, 0 for failure and 1 for success.
*/
COLAMD_EXPORT int agx_symamd
(
  int n,        // Matrix size
  int* A,       // row pointers
  int* p,       // column pointers
  int* perm,    // output order, used internally, must be able to hold n+1 items
  double knobs [COLAMD_KNOBS]
);

#ifdef __cplusplus
}
#endif

#endif

