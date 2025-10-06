/**********************************************************************
Copyright (c) 2011-2024 Intel Corporation.
Copyright (c) 2025 Michael H. Anderson. All rights reserved.
This software includes contributions protected by
U.S. Patents 11,848,686 and 12,341,532.

Redistribution and use in source and binary forms, with or without
modification, are permitted for non-commercial evaluation purposes
only, provided that the following conditions are met:

Redistributions of source code must retain the above copyright notices,
patent notices, this list of conditions, and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notices,
patent notices, this list of conditions, and the following disclaimer in the
documentation and/or other materials provided with the distribution.

Neither the name of Intel Corporation, nor Michael H. Anderson, nor the names
of their contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

Commercial deployment or use of this software requires a separate license
from the copyright holders and patent owners.

In other words, this code is provided solely for the purposes of
evaluation and is not licensed or intended to be licensed or used as part of
or in connection with any commercial or non-commercial use other than evaluation
of the potential for a license from Michael H. Anderson. Neither Michael H. Anderson
nor any affiliated person grants any express or implied rights under any patents,
copyrights, trademarks, or trade secret information. No content may be copied,
stored, or utilized in any way without express written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES,
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SPDX-License-Identifier: LicenseRef-Intel-Anderson-BSD-3-Clause-With-Restrictions
**********************************************************************/

#ifndef __aarch64__

#include "ec_base.h"
#include "erasure_code.h"
#include "test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memset, memcmp
typedef unsigned char u8;
#include "PC_CPU_ID.c"

#ifdef __GNUC__

/// Macro to define a time value
#define ECCTIME struct timeval
/// Macro to get current time
#define ECCGETTIME(X) gettimeofday (&X, NULL)
/// Macro to compute elapsed time
#define ECCELAPSED(X, Y, Z) X = (((Z.tv_sec - Y.tv_sec) * 1000000LL + Z.tv_usec - Y.tv_usec) / 1000)
/// Macro to end thread
#define ECCENDTHREAD
/// Macro to hold thread info
#define ECCTHREAD pthread_t
/// Macto to start thread
#define ECCTHREADSTART(T, F, A) pthread_create (&T, NULL, (void *) F, (void *) &A)
/// Macro to wait for thread completion
#define ECCTHREADWAIT(T) pthread_join (T, NULL)

#else

// Same as above
#define ECCTIME                 DWORD
#define ECCGETTIME(X)           X = timeGetTime ()
#define ECCELAPSED(X, Y, Z)     X = (Z) - (Y)
#define ECCENDTHREAD            _endthread ()
#define ECCTHREAD               HANDLE
#define ECCTHREADSTART(T, F, A) T = (HANDLE) _beginthread (F, 0, &A)
#define ECCTHREADWAIT(T)        WaitForSingleObject (T, INFINITE)

#endif

#define PC_MAX_ERRS 32

// Utility print routine
void
dump_u8xu8 (unsigned char *s, int k, int m)
{
    int i, j;
    for (i = 0; i < k; i++)
    {
        for (j = 0; j < m; j++)
        {
            printf (" %3x", 0xff & s[ j + (i * m) ]);
        }
        printf ("\n");
    }
    printf ("\n");
}
extern int
ec_encode_data_avx512_gfni (int len, int m, int p, unsigned char *g_tbls, unsigned char **data,
                            unsigned char **coding);

#define NOPAPI 1

#ifdef _WIN64
#include <process.h>
#define __builtin_prefetch(a, b, c) _mm_prefetch ((const char *) (a), _MM_HINT_T0)
#define _popcnt64                   __popcnt64
#define NOPAPI                      1
#else
#include <sys/time.h>
#include <pthread.h>
#endif

#ifdef __aarch64__
#define NOPAPI 1
#include "aarch64/PCLib_AARCH64_NEON.c"
#include <arm_neon.h>
extern void
ec_encode_data_neon (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
extern void
ec_encode_data_neon (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
#else
#include <immintrin.h>
#include "PCLib_2D_AVX512_GFNI.c"
#endif

#define PC_MAX_CORES 32

#ifndef GT_L3_CACHE
#define GT_L3_CACHE 32 * 1024 * 1024 /* some number > last level cache */
#endif

#define COLD_TEST // Added for PolyCode Development

#if !defined(COLD_TEST) && !defined(TEST_CUSTOM)
// Cached test, loop many times over small dataset
#define TEST_SOURCES  32
#define TEST_LEN(m)   ((128 * 1024 / m) & ~(64 - 1))
#define TEST_TYPE_STR "_warm"
#elif defined(COLD_TEST)
// Uncached test.  Pull from large mem base.
#define TEST_SOURCES  255
#define TEST_LEN(m)   ((GT_L3_CACHE / m) & ~(64 - 1))
#define TEST_TYPE_STR "_cold"
#elif defined(TEST_CUSTOM)
#define TEST_TYPE_STR "_cus"
#endif
#ifndef TEST_SEED
#define TEST_SEED 0x1234
#endif

#define MMAX TEST_SOURCES
#define KMAX TEST_SOURCES // Maximum data buffer count, excluding parity

#define BAD_MATRIX -1

struct PCBenchStruct
{
    unsigned char **Data;
    unsigned char **Syn;
    unsigned char k;
    unsigned char p;
    unsigned char *g_tbls;
    int testNum;
    int testReps;
};

void
BenchWorker (void *t)
{
    struct PCBenchStruct *pcBench = (struct PCBenchStruct *) t;
   
    int m = pcBench->k + pcBench->p;

    for (int i = 0; i < pcBench->testReps; i++)
    {
        // Select the appropriate test
        switch (pcBench->testNum)
        {
        case 1:
            ec_encode_data_avx512_gfni (TEST_LEN (m), pcBench->k, pcBench->p, pcBench->g_tbls,
                                        pcBench->Data, pcBench->Syn);
            break;
        default:
            printf ("Error Test '%d' not valid\n", pcBench->testNum);
        }
    }

    ECCENDTHREAD;
}

void
usage (const char *app_name)
{
    fprintf (stderr,
             "Usage: %s [options]\n"
             "  -h        Help\n"
             "  -k <val>  Number of source buffers\n"
             "  -p <val>  Number of parity buffers\n"
             "  -c <val>  Number of cores\n",
             app_name);
}

// Create buffers and data structures for benchmark to run in another thread
int
InitClone (struct PCBenchStruct * ps, unsigned char k, unsigned char p, int testNum, int testReps)
{
    // Save parms for test in structure
    ps->k = k;
    ps->p = p;
    int m = k + p;
    ps->testNum = testNum;
    ps->testReps = testReps;

    // Now allocate data buffers (k+p)
    unsigned char **buffs, *buf;
    if (posix_memalign ((void *)&buffs, 64, sizeof ( unsigned char * ) * m ) )
    {
        printf ("Error allocating buffs\n");
        return 0;
    }
    memset (buffs, 0, sizeof (unsigned char *) * m);
    for (int i = 0; i < m; i++)
    {
        if (posix_memalign ((void *)&buf, 64, TEST_LEN (m)))
        {
            printf ("Error allocating buffers\n");
            return 0;
        }
        buffs[ i ] = buf;
    }
    ps->Data = buffs;

    if (posix_memalign ((void *)&buffs, 64, sizeof (unsigned char *) * p))
    {
        printf ("Error allocating Syns\n");
        return 0;
    }
    memset (buffs, 0, sizeof (unsigned char *) * p);

    for (int i = 0; i < p; i++)
    {
        if (posix_memalign ((void *)&buf, 64, TEST_LEN (m)))
        {
            printf ("Error allocating buffers\n");
            return 0;
        }
        buffs[ i ] = buf;
    }
    ps->Syn = buffs;
    
    if (posix_memalign ((void *)&ps->g_tbls, 64, 255*32*8))
    {
        printf ("Error allocating g_tbls\n");
        return 0;
    }
 
    return 1;
}

// Free data buffers for other thread benchmarks
void 
FreeClone (struct PCBenchStruct *ps, unsigned char k, unsigned char p) 
{
    int m = k + p;
    for (int i = 0; i < m; i++)
    {
        aligned_free (ps->Data[ i ]);
    }

    for (int i = 0; i < p; i++)
    {
        aligned_free (ps->Syn[ i ]);
    }
    aligned_free (ps->g_tbls);
    aligned_free (ps->Data);
    aligned_free (ps->Syn);
}

int 
main (int argc, char *argv[])
{
    // Work variables
    int i, j, m, k, p, nerrs, ret = -1, cores = 1;
    void *buf;
    u8 *a, *g_tbls = 0, *z0 = 0;
    u8 *temp_buffs[ TEST_SOURCES ] = { NULL };
    u8 *buffs[ TEST_SOURCES ] = { NULL };

    struct perf start;

    /* Set default parameters */
    k = 223;
    p = 32;
    nerrs = 32;

    /* Parse arguments */
    for (i = 1; i < argc; i++)
    {
        if (strcmp (argv[ i ], "-k") == 0)
        {
            k = atoi (argv[ ++i ]);
        }
        else if (strcmp (argv[ i ], "-p") == 0)
        {
            p = atoi (argv[ ++i ]);
        }
        else if (strcmp (argv[ i ], "-c") == 0)
        {
            cores = atoi (argv[ ++i ]);
        }
        else if (strcmp (argv[ i ], "-h") == 0)
        {
            usage (argv[ 0 ]);
            return 0;
        }
        else
        {
            usage (argv[ 0 ]);
            return -1;
        }
    }

    // Do a little paramater validation
    if (nerrs > k)
    {
        nerrs = k;
    }

    if (k <= 0)
    {
        printf ("Number of source buffers (%d) must be > 0\n", k);
        return -1;
    }

    if (p <= 0)
    {
        printf ("Number of parity buffers (%d) must be > 0\n", p);
        return -1;
    }

    if (cores <= 0)
    {
        cores = 1;
    }

    // Match errors to parity count and compute codeword size
    nerrs = p;
    m = k + p;

    if (m > MMAX)
    {
        printf ("Number of total buffers (data and parity) cannot be higher than %d\n", MMAX);
        return -1;
    }

    // Print output header
    PC_CPU_ID ();
    printf ("Testing with %u data buffers and %u parity buffers\n", k, p);
    printf ("erasure_code_perf: %dx%d %d\n", m, TEST_LEN (m), nerrs);

    // Build the power, log and inverse tables
    pc_bpow_2d (3);
    pc_blog_2d ();
    pc_binv_2d ();

    // Create memory for encoding matrices
    // a = malloc ( MMAX * ( KMAX*2 ) ) ;
    a = malloc (sizeof (Vand1b));
    if (a == NULL)
    {
        printf ("Error allocating a\n");
        goto exit;
    }

    // Initialize the Vandermonde matrix
    pc_gen_rsr_matrix_2d (a, 4);

    pc_bvan_2d (a, 4);

    // Initialize the encoding matrix
    pc_gen_poly_matrix_2d (a, 255, 255 - 4);
    pc_bmat_2d (a, 4);

    // Allocate the arrays
    if (posix_memalign (&buf, 64, TEST_LEN (m)))
    {
        printf ("Error allocating buffers\n");
        goto exit;
    }
    z0 = buf;
    memset (z0, 0, TEST_LEN (m));

    for (i = 0; i < m; i++)
    {
        if (posix_memalign (&buf, 64, TEST_LEN (m)))
        {
            printf ("Error allocating buffers\n");
            goto exit;
        }
        buffs[ i ] = buf;
    }

    for (i = 0; i < p; i++)
    {
        if (posix_memalign (&buf, 64, TEST_LEN (m)))
        {
            printf ("Error allocating buffers\n");
            goto exit;
        }
        temp_buffs[ i ] = buf;
    }

    // Allocate gtbls
    if (posix_memalign (&buf, 64, KMAX * TEST_SOURCES * 32))
    {
        printf ("Error allocating g_tbls\n");
        goto exit;
    }
    g_tbls = buf;

    // Make random data
    for (i = 0; i < k; i++)
        for (j = 0; j < TEST_LEN (m); j++)
            buffs[ i ][ j ] = rand ();
    // buffs [ k - 1 ] [ 59 ] = 1 ;;

    // Print test type
    printf ("Testing AVX512-GFNI\n");

    struct PCBenchStruct Bench[ PC_MAX_CORES ] = { 0 };

    // Initialize the clones
    for (i = 0; i < cores; i++)
    {
        if (InitClone(&Bench[i], k, p, 1, 200) == 0)
        {
            printf ("Initclone %d failed\n", i);
        }
    }

    ECCTIME startTime, endTime;
    double elapsedTime, mbPerSecond, totBytes;
    ECCTHREAD clone [ PC_MAX_CORES ];

    ECCGETTIME (startTime);
 
    // Start each benchmark thread
    for (i = 0; i < cores; i++)
    {
        ECCTHREADSTART (clone[ i ], BenchWorker, Bench[i]);
    }

    // Wait for each benchmark thread to complete
    for (i = 0; i < cores; i++)
    {
        ECCTHREADWAIT (clone[ i ]);
    }
 
    ECCGETTIME (endTime);

    ECCELAPSED (elapsedTime, startTime, endTime);

    //printf ("StartTime = %d Endtime = %d ElapsedTime = %.0f\n", startTime, endTime,
    //        elapsedTime);

    totBytes = TEST_LEN (m);
    totBytes = totBytes * 200 * m * cores;
    totBytes /= 1000000;

    if (elapsedTime > 0)
        mbPerSecond = ( totBytes / elapsedTime ) * 1000 ;
    else
        mbPerSecond = 0;

    printf ("erasure_code_encode_cold: k=%d p=%d bandwidth %.0f MB in %.3f sec = %.2f MB/s\n", 
        k, p, totBytes, elapsedTime/1000, mbPerSecond ) ;

    for (i = 0; i < cores; i++)
    {
        FreeClone (&Bench[ i ], k,p);
    }
    // Perform the baseline benchmark

    BENCHMARK (&start, BENCHMARK_TIME,
               ec_encode_data_avx512_gfni (TEST_LEN (m), k, p, g_tbls, buffs, &buffs[ k ]));

    printf ("erasure_code_encode" TEST_TYPE_STR ": k=%d p=%d ", k, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    unsigned char LFSRTab[ 32 ];
    // Test intrinsics lfsr
    pc_gen_poly_2d (LFSRTab, p);
    // dump_u8xu8 ( LFSRTab, 1, 4 ) ;
    BENCHMARK (&start, BENCHMARK_TIME,
               pc_encode_data_avx512_gfni_2d (TEST_LEN (m), k, p, LFSRTab, buffs, &buffs[ k ]));
    // pc_encode_data_avx512_gfni_2d(64, k, p, LFSRTab, buffs, &buffs [ k ]);

    printf ("polynomial_code_pls" TEST_TYPE_STR ": k=%d p=%d ", k, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    // for ( i = 0 ; i < m ; i ++ )
    //{
    //     dump_u8xu8 ( ( unsigned char * ) buffs [ i ], 1, 64 ) ;
    // }

    BENCHMARK (&start, BENCHMARK_TIME,
               ec_encode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs));

    printf ("dot_prod_decode" TEST_TYPE_STR ":     k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    // Now benchmark parallel syndrome sequencer - First create power vector
    unsigned char pwrTab[ 32 ];
    i = 3;
    for (j = p - 2; j >= 0; j--)
    {
        pwrTab[ j ] = i;
        i = pc_mul_2d (i, 3);
    }

    int done;
    // printf ( "Before Benchmark\n" ) ;
    BENCHMARK (&start, BENCHMARK_TIME,
               done = pc_decode_data_avx512_gfni_2d (TEST_LEN (m), m, p, pwrTab, buffs, temp_buffs,
                                                     1));
    //        done=pc_decode_data_avx512_gfni_2d(64, m, p, pwrTab, buffs, temp_buffs, 1);
    // printf ( "After benchmark\n" ) ;

    printf ("polynomial_code_pss" TEST_TYPE_STR ": k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    printf ("Length decoded = %x TEST_LEN(m) = %x\n", done, TEST_LEN (m));
    for (i = 0; i < p; i++)
    {
        if (0 != memcmp (z0, temp_buffs[ i ], TEST_LEN (m)))
        {
            printf ("Fail zero compare (%d, %d, %d, %d) - ", m, k, p, i);
            dump_u8xu8 (z0, 1, 16);
            dump_u8xu8 (temp_buffs[ i ], 1, 256);
            // goto exit;
        }
    }

    BENCHMARK (&start, BENCHMARK_TIME, PC_SingleEncoding (buffs, TEST_LEN (m), m));

    printf ("polynomial_code_sen" TEST_TYPE_STR ": k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    unsigned char syn[ 4 ];
    BENCHMARK (&start, BENCHMARK_TIME, done = PC_SingleDecoding (buffs, TEST_LEN (m), m, syn));

    printf ("polynomial_code_sde" TEST_TYPE_STR ": k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    BENCHMARK (&start, BENCHMARK_TIME, PC_SingleEncoding_u (buffs, TEST_LEN (m), m));

    printf ("polynomial_code_senu" TEST_TYPE_STR ": k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    BENCHMARK (&start, BENCHMARK_TIME, done = PC_SingleDecoding_u (buffs, TEST_LEN (m), m, syn));

    printf ("polynomial_code_sdeu" TEST_TYPE_STR ": k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    printf (" done all: Pass\n");
    fflush (stdout);

    ret = 0;
exit:
    aligned_free (z0);
    free (a);
    for (i = 0; i < TEST_SOURCES; i++)
    {
        aligned_free (buffs[ i ]);
        aligned_free (temp_buffs[ i ]);
    }
    aligned_free (g_tbls);
    return ret;
}
#else
#include <stdio.h>
int
main (void)
{
    printf ("No support for multi-level encoding on ARM64/NEON\n");
}
#endif
