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

#include "ec_base.h"
#include "erasure_code.h"
#include "poly_code.h"
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
#define PC_MAXTEST 4
#define NOPAPI     1
#include <arm_neon.h>
#include "aarch64/PCLib_AARCH64_NEON.c"
extern void
ec_encode_data_neon (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
#else
#define PC_MAXTEST 6
extern int
ec_encode_data_avx512_gfni (int len, int m, int p, unsigned char *g_tbls, unsigned char **data,
                            unsigned char **coding);
#include <immintrin.h>
#include "PCLib_2D_AVX512_GFNI.c"
#include "PCLib_AVX512_GFNI.c"
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
    unsigned char *plyTab;
    unsigned char *pwrTab;
    unsigned char *plyTab2d;
    unsigned char *pwrTab2d;
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
#ifndef __aarch64__
        // Select the appropriate test
        switch (pcBench->testNum)
        {
        case 1:
            ec_encode_data_avx512_gfni (TEST_LEN (m), pcBench->k, pcBench->p, pcBench->g_tbls,
                                        pcBench->Data, &pcBench->Data[ pcBench->k ]);
            break;
        case 2:
            ec_encode_data_avx512_gfni (TEST_LEN (m), m, pcBench->p, pcBench->g_tbls, pcBench->Data,
                                        pcBench->Syn);
            break;
        case 3:
            pc_encode_data_avx512_gfni (TEST_LEN (m), pcBench->k, pcBench->p, pcBench->plyTab,
                                        pcBench->Data, &pcBench->Data[ pcBench->k ]);
            break;
        case 4:
            pc_decode_data_avx512_gfni (TEST_LEN (m), m, pcBench->p, pcBench->pwrTab, pcBench->Data,
                                        pcBench->Syn, 1);
            break;
        case 5:
            pc_encode_data_avx512_gfni_2d (TEST_LEN (m), pcBench->k, pcBench->p, pcBench->plyTab2d,
                                           pcBench->Data, &pcBench->Data[ pcBench->k ]);
            break;
        case 6:
            pc_decode_data_avx512_gfni_2d (TEST_LEN (m), m, pcBench->p, pcBench->pwrTab2d,
                                           pcBench->Data, pcBench->Syn, 1);
            break;

        default:
            printf ("Error Test '%d' not valid\n", pcBench->testNum);
        }
#else
        switch (pcBench->testNum)
        {
        case 1:
            ec_encode_data_neon (TEST_LEN (m), pcBench->k, pcBench->p, pcBench->g_tbls,
                                 pcBench->Data, &pcBench->Data[ pcBench->k ]);
            break;
        case 2:
            ec_encode_data_neon (TEST_LEN (m), m, pcBench->p, pcBench->g_tbls, pcBench->Data,
                                 &pcBench->Data[ pcBench->k ]);
            break;
        case 3:
            pc_encode_data_neon (TEST_LEN (m), pcBench->k, pcBench->p, pcBench->plyTab,
                                 pcBench->Data, &pcBench->Data[ pcBench->k ]);
            break;
        case 4:
            pc_decode_data_neon (TEST_LEN (m), m, pcBench->p, pcBench->pwrTab, pcBench->Data,
                                 pcBench->Syn, 1);
            break;
        default:
            printf ("Error Test '%d' not valid\n", pcBench->testNum);
        }
#endif
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
InitClone (struct PCBenchStruct *ps, unsigned char k, unsigned char p, int testNum, int testReps)
{
    // Save parms for test in structure
    ps->k = k;
    ps->p = p;
    int m = k + p;
    ps->testNum = testNum;
    ps->testReps = testReps;

    // Now allocate data buffers (k+p)
    unsigned char **buffs, *buf;
    if (posix_memalign ((void *) &buffs, 64, sizeof (unsigned char *) * m))
    {
        printf ("Error allocating buffs\n");
        return 0;
    }
    memset (buffs, 0, sizeof (unsigned char *) * m);
    for (int i = 0; i < m; i++)
    {
        if (posix_memalign ((void *) &buf, 64, TEST_LEN (m)))
        {
            printf ("Error allocating buffers\n");
            return 0;
        }
        memset (buf, i, TEST_LEN (m));
        buffs[ i ] = buf;
    }
    ps->Data = buffs;

    if (posix_memalign ((void *) &buffs, 64, sizeof (unsigned char *) * p))
    {
        printf ("Error allocating Syns\n");
        return 0;
    }
    memset (buffs, 0, sizeof (unsigned char *) * p);

    for (int i = 0; i < p; i++)
    {
        if (posix_memalign ((void *) &buf, PC_STRIDE, TEST_LEN (m)))
        {
            printf ("Error allocating buffers\n");
            return 0;
        }
        memset (buf, i, TEST_LEN (m));
        buffs[ i ] = buf;
    }
    ps->Syn = buffs;

    if (posix_memalign ((void *) &ps->g_tbls, 64, PC_FIELD_SIZE * PC_MAX_PAR * PC_MAX_TAB))
    {
        printf ("Error allocating g_tbls\n");
        return 0;
    }
    memset (ps->g_tbls, 1, PC_FIELD_SIZE * PC_MAX_PAR * PC_MAX_TAB);

    ps->plyTab = malloc (PC_FIELD_SIZE * PC_MAX_PAR);
    if (ps->plyTab == 0)
        return 0;

    ps->pwrTab = malloc (PC_FIELD_SIZE * PC_MAX_PAR);
    if (ps->pwrTab == 0)
        return 0;

    // Initialize tables for encoding
    unsigned char a[ PC_FIELD_SIZE ];
    gf_gen_poly (a, p);

    // Initialize decoding table
    unsigned char b[ PC_FIELD_SIZE ];
    int i = PC_GEN_x11d;
    for (int j = p - 2; j >= 0; j--)
    {
        b[ j ] = i;
        i = gf_mul (i, PC_GEN_x11d);
    }

    // printf ("poly and pwr tables p=%d\n", p);
    // dump_u8xu8 (a, 1, p);
    // dump_u8xu8 (b, 1, p-1);

    // Initialize the constants for either affine or ARM64 multiply
    ec_init_tables (p, 1, a, ps->plyTab);
    ec_init_tables (p - 1, 1, b, ps->pwrTab);

#ifndef __aarch64__
    // Create generator polynomial for LFSR
    ps->plyTab2d = malloc (PC_FIELD_SIZE);
    if (ps->plyTab2d == 0)
        return 0;

    ps->pwrTab2d = malloc (PC_FIELD_SIZE);
    if (ps->pwrTab2d == 0)
        return 0;

    // Create generator polynomial for LFSR encoder
    pc_gen_poly_2d (ps->plyTab2d, p);

    // Create decreasing power values for Syndrome deocder
    i = PC_GEN_x11b;
    for (int j = p - 2; j >= 0; j--)
    {
        ps->pwrTab2d[ j ] = i;
        i = pc_mul_2d (i, PC_GEN_x11b);
    }
#endif

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
    free (ps->plyTab);
    free (ps->pwrTab);
#ifndef __aarch64__
    free (ps->plyTab2d);
    free (ps->pwrTab2d);
#endif
}

int
main (int argc, char *argv[])
{
    // Work variables
    int i, m, k, p, ret = -1, cores;

    /* Set default parameters */
    k = 223;
    p = 32;

    cores = PC_CPU_ID ();

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

    // m is total size of codeword
    m = k + p;

    if (m > MMAX)
    {
        printf ("Number of total buffers (data and parity) cannot be higher than %d\n", MMAX);
        return -1;
    }

    // Print output header
    printf ("Testing with %u data buffers and %u parity buffers\n", k, p);
    printf ("erasure_code_perf: %dx%d %d\n", m, TEST_LEN (m), p);
#ifndef __aarch64__
    // Build the power, log and inverse tables
    pc_bpow_2d (3);
    pc_blog_2d ();
    pc_binv_2d ();
    unsigned char *a = malloc (sizeof (Vand1b));
#else
    // Create memory for encoding matrices

    unsigned char *a = malloc (MMAX * (KMAX * 2));
#endif
    if (a == NULL)
    {
        printf ("Error allocating a\n");
        goto exit;
    }
#ifndef __aarch64__
    // Initialize the Vandermonde matrix
    pc_gen_rsr_matrix_2d (a, 4);

    pc_bvan_2d (a, 4);

    // Initialize the encoding matrix
    pc_gen_poly_matrix_2d (a, 255, 255 - 4);
    pc_bmat_2d (a, 4);
#endif

#ifndef __aarch64__
    // Print test type
    printf ("Testing AVX512-GFNI\n");
#else
    printf ("Testing ARM64 NEON\n");
#endif
    struct PCBenchStruct Bench[ PC_MAX_CORES ] = { 0 };

    // Initialize the clones
    for (i = 0; i < cores; i++)
    {
#ifndef __aarch64__
        if (InitClone (&Bench[ i ], k, p, 1, PC_TEST_LOOPS) == 0)
#else
        if (InitClone (&Bench[ i ], k, p, 1, (PC_TEST_LOOPS / 10)) == 0)
#endif
        {
            printf ("Initclone %d failed\n", i);
        }
    }

    ECCTIME startTime, endTime;
    double elapsedTime, mbPerSecond, totBytes;
    ECCTHREAD clone[ PC_MAX_CORES ];

    for (int curCore = 1; curCore <= cores; curCore++)
    {
        printf ("Testing with %d of %d cores\n", curCore, cores);
        for (int curTest = 1; curTest <= PC_MAXTEST; curTest++)
        {
            for (int curBench = 0; curBench < cores; curBench++)
            {
                Bench[ curBench ].testNum = curTest;
            }
            ECCGETTIME (startTime);

            // Start each benchmark thread
            for (i = 0; i < curCore; i++)
            {
                ECCTHREADSTART (clone[ i ], BenchWorker, Bench[ i ]);
            }

            // Wait for each benchmark thread to complete
            for (i = 0; i < curCore; i++)
            {
                ECCTHREADWAIT (clone[ i ]);
            }

            ECCGETTIME (endTime);

            ECCELAPSED (elapsedTime, startTime, endTime);

            // printf ("StartTime = %d Endtime = %d ElapsedTime = %.0f\n", startTime, endTime,
            //         elapsedTime);

            totBytes = TEST_LEN (m);
            totBytes = totBytes * Bench[ 0 ].testReps * m * curCore;
            totBytes /= 1000000;

            if (elapsedTime > 0)
                mbPerSecond = (totBytes / elapsedTime) * 1000;
            else
                mbPerSecond = 0;

            switch (curTest)
            {
            case 1:
                printf ("erasure_code_encode_cold: cores = %d k=%d p=%d bandwidth %.0f MB in %.3f "
                        "sec = %.2f MB/s\n",
                        curCore, k, p, totBytes, elapsedTime / 1000, mbPerSecond);
                break;
            case 2:
                printf ("erasure_code_decode_cold: cores = %d k=%d p=%d bandwidth %.0f MB in %.3f "
                        "sec = %.2f MB/s\n",
                        curCore, k + p, p, totBytes, elapsedTime / 1000, mbPerSecond);
                break;
            case 3:
                printf ("polynomial_code_pls_cold: cores = %d k=%d p=%d bandwidth %.0f MB in %.3f "
                        "sec = %.2f MB/s\n",
                        curCore, k, p, totBytes, elapsedTime / 1000, mbPerSecond);
                break;
            case 4:
                printf ("polynomial_code_pss_cold: cores = %d k=%d p=%d bandwidth %.0f MB in %.3f "
                        "sec = %.2f MB/s\n",
                        curCore, k + p, p, totBytes, elapsedTime / 1000, mbPerSecond);
                break;
            case 5:
                printf ("polynomial_code_pls_2d  : cores = %d k=%d p=%d bandwidth %.0f MB in %.3f "
                        "sec = %.2f MB/s\n",
                        curCore, k, p, totBytes, elapsedTime / 1000, mbPerSecond);
                break;
            case 6:
                printf ("polynomial_code_pss_2d  : cores = %d k=%d p=%d bandwidth %.0f MB in %.3f "
                        "sec = %.2f MB/s\n",
                        curCore, k + p, p, totBytes, elapsedTime / 1000, mbPerSecond);
                break;
            }
        }
        if (elapsedTime > 2000) // If greater than 2 seconds
        {
            // printf ("Elapsed Time = %lf\n", elapsedTime);
            for (int curP = 0; curP < cores; curP++)
            {
                Bench[ curP ].testReps /= 2;
            }
        }
    }

    for (i = 0; i < cores; i++)
    {
        FreeClone (&Bench[ i ], k, p);
    }
    printf (" done all: Pass\n");
    fflush (stdout);

    ret = 0;

exit:
    free (a);

    return ret;
}
