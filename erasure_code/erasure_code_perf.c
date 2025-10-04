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
#include "test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memset, memcmp
typedef unsigned char u8;

// Utility print routine
void dump_u8xu8 (unsigned char *s, int k, int m)
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
#define __builtin_prefetch(a, b, c) _mm_prefetch ((const char *) (a), _MM_HINT_T0)
#define _popcnt64                   __popcnt64
#define NOPAPI                      1
#endif
#ifdef __aarch64__
#define NOPAPI 1
#include <arm_neon.h>
#include "aarch64/PCLib_AARCH64_NEON.c"
extern void ec_encode_data_neon (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
extern void ec_encode_data_neon (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
#else
#include <immintrin.h>
extern void ec_encode_data_avx2_gfni (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
#include "PCLib_AVX2_GFNI.c"
extern void ec_encode_data_avx512_gfni (int len, int k, int p, u8 *g_tbls, u8 **buffs, u8 **dest);
#include "PCLib_AVX512_GFNI.c"
extern void gf_gen_poly (unsigned char *, int);
extern int find_roots (unsigned char *S, unsigned char *roots, int lenPoly);
extern int gf_invert_matrix (unsigned char *in_mat, unsigned char *out_mat, int size);
extern int berlekamp_massey (unsigned char *S, int length, unsigned char *keyEq);
extern int PGZ (unsigned char *S, int length, unsigned char *keyEq);
#endif

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

#ifndef NOPAPI
#include <papi.h>
void handle_error (int code)
{
    fprintf (stderr, "PAPI error: %s\n", PAPI_strerror (code));
    exit (1);
}

int InitPAPI (void)
{
    int event_set = PAPI_NULL, event_code, ret;

    // Initialize PAPI
    if ((ret = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
    {
        printf ("init fail\n");
        handle_error (ret);
    }

    // Create event set
    if ((ret = PAPI_create_eventset (&event_set)) != PAPI_OK)
    {
        printf ("create set failed\n");
        handle_error (ret);
    }

    // Add native event
    if ((ret = PAPI_event_name_to_code ("perf::CPU-CYCLES", &event_code)) != PAPI_OK)
    {
        handle_error (ret);
    }
    if ((ret = PAPI_add_event (event_set, event_code)) != PAPI_OK)
    {
        handle_error (ret);
    }

    // Try perf::INSTRUCTIONS
    if ((ret = PAPI_event_name_to_code ("perf::INSTRUCTIONS", &event_code)) != PAPI_OK)
    {
        handle_error (ret);
    }

    if ((ret = PAPI_add_event (event_set, event_code)) != PAPI_OK)
    {
        handle_error (ret);
    }
    return event_set;
}

void TestPAPIRoots (void)
{
    int event_set = PAPI_NULL, ret;
    long long values[ 2 ];
    double CPI;

    event_set = InitPAPI ();

    if (event_set == PAPI_NULL)
    {
        printf ("PAPI failed to initialize\n");
        exit (1);
    }
    unsigned char roots[ 16 ];

    for (int lenPoly = 2; lenPoly <= 16; lenPoly++)
    {
        int rootCount = 0;
        unsigned char S[ 16 ], keyEq[ 16 ];
        gf_gen_poly (S, lenPoly);
        // printf ( "Generator poly\n" ) ;
        // dump_u8xu8 ( S, 1, lenPoly ) ;

        for (int i = 0; i < lenPoly; i++)
        {
            keyEq[ i ] = S[ lenPoly - i - 1 ];
        }

        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // Workload
        rootCount = find_roots (keyEq, roots, lenPoly);

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // printf ( "Rootcount = %d\n", rootCount ) ;
        // dump_u8xu8 ( roots, 1, rootCount ) ;

        double baseVal = values[ 0 ];
        CPI = (double) values[ 0 ] / values[ 1 ];
        printf ("find_roots_sca %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                values[ 0 ], values[ 1 ], CPI);

        int rootCount2 =
                find_roots_AVX512_GFNI (keyEq, roots, lenPoly); // Run once to fill in Vandermonde
        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // Workload
        rootCount = find_roots_AVX512_GFNI (keyEq, roots, lenPoly);

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        if (rootCount != rootCount2)
        {
            printf ("Rootcount doesn't match %d %d\n", rootCount, rootCount2);
        }
        // printf ( "Rootcount2 = %d\n", rootCount ) ;
        // dump_u8xu8 ( roots, 1, rootCount ) ;

        double vecVal = values[ 0 ];
        double Speedup = baseVal / vecVal;
        CPI = (double) values[ 0 ] / values[ 1 ];
        printf ("find_roots_vec %2d %11lld cycles %11lld instructions CPI %.3lf Speedup = %.3lf\n",
                lenPoly, values[ 0 ], values[ 1 ], CPI, Speedup);
    }
}
void TestPAPIInv (void)
{

    int event_set = PAPI_NULL, ret;
    long long values[ 2 ];
    double CPI;
    event_set = InitPAPI ();

    if (event_set == PAPI_NULL)
    {
        printf ("PAPI failed to initialize\n");
        exit (1);
    }

    for (int lenPoly = 4; lenPoly <= 32; lenPoly++)
    {
        unsigned char in_mat[ 32 * 32 ], out_mat[ 32 * 32 ], base = 1, val = 1;

        for (int i = 0; i < lenPoly; i++)
        {
            for (int j = 0; j < lenPoly; j++)
            {
                in_mat[ i * lenPoly + j ] = val;
                val = gf_mul (val, base);
            }
            base = gf_mul (base, 2);
        }

        // printf ( "Vandermonde\n" ) ;
        // dump_u8xu8 ( in_mat, lenPoly, lenPoly ) ;
        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // Workload
        ret = gf_invert_matrix_AVX512_GFNI (in_mat, out_mat, lenPoly);

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // printf ( "Outmat\n" ) ;
        // dump_u8xu8 ( out_mat, lenPoly, lenPoly ) ;

        CPI = (double) values[ 0 ] / values[ 1 ];
        double vecVal = values[ 0 ];
        printf ("invert_matrix_vec %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                values[ 0 ], values[ 1 ], CPI);

        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        gf_invert_matrix (in_mat, out_mat, lenPoly);

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // printf ( "Outmat\n" ) ;
        // dump_u8xu8 ( out_mat, lenPoly, lenPoly ) ;

        CPI = (double) values[ 0 ] / values[ 1 ];
        double baseVal = values[ 0 ];
        double Speedup = baseVal / vecVal;
        printf ("invert_matrix_sca %2d %11lld cycles %11lld instructions CPI %.3lf Speedup = "
                "%.3lf\n",
                lenPoly, values[ 0 ], values[ 1 ], CPI, Speedup);
    }
}

void TestPAPIbm (void)
{

    int event_set = PAPI_NULL, ret, len;
    long long values[ 2 ];
    double CPI;
    event_set = InitPAPI ();

    if (event_set == PAPI_NULL)
    {
        printf ("PAPI failed to initialize\n");
        exit (1);
    }

    for (int lenPoly = 4; lenPoly <= 32; lenPoly += 2)
    {
        unsigned char S[ 32 ], keyEq[ 16 ];

        unsigned char base = 1;
        for (int i = 0; i < lenPoly; i++)
        {
            // int rvs = lenPoly - i - 1 ;
            int rvs = i;
            S[ rvs ] = 0;
            unsigned char val = 1;
            for (int j = 0; j < lenPoly / 2; j++)
            {
                S[ rvs ] ^= val;
                val = gf_mul (val, base);
            }
            base = gf_mul (base, 2);
        }

        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // Workload
        for (int i = 0; i < 1000; i++)
        {
            len = PGZ (S, lenPoly, keyEq);
        }

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        CPI = (double) values[ 0 ] / values[ 1 ];
        printf ("PGZ =  %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                values[ 0 ] / 1000, values[ 1 ] / 1000, CPI);

        // Now test Berlekamp
        unsigned char bmKeyEq[ 17 ];
        int bmLen;
        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // Workload
        for (int i = 0; i < 1000; i++)
        {
            bmLen = berlekamp_massey (S, lenPoly, bmKeyEq);
        }

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        unsigned char bmKeyEqRev[ 17 ];
        for (int curKey = 0; curKey < len; curKey++)
        {
            bmKeyEqRev[ curKey ] = bmKeyEq[ len - curKey ];
        }
        if ((memcmp (bmKeyEqRev, keyEq, len) != 0) || (bmLen != len))
        {
            printf ("Mismatch %d terms\n", len);
            dump_u8xu8 (keyEq, 1, len);
            dump_u8xu8 (bmKeyEqRev, 1, len);
            exit (1);
        }

        // printf ( "S and Lambda lenPoly = %d len = %d\n", lenPoly, len ) ;
        // dump_u8xu8 ( S, 1, lenPoly ) ;
        // dump_u8xu8 ( keyEq, 1, len+1 ) ;

        CPI = (double) values[ 0 ] / values[ 1 ];
        printf ("BM_sca %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                values[ 0 ] / 1000, values[ 1 ] / 1000, CPI);

        if ((ret = PAPI_start (event_set)) != PAPI_OK)
        {
            handle_error (ret);
        }

        // Workload
        for (int i = 0; i < 1000; i++)
        {
            bmLen = berlekamp_massey_AVX512_GFNI (S, lenPoly, bmKeyEq);
        }

        if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
        {
            handle_error (ret);
        }

        for (int curKey = 0; curKey < len; curKey++)
        {
            bmKeyEqRev[ curKey ] = bmKeyEq[ len - curKey ];
        }
        if ((memcmp (bmKeyEqRev, keyEq, len) != 0) || (bmLen != len))
        {
            printf ("Mismatch %d terms\n", len);
            dump_u8xu8 (keyEq, 1, len);
            dump_u8xu8 (bmKeyEqRev, 1, len);
            exit (1);
        }

        CPI = (double) values[ 0 ] / values[ 1 ];
        printf ("BM_vec %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                values[ 0 ] / 1000, values[ 1 ] / 1000, CPI);
    }
}

#endif

void usage (const char *app_name)
{
    fprintf (stderr,
             "Usage: %s [options]\n"
             "  -h        Help\n"
             "  -k <val>  Number of source buffers\n"
             "  -p <val>  Number of parity buffers\n"
             "  -2 <val>  If 1 then AVX2 testing\n",
             app_name);
}

void ec_encode_perf (int m, int k, u8 *a, u8 *g_tbls, u8 **buffs, struct perf *start)
{
    ec_init_tables (k, m - k, &a[ k * k ], g_tbls);
    BENCHMARK (start, BENCHMARK_TIME,
               ec_encode_data (TEST_LEN (m), k, m - k, g_tbls, buffs, &buffs[ k ]));
}

int ec_decode_perf (int m, int k, u8 *a, u8 *g_tbls, u8 **buffs, u8 *src_in_err, u8 *src_err_list,
                    int nerrs, u8 **temp_buffs, struct perf *start)
{
    int i, j, r;
    u8 *b, *c, *d;
    u8 *recov[ TEST_SOURCES ];
    b = c = d = 0;

    // Allocate work buffers
    b = malloc (MMAX * KMAX);
    if (b == NULL)
    {
        printf ("Error allocating b\n");
        goto exit;
    }
    c = malloc (MMAX * KMAX);
    if (c == NULL)
    {
        printf ("Error allocating c\n");
        goto exit;
    }
    d = malloc (MMAX * KMAX);
    if (d == NULL)
    {
        printf ("Error allocating d\n");
        goto exit;
    }

    // Construct b by removing error rows
    for (i = 0, r = 0; i < k; i++, r++)
    {
        while (src_in_err[ r ])
            r++;
        recov[ i ] = buffs[ r ];
        for (j = 0; j < k; j++)
            b[ k * i + j ] = a[ k * r + j ];
    }

    if (gf_invert_matrix (b, d, k) < 0)
        return BAD_MATRIX;

    for (i = 0; i < nerrs; i++)
        for (j = 0; j < k; j++)
            c[ k * i + j ] = d[ k * src_err_list[ i ] + j ];

    // Recover data
    ec_init_tables (k, nerrs, c, g_tbls);
    BENCHMARK (start, BENCHMARK_TIME,
               ec_encode_data (TEST_LEN (m), k, nerrs, g_tbls, recov, temp_buffs));
exit:
    free (d);
    free (c);
    free (b);

    return 0;
}

#define FIELD_SIZE 256

void inject_errors_in_place (unsigned char **data, int index, int num_errors,
                             unsigned char *error_positions, uint8_t *original_values)
{
    for (int i = 0; i < num_errors; i++)
    {
        int pos = error_positions[ i ];
        original_values[ i ] = data[ pos ][ index ];
        uint8_t error = (rand () % (FIELD_SIZE - 1)) + 1;
        data[ pos ][ index ] = data[ pos ][ index ] ^ error;
    }
}

int verify_correction_in_place (unsigned char **data, int index, int num_errors,
                                unsigned char *error_positions, uint8_t *original_values)
{
    for (int i = 0; i < num_errors; i++)
    {
        if (data[ error_positions[ i ] ][ index ] != original_values[ i ])
        {
            printf ("Error data= %d orig = %d\n", data[ error_positions[ i ] ][ index ],
                    original_values[ i ]);
            return 0;
        }
    }
    return 1;
}

int test_pgz_decoder (int index, int m, int p, unsigned char *g_tbls, unsigned char **data,
                      unsigned char **coding, int avx2)
{
    int successes = 0, total_tests = 0;

    for (int num_errors = 1; num_errors <= (p / 2); num_errors++)
    {
        for (int start = 0; start < m - (p / 2); start++)
        {
            unsigned char error_positions[ 16 ];
            uint8_t original_values[ 16 ];
            for (int i = 0; i < (p / 2); i++)
            {
                error_positions[ i ] = start + i;
            }
            inject_errors_in_place (data, index, num_errors, error_positions, original_values);
#ifdef __aarch64__
            pc_decode_data_neon (TEST_LEN (m), m, p, g_tbls, data, coding, 1);
#else
            if (avx2 == 0)
            {
                pc_decode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, data, coding, 1);
            }
            else
            {
                pc_decode_data_avx2_gfni (TEST_LEN (m), m, p, g_tbls, data, coding, 1);
            }
#endif

            if (verify_correction_in_place (data, index, num_errors, error_positions,
                                            original_values))
            {
                successes++;
            }
            else
            {
                printf ("Failed: Sequential, %d errors at %d\n", num_errors, start);
                return 0;
            }
            total_tests++;
        }
    }

    for (int num_errors = 1; num_errors <= (p / 2); num_errors++)
    {
        for (int trial = 0; trial < 1000; trial++)
        {
            uint8_t error_positions[ 16 ];
            uint8_t original_values[ 16 ];
            int available[ FIELD_SIZE ];
            for (int i = 0; i < m; i++)
            {
                available[ i ] = i;
            }
            for (int i = 0; i < num_errors; i++)
            {
                int idx = rand () % (m - i);
                error_positions[ i ] = available[ idx ];
                available[ idx ] = available[ m - 1 - i ];
            }
            inject_errors_in_place (data, index, num_errors, error_positions, original_values);

#ifdef __aarch64__
            pc_decode_data_neon (TEST_LEN (m), m, p, g_tbls, data, coding, 1);
#else
            if (avx2 == 0)
            {
                pc_decode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, data, coding, 1);
            }
            else
            {
                pc_decode_data_avx2_gfni (TEST_LEN (m), m, p, g_tbls, data, coding, 1);
            }
#endif

            if (verify_correction_in_place (data, index, num_errors, error_positions,
                                            original_values))
            {
                successes++;
            }
            else
            {
                printf ("Failed: Random, %d errors, trial %d\n", num_errors, trial);
                return 0;
            }
            total_tests++;
        }
    }

    printf ("Tests completed\n");
    return 1;
}

int main (int argc, char *argv[])
{
    // Work variables
    int i, j, m, k, p, nerrs, ret = -1;
    void *buf;
    u8 *a, *g_tbls = 0, *z0 = 0;
    u8 *temp_buffs[ TEST_SOURCES ] = { NULL };
    u8 *buffs[ TEST_SOURCES ] = { NULL };

    struct perf start;

    u8 avx2 = 0;

    /* Set default parameters */
    k = 12;
    p = 8;
    nerrs = 4;

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
#ifndef __aarch64__
        }
        else if (strcmp (argv[ i ], "-2") == 0)
        {
            avx2 = atoi (argv[ ++i ]);
#endif
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

    // Match errors to parity count and compute codeword size
    nerrs = p;
    m = k + p;

#ifndef NOPAPI
    // Do early performance testing
    if (avx2 == 0)
    {
        TestPAPIRoots ();
        TestPAPIInv ();
        TestPAPIbm ();
    }
#endif
    if (m > MMAX)
    {
        printf ("Number of total buffers (data and parity) cannot be higher than %d\n", MMAX);
        return -1;
    }

    // Create memory for encoding matrices
    a = malloc (MMAX * (KMAX * 2));
    if (a == NULL)
    {
        printf ("Error allocating a\n");
        goto exit;
    }
    // Print output header
    printf ("Testing with %u data buffers and %u parity buffers\n", k, p);
    printf ("erasure_code_perf: %dx%d %d\n", m, TEST_LEN (m), nerrs);

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

            // Print test type
#ifdef __aarch64__
    printf ("Testing ARM64-NEON\n");
#else
    if (avx2 == 0)
    {
        printf ("Testing AVX512-GFNI\n");
    }
    else
    {
        printf ("Testing AVX2-GFNI\n");
    }
#endif

    // Perform the baseline benchmark
    gf_gen_poly_matrix (a, m, k);
    ec_init_tables (k, m - k, &a[ k * k ], g_tbls);

#ifdef __aarch64__
    BENCHMARK (&start, BENCHMARK_TIME,
               ec_encode_data_neon (TEST_LEN (m), k, p, g_tbls, buffs, &buffs[ k ]));
#else
    if (avx2 == 0)
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   ec_encode_data_avx512_gfni (TEST_LEN (m), k, p, g_tbls, buffs, &buffs[ k ]));
    }
    else
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   ec_encode_data_avx2_gfni (TEST_LEN (m), k, p, g_tbls, buffs, &buffs[ k ]));
    }
#endif
    printf ("erasure_code_encode" TEST_TYPE_STR ": k=%d p=%d ", k, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    // Test intrinsics lfsr
    gf_gen_poly (a, p);
    ec_init_tables (p, 1, a, g_tbls);

#ifdef __aarch64__
    BENCHMARK (&start, BENCHMARK_TIME,
               pc_encode_data_neon (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs));
#else
    if (avx2 == 0)
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   pc_encode_data_avx512_gfni (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs));
    }
    else
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   pc_encode_data_avx2_gfni (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs));
    }
#endif
    for (i = 0; i < p; i++)
    {
        if (0 != memcmp (buffs[ k + i ], temp_buffs[ i ], TEST_LEN (m)))
        {
            printf ("Fail parity compare (%d, %d, %d, %d) - ", m, k, p, i);
            dump_u8xu8 (buffs[ k + i ], 1, 16);
            dump_u8xu8 (temp_buffs[ i ], 1, 16);
            goto exit;
        }
    }
    printf ("polynomial_code_pls" TEST_TYPE_STR ": k=%d p=%d ", k, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    // Test decoding with dot product
    gf_gen_rsr_matrix (a, m + p, m);
    ec_init_tables (p, m, &a[ m * m ], g_tbls);
#ifdef __aarch64__
    BENCHMARK (&start, BENCHMARK_TIME,
               ec_encode_data_neon (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs));
#else
    if (avx2 == 0)
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   ec_encode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs));
    }
    else
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   ec_encode_data_avx2_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs));
    }
#endif
    printf ("dot_prod_decode" TEST_TYPE_STR ":     k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    // Test result of codeword encoding for zero
    for (i = 0; i < p; i++)
    {
        if (0 != memcmp (z0, temp_buffs[ i ], TEST_LEN (m)))
        {
            printf ("Fail zero compare (%d, %d, %d, %d) - ", m, k, p, i);
            dump_u8xu8 (z0, 1, 16);
            dump_u8xu8 (temp_buffs[ i ], 1, 256);
            goto exit;
        }
    }

    // Now benchmark parallel syndrome sequencer - First create power vector
    i = 2;
    for (j = p - 2; j >= 0; j--)
    {
        a[ j ] = i;
        i = gf_mul (i, 2);
    }

    ec_init_tables (p - 1, 1, a, g_tbls);
#ifdef __aarch64__
    BENCHMARK (&start, BENCHMARK_TIME,
               pc_decode_data_neon (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs, 1));
#else
    if (avx2 == 0)
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   pc_decode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs, 1));
    }
    else
    {
        BENCHMARK (&start, BENCHMARK_TIME,
                   pc_decode_data_avx2_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs, 1));
    }
#endif
    printf ("polynomial_code_pss" TEST_TYPE_STR ": k=%d p=%d ", m, p);
    perf_print (start, (long long) (TEST_LEN (m)) * (m));

    if (test_pgz_decoder (0, m, p, g_tbls, buffs, temp_buffs, avx2) == 0)
    {
        printf ("Decoder failed\n");
        goto exit;
    }
#ifndef NOPAPI
    int event_set = InitPAPI (); // PAPI_NULL, event_code ;
    if ((ret = PAPI_start (event_set)) != PAPI_OK)
    {
        handle_error (ret);
    }
    // Workload
    if (avx2 == 0)
    {
        ec_encode_data_avx512_gfni (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs);
    }
    else
    {
        ec_encode_data_avx2_gfni (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs);
    }

    long long values[ 2 ];
    if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
    {
        handle_error (ret);
    }

    double CPI;
    CPI = (double) values[ 0 ] / values[ 1 ];
    double BPC;
    BPC = (double) (TEST_LEN (m) * m) / values[ 0 ];

    printf ("EC_Encode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[ 0 ],
            values[ 1 ], CPI, BPC);

    if ((ret = PAPI_start (event_set)) != PAPI_OK)
    {
        handle_error (ret);
    }
    // Workload
    if (avx2 == 0)
    {
        pc_encode_data_avx512_gfni (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs);
    }
    else
    {
        pc_encode_data_avx2_gfni (TEST_LEN (m), k, p, g_tbls, buffs, temp_buffs);
    }

    if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
    {
        handle_error (ret);
    }

    CPI = (double) values[ 0 ] / values[ 1 ];
    BPC = (double) (TEST_LEN (m) * m) / values[ 0 ];

    printf ("PC_Encode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[ 0 ],
            values[ 1 ], CPI, BPC);

    if ((ret = PAPI_start (event_set)) != PAPI_OK)
    {
        handle_error (ret);
    }

    // Workload
    if (avx2 == 0)
    {
        ec_encode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs);
    }
    else
    {
        ec_encode_data_avx2_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs);
    }

    if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
    {
        handle_error (ret);
    }

    CPI = (double) values[ 0 ] / values[ 1 ];
    BPC = (double) (TEST_LEN (m) * m) / values[ 0 ];

    printf ("EC_Decode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[ 0 ],
            values[ 1 ], CPI, BPC);

    if ((ret = PAPI_start (event_set)) != PAPI_OK)
    {
        handle_error (ret);
    }

    // Workload
    if (avx2 == 0)
    {
        pc_decode_data_avx512_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs, 1);
    }
    else
    {
        pc_decode_data_avx2_gfni (TEST_LEN (m), m, p, g_tbls, buffs, temp_buffs, 1);
    }

    if ((ret = PAPI_stop (event_set, values)) != PAPI_OK)
    {
        handle_error (ret);
    }

    CPI = (double) values[ 0 ] / values[ 1 ];
    BPC = (double) (TEST_LEN (m) * m) / values[ 0 ];

    printf ("PC_Decode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[ 0 ],
            values[ 1 ], CPI, BPC);

    PAPI_cleanup_eventset (event_set);
    PAPI_destroy_eventset (&event_set);
    PAPI_shutdown ();
#endif
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
