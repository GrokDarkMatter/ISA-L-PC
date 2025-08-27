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

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memset, memcmp
#include "erasure_code.h"
#include "test.h"
#include "ec_base.h"

typedef unsigned char u8;

// Utility print routine
void
dump_u8xu8(unsigned char *s, int k, int m)
{
        int i, j;
        for (i = 0; i < k; i++) {
                for (j = 0; j < m; j++) {
                        printf(" %3x", 0xff & s[j + (i * m)]);
                }
                printf("\n");
        }
        printf("\n");
}

#ifdef __aarch64__
#include <arm_neon.h>
#include "aarch64/PCLib_AARCH64_NEON.c" 
extern void ec_encode_data_neon ( int len, int k, int p, u8 * g_tbls, u8 ** buffs, u8 ** dest ) ;
extern void ec_encode_data_neon ( int len, int k, int p, u8 * g_tbls, u8 ** buffs, u8 ** dest ) ;
#else
#include <immintrin.h>
#include "PCLib_AVX2_GFNI.c" 
#include "PCLib_AVX512_GFNI.c"
extern void ec_encode_data_avx512_gfni ( int len, int k, int p, u8 * g_tbls, u8 ** buffs, u8 ** dest ) ;
extern void ec_encode_data_avx2_gfni ( int len, int k, int p, u8 * g_tbls, u8 ** buffs, u8 ** dest ) ;
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

void
usage(const char *app_name)
{
        fprintf(stderr,
                "Usage: %s [options]\n"
                "  -h        Help\n"
                "  -k <val>  Number of source buffers\n"
                "  -p <val>  Number of parity buffers\n"
                "  -2 <val>  If 1 then AVX2 testing\n",
                app_name);
}

void
ec_encode_perf(int m, int k, u8 *a, u8 *g_tbls, u8 **buffs, struct perf *start)
{
        ec_init_tables(k, m - k, &a[k * k], g_tbls);
        BENCHMARK(start, BENCHMARK_TIME,
                  ec_encode_data(TEST_LEN(m), k, m - k, g_tbls, buffs, &buffs[k]));
}

int
ec_decode_perf(int m, int k, u8 *a, u8 *g_tbls, u8 **buffs, u8 *src_in_err, u8 *src_err_list,
               int nerrs, u8 **temp_buffs, struct perf *start)
{
        int i, j, r;
        u8 *b, *c, *d;
        u8 *recov[TEST_SOURCES];
        b = c = d = 0 ;

        // Allocate work buffers
        b=malloc(MMAX * KMAX) ;
        if ( b == NULL )
        {
                printf("Error allocating b\n") ;
                goto exit;
        }
        c=malloc(MMAX * KMAX) ;
        if ( c == NULL )
        {
                printf("Error allocating c\n") ;
                goto exit;
        }
        d=malloc(MMAX * KMAX) ;
        if ( d == NULL )
        {
                printf("Error allocating d\n") ;
                goto exit;
        }

        // Construct b by removing error rows
        for (i = 0, r = 0; i < k; i++, r++) {
                while (src_in_err[r])
                        r++;
                recov[i] = buffs[r];
                for (j = 0; j < k; j++)
                        b[k * i + j] = a[k * r + j];
        }

        if (gf_invert_matrix(b, d, k) < 0)
                return BAD_MATRIX;

        for (i = 0; i < nerrs; i++)
                for (j = 0; j < k; j++)
                        c[k * i + j] = d[k * src_err_list[i] + j];

        // Recover data
        ec_init_tables(k, nerrs, c, g_tbls);
        BENCHMARK(start, BENCHMARK_TIME,
                  ec_encode_data(TEST_LEN(m), k, nerrs, g_tbls, recov, temp_buffs));
exit:
        free(d);
        free(c);
        free(b);

        return 0;
}

int
main(int argc, char *argv[])
{
        // Work variables
        int i, j, m, k, p, nerrs, pp, ret = -1;
        void *buf ;
        u8 *a, *g_tbls=0, *z0=0 ;
        u8 *temp_buffs[TEST_SOURCES] = { NULL };
        u8 *buffs[TEST_SOURCES] = { NULL };
        struct perf start;
#ifndef __aarch64__
        u8 avx2=0;
#endif

        /* Set default parameters */
        k = 12;
        p = 8;
        nerrs = 4;
        pp = 1 ;

        /* Parse arguments */
        for (i = 1; i < argc; i++) {
                if (strcmp(argv[i], "-k") == 0) {
                        k = atoi(argv[++i]);
                } else if (strcmp(argv[i], "-p") == 0) {
                        p = atoi(argv[++i]);
#ifndef __aarch64__
                } else if (strcmp(argv[i], "-2") == 0) {
                        avx2 = atoi(argv[++i]);
#endif
                } else if (strcmp(argv[i], "-h") == 0) {
                        usage(argv[0]);
                        return 0;
                } else {
                        usage(argv[0]);
                        return -1;
                }
        }

        // Do a little paramater validation
        if ( pp >= ( k + p ) )
        {
                printf("Error location (%d) cannot be higher than number of symbols-1 (%d)\n",
                       pp, k+p-1 );
                return - 1;
        }

        if (nerrs > k) {
                nerrs = k ;
        }

        if (k <= 0) {
                printf("Number of source buffers (%d) must be > 0\n", k);
                return -1;
        }

        if (p <= 0) {
                printf("Number of parity buffers (%d) must be > 0\n", p);
                return -1;
        }

        if (nerrs <= 0) {
                printf("Number of errors (%d) must be > 0\n", nerrs);
                return -1;
        }

        // Match errors to parity count and compute codeword size
        nerrs = p ;
        m = k + p;

        if (m > MMAX) {
                printf("Number of total buffers (data and parity) cannot be higher than %d\n",
                       MMAX);
                return -1;
        }

        // Create memory for encoding matrices
        a = malloc ( MMAX * ( KMAX*2 ) ) ;
        if ( a == NULL )
        {
                printf("Error allocating a\n") ;
                goto exit;
        }

        // Print output header
        printf("Testing with %u data buffers and %u parity buffers\n", k, p ) ;
        printf("erasure_code_perf: %dx%d %d\n", m, TEST_LEN(m), nerrs);

        // Allocate the arrays
        if (posix_memalign(&buf, 64, TEST_LEN(m))) {
                printf("Error allocating buffers\n");
                goto exit;
        }
        z0 = buf;
        memset ( z0, 0, TEST_LEN(m)) ;

        for (i = 0; i < m; i++) {
                if (posix_memalign(&buf, 64, TEST_LEN(m))) {
                        printf("Error allocating buffers\n");
                        goto exit;
                }
                buffs[i] = buf;
        }

        for (i = 0; i < p; i++) {
                if (posix_memalign(&buf, 64, TEST_LEN(m))) {
                        printf("Error allocating buffers\n");
                        goto exit;
                }
                temp_buffs[i] = buf;
        }

        // Allocate gtbls
        if (posix_memalign(&buf, 64, KMAX * TEST_SOURCES * 32)) {
                printf("Error allocating g_tbls\n") ;
                goto exit;
        }
        g_tbls = buf ;

        // Make random data
        for (i = 0; i < k; i++)
                for (j = 0; j < TEST_LEN(m); j++)
                        buffs[i][j] = 0;
                        //buffs[i][j] = rand();

        memset ( buffs [ 0 ], 1, TEST_LEN(m) ) ;

        // Print test type
#ifdef __aarch64__
        printf ( "Testing ARM64-NEON\n" ) ;
#else
        if ( avx2 == 0 )
        {
                printf ( "Testing AVX512-GFNI\n" ) ;
        }
        else
        {
                printf ( "Testing AVX2-GFNI\n" ) ;
        }
#endif

        // Perform the baseline benchmark
        gf_gen_poly_matrix(a, m, k ) ;
        //dump_u8xu8 ( a, m, k ) ;
        ec_init_tables(k, m - k, &a[k * k], g_tbls);
#ifdef __aarch64__
        BENCHMARK(&start, BENCHMARK_TIME,
                  ec_encode_data_neon(TEST_LEN(m), k, m - k, g_tbls, buffs, &buffs[k]));
#else
        if ( avx2 == 0 )
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        ec_encode_data_avx512_gfni(TEST_LEN(m), k, m - k, g_tbls, buffs, &buffs[k]));
        }
        else
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        ec_encode_data_avx2_gfni(TEST_LEN(m), k, m - k, g_tbls, buffs, &buffs[k]));
        }
#endif
        printf("erasure_code_encode" TEST_TYPE_STR ": k=%d p=%d ", k, p);
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        // Test intrinsics lfsr
        gf_gen_poly ( a, p ) ;
        ec_init_tables ( p, 1, a, g_tbls ) ;
        //dump_u8xu8 ( g_tbls, p, 32 ) ;

#ifdef __aarch64__
        BENCHMARK(&start, BENCHMARK_TIME,
                  pc_encode_data_neon(TEST_LEN(m), k, m - k, g_tbls, buffs, temp_buffs));
#else
        if ( avx2 == 0 )
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        pc_encode_data_avx512_gfni(TEST_LEN(m), k, m - k, g_tbls, buffs, temp_buffs));
        }
        else
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        pc_encode_data_avx2_gfni(TEST_LEN(m), k, m - k, g_tbls, buffs, temp_buffs));
        }
#endif
        for (i = 0; i < p; i++) {
                if (0 != memcmp(buffs[k+i], temp_buffs[i], 64 ) ) {
                        printf("Fail parity compare (%d, %d, %d, %d) - ", m, k, p, i);
                        dump_u8xu8 ( buffs [ k+i ], 1, 16 ) ;
                        dump_u8xu8 ( temp_buffs [ i ], 1, 16 ) ;
                        goto exit;
                }
        }
        printf("polynomial_code_pls" TEST_TYPE_STR ": k=%d p=%d ", k, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        // Test decoding with dot product
        gf_gen_rsr_matrix( a, m+p, m ) ;
        ec_init_tables ( p, m, &a[ m*m ], g_tbls ) ;
#ifdef __aarch64__
        BENCHMARK(&start, BENCHMARK_TIME,
                  ec_encode_data_neon(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));
#else
        if ( avx2 == 0 )
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        ec_encode_data_avx512_gfni(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));
        }
        else
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        ec_encode_data_avx2_gfni(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));
        }
#endif
        printf("dot_prod_decode" TEST_TYPE_STR ":     k=%d p=%d ", m, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        // Test result of codeword encoding for zero
        for (i = 0; i < p; i++) {
                if (0 != memcmp(z0, temp_buffs[i], TEST_LEN(m) ) ) {
                        printf("Fail zero compare (%d, %d, %d, %d) - ", m, k, p, i);
                        dump_u8xu8 ( z0, 1, 16 ) ;
                        dump_u8xu8 ( temp_buffs [ i ], 1, 256 ) ;
                        goto exit;
                }
        }

        // Now test parallel syndrome sequencer
        // First create power vector
        i = 2 ;
        for ( j = p - 2 ; j >= 0 ; j -- )
        {
                a [ j ] = i ;
                i = gf_mul ( i, 2 ) ;
        }
        //printf ( "Vectors for pss\n" ) ;
        //dump_u8xu8 ( a, 1, p-1 ) ;
        ec_init_tables ( p - 1, 1, a, g_tbls ) ;
        //dump_u8xu8 ( g_tbls, p-1, 8 ) ;
        //buffs [ 0 ] [ 65 ] ^= 1 ;
#ifdef __aarch64__
        BENCHMARK(&start, BENCHMARK_TIME,
                  pc_decode_data_neon(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));
#else
        if ( avx2 == 0 )
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        pc_decode_data_avx512_gfni(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));
        }
        else
        {
                BENCHMARK(&start, BENCHMARK_TIME,
                        pc_decode_data_avx2_gfni(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));
        }
#endif
        printf("polynomial_code_pss" TEST_TYPE_STR ": k=%d p=%d ", m, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));                

        printf("done all: Pass\n");

        ret = 0;
exit:
        free ( z0 ) ;
        free ( a ) ;
        for (i = 0; i < TEST_SOURCES; i++) {
                aligned_free(buffs[i]);
                aligned_free(temp_buffs[i]);
        }
        aligned_free(g_tbls);
        return ret;
}
