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

extern void pc_encode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, 
        unsigned char **data, unsigned char **coding) ;

extern void pc_decode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, 
        unsigned char **data, unsigned char **coding, int offSet) ;


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

void
usage(const char *app_name)
{
        fprintf(stderr,
                "Usage: %s [options]\n"
                "  -h        Help\n"
                "  -k <val>  Number of source buffers\n"
                "  -p <val>  Number of parity buffers\n"
                "  -e <val>  Number of simulated buffers with errors (cannot be higher than p or "
                "k)\n"
                "  -pe <val> Error value for Polynomial Code decoding"
                "  -pp <val> Error position for Polynomial Code decoding",
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
        int i, j, m, k, p, nerrs, pp, ret = -1;
        void *buf;
        u8 *temp_buffs[TEST_SOURCES] = { NULL };
        u8 *buffs[TEST_SOURCES] = { NULL };
        u8 *a ;
        u8 *g_tbls=0, src_in_err[TEST_SOURCES];
        u8 src_err_list[TEST_SOURCES] ;
        struct perf start;

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
                } else if (strcmp(argv[i], "-e") == 0) {
                        nerrs = atoi(argv[++i]);
                } else if (strcmp(argv[i], "-pp") == 0) {
                        pp = atoi(argv[++i]);
                } else if (strcmp(argv[i], "-h") == 0) {
                        usage(argv[0]);
                        return 0;
                } else {
                        usage(argv[0]);
                        return -1;
                }
        }

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

        nerrs = p ;

        m = k + p;

        if (m > MMAX) {
                printf("Number of total buffers (data and parity) cannot be higher than %d\n",
                       MMAX);
                return -1;
        }

        u8 *err_list = malloc((size_t) nerrs);
        if (err_list == NULL) {
                printf("Error allocating list of array of error indices\n");
                return -1;
        }

        if ( nerrs > k )
        {
                printf ( "Number of errors (%d) must not be greater than k (%d)\n", nerrs, k ) ;
                return -1 ;
        }

        srand(TEST_SEED);

        for (i = 0; i < nerrs;) {
                u8 next_err = rand() % k;
                for (j = 0; j < i; j++)
                        if (next_err == err_list[j])
                                break;
                if (j != i)
                        continue;
                err_list[i++] = next_err;
        }

        a = malloc ( MMAX * ( KMAX*2 ) ) ;
        if ( a == NULL )
        {
                printf("Error allocating a\n") ;
                goto exit;
        }

        printf("Testing with %u data buffers and %u parity buffers (num errors = %u, in [ ", k, p,
               nerrs);
        for (i = 0; i < nerrs; i++)
                printf("%d ", (int) err_list[i]);

        printf("])\n");

        printf("erasure_code_perf: %dx%d %d\n", m, TEST_LEN(m), nerrs);

        memcpy(src_err_list, err_list, nerrs);
        memset(src_in_err, 0, TEST_SOURCES);
        for (i = 0; i < nerrs; i++)
                src_in_err[src_err_list[i]] = 1;

        // Allocate the arrays
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
                        buffs[i][j] = rand();


        gf_gen_poly_matrix(a, m, k ) ;
        //printf ( "Poly Matrix\n" ) ;
        //dump_u8xu8( a, m, k ) ;

        // Start encode test
        ec_encode_perf(m, k, a, g_tbls, buffs, &start);
        printf("erasure_code_encode" TEST_TYPE_STR ": k=%d p=%d ", k, p);
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        // Make random data
        for (i = 0; i < k+p; i++)
                for (j = 0; j < TEST_LEN(m); j++)
                        //buffs[i][j] = 0;
                        buffs[i][j] = rand();
        //memset ( buffs [ k - 1 ], 1, TEST_LEN(m) ) ;

        // First encode data with polynomial matrix
        gf_gen_poly_matrix ( a, m, k ) ;
        ec_init_tables ( k, p, &a[k*k], g_tbls ) ;
        ec_encode_data ( TEST_LEN(m), k, p, g_tbls, buffs, &buffs [ k ] ) ;

        // Test intrinsics lfsr
        gf_gen_poly ( a, p ) ;
        ec_init_tables ( p, 1, a, g_tbls ) ;

        BENCHMARK(&start, BENCHMARK_TIME,
                pc_encode_data_avx512_gfni( TEST_LEN(m), k, p, g_tbls, buffs, temp_buffs ) ) ;
        for (i = 0; i < nerrs; i++) {
                if (0 != memcmp(buffs[k+i], temp_buffs[i], TEST_LEN(m) ) ) {
                        printf("Fail error recovery1 (%d, %d, %d) - ", m, k, nerrs);
                        dump_u8xu8 ( buffs [ k ], 1, 16 ) ;
                        dump_u8xu8 ( temp_buffs [ 0 ], 1, 16 ) ;
                        goto exit;
                }
        }
        printf("polynomial_code_ls" TEST_TYPE_STR ":  k=%d p=%d ", k, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        // Test decoding with dot product
        BENCHMARK(&start, BENCHMARK_TIME,
                ec_encode_data( TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs ) ) ;

        printf("dot_prod_decode" TEST_TYPE_STR ":     k=%d p=%d ", m, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

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
        BENCHMARK(&start, BENCHMARK_TIME,
                pc_decode_data_avx512_gfni( TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs, 0 ) ) ;
        printf("polynomial_code_pss" TEST_TYPE_STR ": k=%d p=%d ", m, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));                

        printf("done all: Pass\n");

        ret = 0;
exit:
        free ( a ) ;
        free(err_list);
        for (i = 0; i < TEST_SOURCES; i++) {
                aligned_free(buffs[i]);
                aligned_free(temp_buffs[i]);
        }
        aligned_free(g_tbls);
        return ret;
}
