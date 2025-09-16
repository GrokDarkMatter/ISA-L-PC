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

#define PC_MAX_ERRS 32

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
extern int ec_encode_data_avx512_gfni ( int len, int m, int p, unsigned char * g_tbls, 
    unsigned char ** data, unsigned char ** coding ) ;

#ifdef _WIN64
#define __builtin_prefetch(a,b,c) _mm_prefetch((const char*)(a), _MM_HINT_T0)
#define _popcnt64 __popcnt64
#define NOPAPI 1
#endif
#ifdef __aarch64__
#define NOPAPI 1
#include <arm_neon.h>
#include "aarch64/PCLib_AARCH64_NEON.c" 
extern void ec_encode_data_neon ( int len, int k, int p, u8 * g_tbls, u8 ** buffs, u8 ** dest ) ;
extern void ec_encode_data_neon ( int len, int k, int p, u8 * g_tbls, u8 ** buffs, u8 ** dest ) ;
#else
#include <immintrin.h>
#include "PCLib_1B_AVX512_GFNI.c"
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
void handle_error(int code)
{
    fprintf ( stderr, "PAPI error: %s\n", PAPI_strerror ( code ) ) ;
    exit ( 1 ) ;
}

int InitPAPI ( void )
{
        int event_set = PAPI_NULL, event_code, ret ;

        // Initialize PAPI
        if ((ret = PAPI_library_init ( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT ) 
        {
                printf ( "init fail\n" ) ;
                handle_error ( ret ) ;
        }

        // Create event set
        if ( ( ret = PAPI_create_eventset ( &event_set ) ) != PAPI_OK ) 
        {
                printf ( "create set failed\n" ) ;
                handle_error ( ret ) ;
        }

        // Add native event
        if ( ( ret = PAPI_event_name_to_code ( "perf::CPU-CYCLES", &event_code ) ) != PAPI_OK )
        {
                handle_error ( ret ) ;
        }
        if ( ( ret = PAPI_add_event ( event_set, event_code ) ) != PAPI_OK ) 
        {
                handle_error ( ret ) ;
        }

        // Try perf::INSTRUCTIONS
        if ( ( ret = PAPI_event_name_to_code ( "perf::INSTRUCTIONS", &event_code ) ) != PAPI_OK )
        {
                handle_error ( ret ) ;
        }

        if ( ( ret = PAPI_add_event ( event_set, event_code ) ) != PAPI_OK ) 
        {
                handle_error ( ret ) ;
        }
        return event_set ;
}

void TestPAPIRoots ( void )
{
        int event_set = PAPI_NULL, ret ;
        long long values [ 2 ] ;
        double CPI;

        event_set = InitPAPI() ;

        if ( event_set == PAPI_NULL )
        {
                printf ( "PAPI failed to initialize\n" ) ;
                exit ( 1 ) ;
        }
        unsigned char roots [ 16 ] ;

        for ( int lenPoly = 2 ; lenPoly <= 16 ; lenPoly ++ )
        {
                int rootCount = 0 ;
                unsigned char S [ 16 ], keyEq [ 16 ] ;
                pc_gen_poly_1b ( S, lenPoly ) ;
                //printf ( "Generator poly\n" ) ;
                //dump_u8xu8 ( S, 1, lenPoly ) ;

                for ( int i = 0 ; i < lenPoly ; i ++ )
                {
                        keyEq [ i ] = S [ lenPoly - i - 1 ] ;
                }

                if ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK ) 
                {
                        handle_error(ret);
                }

                // Workload
                rootCount = find_roots_1b ( keyEq, roots, lenPoly ) ;

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
                {
                        handle_error ( ret ) ;
                }

                //printf ( "Rootcount = %d\n", rootCount ) ;
                //dump_u8xu8 ( roots, 1, rootCount ) ;

                double baseVal = values [ 0 ] ;
                CPI = ( double ) values[ 0 ] / values[ 1 ] ;
                printf ( "find_roots_sca %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly, values [ 0 ], values [ 1 ], CPI ) ;

                int rootCount2 = find_roots_1b_AVX512_GFNI ( keyEq, roots, lenPoly ) ; // Run once to fill in Vandermonde
                if ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK) 
                {
                        handle_error(ret);
                }

                // Workload
                rootCount = find_roots_1b_AVX512_GFNI ( keyEq, roots, lenPoly ) ;

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
                {
                        handle_error(ret);
                }

                if ( rootCount != rootCount2 )
                {
                        printf ( "Rootcount doesn't match %d %d\n", rootCount, rootCount2 ) ;
                }
                //printf ( "Rootcount2 = %d\n", rootCount ) ;
                //dump_u8xu8 ( roots, 1, rootCount ) ;

                double vecVal = values [ 0 ] ;
                double Speedup = baseVal / vecVal ;
                CPI = ( double ) values [ 0 ] / values [ 1 ] ;
                printf("find_roots_vec %2d %11lld cycles %11lld instructions CPI %.3lf Speedup = %.3lf\n", 
                        lenPoly, values[ 0 ], values[ 1 ], CPI, Speedup ) ;

        }
}
void TestPAPIInv ( void )
{

        int event_set = PAPI_NULL, ret ;
        long long values [ 2 ] ;
        double CPI;
        event_set = InitPAPI () ;

        if ( event_set == PAPI_NULL )
        {
                printf ( "PAPI failed to initialize\n" ) ;
                exit ( 1 ) ;
        }

        for ( int lenPoly = 4 ; lenPoly <= 32 ; lenPoly ++ )
        {
                unsigned char in_mat [ 32 * 32 ], out_mat [ 32 * 32 ], base = 1, val = 1 ;

                for ( int i = 0 ; i < lenPoly ; i ++ )
                {
                        for ( int j = 0 ; j < lenPoly ; j ++ )
                        {
                                in_mat [ i * lenPoly + j ] = val ;
                                val = gf_mul ( val, base ) ;
                        }
                        base = gf_mul ( base, 2 ) ;
                }

                //printf ( "Vandermonde\n" ) ;
                //dump_u8xu8 ( in_mat, lenPoly, lenPoly ) ;
                if  ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK )
                {
                        handle_error ( ret ) ;
                }

                // Workload
                ret = gf_invert_matrix_1b_AVX512_GFNI ( in_mat, out_mat, lenPoly ) ;

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK)
                {
                        handle_error ( ret ) ;
                }

                //printf ( "Outmat\n" ) ;
                //dump_u8xu8 ( out_mat, lenPoly, lenPoly ) ;

                CPI = ( double ) values [ 0 ] / values [ 1 ] ;
                double vecVal = values [ 0 ] ;
                printf ( "invert_matrix_vec %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly, values [ 0 ], values [ 1 ], CPI ) ;

                if ( ( ret = PAPI_start ( event_set)) != PAPI_OK )
                {
                        handle_error(ret);
                }

                gf_invert_matrix ( in_mat, out_mat, lenPoly ) ;

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
                {
                        handle_error(ret);
                }

                //printf ( "Outmat\n" ) ;
                //dump_u8xu8 ( out_mat, lenPoly, lenPoly ) ;

                CPI = ( double ) values [ 0 ] / values [ 1 ] ;
                double baseVal = values [ 0 ] ;
                double Speedup = baseVal / vecVal ;
                printf ( "invert_matrix_sca %2d %11lld cycles %11lld instructions CPI %.3lf Speedup = %.3lf\n", 
                        lenPoly, values [ 0 ], values [ 1 ], CPI, Speedup ) ;
        }
}

void TestPAPI1b ( void )
{

        int event_set = PAPI_NULL, ret ;
        long long values [ 2 ] ;
        double CPI;
        event_set = InitPAPI () ;
        unsigned char * a ;

        a = malloc ( 256 * 4 * 255 ) ;
        if ( a == NULL )
        {
                printf ( "Allocating A failed\n" ) ;
                return ;
        }

        if ( event_set == PAPI_NULL )
        {
                printf ( "PAPI failed to initialize\n" ) ;
                exit ( 1 ) ;
        }

        // Build the power table
        pc_bpow_1b ( 3 ) ;
        // Build the log table
        pc_blog_1b () ;
        // Build the inverse table
        pc_binv_1b () ;
        //printf ( "Inverse of 3 is %d\n", pc_itab [ 3 ] ) ;
        //printf ( "%d times 3 is %d\n", pc_itab [ 3 ], pc_mul_1b ( pc_itab [ 3 ], 3 ) ) ;

        for ( int lenPoly = 2 ; lenPoly <= 32 ; lenPoly ++ )
        {
                pc_gen_rsr_matrix_1b ( a, lenPoly ) ;
                //printf ( "RSR matrix\n" ) ;
                //dump_u8xu8 ( a, 2, 255 ) ;
                //pc_bvan2 () ;
                pc_bvan_1b ( a, lenPoly ) ;

                pc_gen_poly_matrix_1b ( a, 255, 255 - lenPoly ) ;
                pc_bmat_1b ( a, lenPoly ) ;

                memset ( a, 0, 255 ) ;
                memset ( &a [ 255 - lenPoly - 1 ], 1, 1 ) ;

                for ( int pos = 0 ; pos < 255 ; pos ++ )
                {
                        a [ pos ] = rand() ;
                }

                if  ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK )
                {
                        handle_error ( ret ) ;
                }

                // Workload
                for ( int i = 0 ; i < 1000 ; i ++ )
                {
                        pc_encoder1b ( a, &a [ 255 - lenPoly ], lenPoly ) ;
                }

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK)
                {
                        handle_error ( ret ) ;
                }

                //printf ( "Parities\n" ) ;
                //dump_u8xu8 ( &a [ 255 - lenPoly ], 1, lenPoly ) ;

                CPI = ( double ) values [ 0 ] / values [ 1 ] ;
                double BPC = ( 255 - lenPoly ) * 1000 ;
                BPC /= values [ 0 ] ;
                printf ( "Encoder_1b %2d %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", lenPoly,
                         values [ 0 ] / 1000, values [ 1 ] / 1000, CPI, BPC ) ;

                if ( ( ret = PAPI_start ( event_set)) != PAPI_OK )
                {
                        handle_error(ret);
                }

                // Workload
                for ( int i = 0 ; i < 1000 ; i ++ )
                {
                        pc_decoder1b ( a, &a [ 256 ], lenPoly ) ;
                }

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
                {
                        handle_error(ret);
                }

                //printf ( "Syndromes\n" ) ;
                //dump_u8xu8 ( &a [ 256 ], 1, lenPoly ) ;

                for ( int pos = 0 ; pos < lenPoly ; pos ++ )
                {
                        if ( a [ 256 + pos ] != 0 )
                        {
                                printf ( "Syndromes\n" ) ;
                                dump_u8xu8 ( &a [ 256 ], 1, lenPoly ) ;
                        }
                }

                CPI = ( double ) values[ 0 ] / values [ 1 ] ;
                BPC = ( 255 - lenPoly ) * 1000 ;
                BPC /= values [ 0 ] ;
                printf ( "Decoder_1b %2d %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", lenPoly, 
                        values [ 0 ] / 1000, values [ 1 ] /1000, CPI, BPC ) ;
        }
        free ( a ) ;
}

void TestPAPIbm ( void )
{

        int event_set = PAPI_NULL, ret, len ;
        long long values [ 2 ] ;
        double CPI;
        event_set = InitPAPI () ;

        if ( event_set == PAPI_NULL )
        {
                printf ( "PAPI failed to initialize\n" ) ;
                exit ( 1 ) ;
        }

        for ( int lenPoly = 4 ; lenPoly <= 32 ; lenPoly += 2 )
        {
                unsigned char S [ 32 ], keyEq [ 16 ] = { 0 } ;

                unsigned char base = 1 ;
                for ( int i = 0 ; i < lenPoly ; i ++ )
                {
                        //int rvs = lenPoly - i - 1 ;
                        int rvs = i ;
                        S [ rvs ] = 0 ;
                        unsigned char val = 1 ;
                        for ( int j = 0 ; j < lenPoly/2 ; j ++ )
                        {
                                S [ rvs ] ^= val ;
                                val = gf_mul ( val, base ) ;
                        }
                        base = gf_mul ( base, 2 ) ;
                }

                if  ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK )
                {
                        handle_error ( ret ) ;
                }

                // Workload
                for ( int i = 0 ; i < 1000 ; i ++ )
                {
                        len = PGZ_1b_AVX512_GFNI ( S, lenPoly, keyEq ) ;
                }

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK)
                {
                        handle_error ( ret ) ;
                }

                CPI = ( double ) values [ 0 ] / values [ 1 ] ;
                printf ( "PGZ =  %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                         values [ 0 ] / 1000, values [ 1 ] / 1000, CPI ) ;

                // Now test Berlekamp
                unsigned char bmKeyEq [ 17 ] ;
                int bmLen ;
                if  ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK )
                {
                        handle_error ( ret ) ;
                }

                // Workload
                for ( int i = 0 ; i < 1000 ; i ++ )
                {
                        bmLen = berlekamp_massey_1b_AVX512_GFNI ( S, lenPoly, bmKeyEq ) ;
                }

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK)
                {
                        handle_error ( ret ) ;
                }

                unsigned char bmKeyEqRev [ 17 ] ;
                for ( int curKey = 0 ; curKey < len ; curKey ++ )
                {
                        bmKeyEqRev [ curKey ] = bmKeyEq [ len - curKey ] ;
                }
                if ( ( memcmp (  bmKeyEqRev, keyEq, len ) != 0 ) || ( bmLen != len ) )
                {
                        printf ( "Mismatch %d terms\n", len ) ;
                        dump_u8xu8 ( keyEq, 1, len ) ;
                        dump_u8xu8 ( bmKeyEqRev, 1, len ) ;
                        exit ( 1 ) ;
                }

                //printf ( "S and Lambda lenPoly = %d len = %d\n", lenPoly, len ) ;
                //dump_u8xu8 ( S, 1, lenPoly ) ;
                //dump_u8xu8 ( keyEq, 1, len+1 ) ;

                CPI = ( double ) values [ 0 ] / values [ 1 ] ;
                printf ( "BM_sca %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly,
                         values [ 0 ] / 1000, values [ 1 ] / 1000, CPI ) ;

                if ( ( ret = PAPI_start ( event_set)) != PAPI_OK )
                {
                        handle_error(ret);
                }

                // Workload
                for ( int i = 0 ; i < 1000 ; i ++ )
                {
                        bmLen = berlekamp_massey_1b_AVX512_GFNI ( S, lenPoly, bmKeyEq ) ;
                }

                if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
                {
                        handle_error(ret);
                }

                for ( int curKey = 0 ; curKey < len ; curKey ++ )
                {
                        bmKeyEqRev [ curKey ] = bmKeyEq [ len - curKey ] ;
                }
                if ( ( memcmp (  bmKeyEqRev, keyEq, len ) != 0 ) || ( bmLen != len ) )
                {
                        printf ( "Mismatch %d terms\n", len ) ;
                        dump_u8xu8 ( keyEq, 1, len ) ;
                        dump_u8xu8 ( bmKeyEqRev, 1, len ) ;
                        exit ( 1 ) ;
                }

                CPI = ( double ) values[ 0 ] / values [ 1 ] ;
                printf ( "BM_vec %2d %11lld cycles %11lld instructions CPI %.3lf\n", lenPoly, 
                        values [ 0 ] / 1000, values [ 1 ] / 1000, CPI ) ;
        }
}

#endif

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

#define FIELD_SIZE 256

void inject_errors_in_place_1b(unsigned char **data, int index, int num_errors, unsigned char *error_positions, uint8_t *original_values)
{
    for (int i = 0; i < num_errors; i++)
    {
        int pos = error_positions[i];
        original_values[i] = data[pos][index];
        uint8_t error = (rand() % (FIELD_SIZE - 1)) + 1;
        data[pos][index] = data[pos][index] ^ error;
    }
}

int verify_correction_in_place_1b(unsigned char **data, int index, int num_errors, unsigned char *error_positions, uint8_t *original_values)
{
    for (int i = 0; i < num_errors; i++)
    {
        if (data[error_positions[i]][index] != original_values[i])
        {
            printf ( "Error data= %d orig = %d\n", data[error_positions[i]][index], original_values[i] ) ;
            return 0;
        }
    }
    return 1;
}

int test_pgz_decoder_1b ( int index, int m, int p, unsigned char * g_tbls,
                unsigned char ** data, unsigned char ** coding, int avx2 )
{
    int successes = 0, total_tests = 0;

    for (int num_errors = 1; num_errors <= (p/2); num_errors++)
    {
        for (int start = 0; start < m - (p/2); start++)
        {
            unsigned char error_positions[16];
            uint8_t original_values[16];
            for (int i = 0; i < (p/2); i++)
            {
                error_positions[i] = start + i;
                //printf ( "Error pos [ %d ] = %d\n", i, start+i ) ;
            }
            inject_errors_in_place_1b ( data, index, num_errors, error_positions, original_values );
            
            pc_decode_data_avx512_gfni_1b ( TEST_LEN(m), m, p, g_tbls, data, coding, 1 ) ;
            //printf ( "PGZ decoder done = %d\n", done ) ;

           if ( verify_correction_in_place_1b(data, index, num_errors, error_positions, original_values ) )
            {
                successes++;
            }
            else
            {
                printf("Failed: Sequential, %d errors at %d\n", num_errors, start);
                return 0 ;
            }
            total_tests++;
        }
    }

    for (int num_errors = 1; num_errors <= (p/2) ; num_errors++)
    {
        for (int trial = 0; trial < 1000; trial++)
        {
            uint8_t error_positions[16];
            uint8_t original_values[16];
            int available[FIELD_SIZE];
            for (int i = 0; i < m; i++)
            {
                available[i] = i;
            }
            for (int i = 0; i < num_errors; i++)
            {
                int idx = rand() % (m - i);
                error_positions[i] = available[idx];
                //printf ( "Error pos [ %d ] = %d\n", i, error_positions [ i ] ) ;
                available[idx] = available[m- 1 - i];
            }
            inject_errors_in_place_1b(data, index, num_errors, error_positions, original_values);

            pc_decode_data_avx512_gfni_1b ( TEST_LEN(m), m, p, g_tbls, data, coding, 1 ) ;

            if (verify_correction_in_place_1b(data, index, num_errors, error_positions, original_values))
            {
                successes++;
            }
            else
            {
                printf("Failed: Random, %d errors, trial %d\n", num_errors, trial);
                return 0 ;
            }
            total_tests++;
        }
    }

    printf("Tests completed\n" ) ;
    return 1 ;
}

int
main(int argc, char *argv[])
{
        // Work variables
        int i, j, m, k, p, nerrs, ret = -1;
        void *buf ;
        u8 *a, *g_tbls=0, *z0=0 ;
        u8 *temp_buffs[TEST_SOURCES] = { NULL };
        u8 *buffs[TEST_SOURCES] = { NULL };

        struct perf start;

        u8 avx2=0;

        /* Set default parameters */
        k = 12;
        p = 8;
        nerrs = 4;

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

        // Match errors to parity count and compute codeword size
        nerrs = p ;
        m = k + p ;

        // Do early performance testing

        TestPAPIRoots () ;
        TestPAPIInv   () ;
        TestPAPI1b    () ;
        TestPAPIbm    () ;
        if (m > MMAX)
        {
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
        if (posix_memalign(&buf, 64, TEST_LEN(m)))
        {
                printf("Error allocating buffers\n");
                goto exit;
        }
        z0 = buf;
        memset ( z0, 0, TEST_LEN(m)) ;

        for (i = 0; i < m; i++) {
                if ( posix_memalign ( &buf, 64, TEST_LEN(m) ) )
                {
                        printf ( "Error allocating buffers\n" ) ;
                        goto exit;
                }
                buffs [ i ] = buf;
        }

        for (i = 0; i < p; i++) {
                if ( posix_memalign ( &buf, 64, TEST_LEN(m) ) )
                {
                        printf( "Error allocating buffers\n" ) ;
                        goto exit;
                }
                temp_buffs [ i ] = buf;
        }

        // Allocate gtbls
        if ( posix_memalign ( &buf, 64, KMAX * TEST_SOURCES * 32 ) )
        {
                printf ( "Error allocating g_tbls\n" ) ;
                goto exit ;
        }
        g_tbls = buf ;

        // Make random data
        for (i = 0; i < k; i++)
                for (j = 0; j < TEST_LEN(m); j++)
                        buffs[ i ][ j ] = 0 ; //rand() ;
        memset ( buffs [ k - 1 ], 1, TEST_LEN(m) ) ;
        //printf ( "memset [ k-1 ]\n" ) ;
        //dump_u8xu8 ( ( unsigned char * ) buffs [ k - 1 ], 1,16 ) ;

        // Print test type
        printf ( "Testing AVX512-GFNI\n" ) ;

        // Perform the baseline benchmark

        BENCHMARK(&start, BENCHMARK_TIME,
            ec_encode_data_avx512_gfni(TEST_LEN(m), k, p, g_tbls, buffs, &buffs[k]));

        printf("erasure_code_encode" TEST_TYPE_STR ": k=%d p=%d ", k, p);
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        unsigned char LFSRTab [ 32 ] ;
        // Test intrinsics lfsr
        pc_gen_poly_1b ( LFSRTab, p ) ;

        BENCHMARK(&start, BENCHMARK_TIME,
                pc_encode_data_avx512_gfni_1b(TEST_LEN(m), k, p, LFSRTab, buffs, &buffs [ k ]));

        printf("polynomial_code_pls" TEST_TYPE_STR ": k=%d p=%d ", k, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        BENCHMARK(&start, BENCHMARK_TIME,
                ec_encode_data_avx512_gfni(TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs));

        printf("dot_prod_decode" TEST_TYPE_STR ":     k=%d p=%d ", m, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        // Now benchmark parallel syndrome sequencer - First create power vector
        unsigned char pwrTab [ 32 ] ;
        i = 3 ;
        for ( j = p - 2 ; j >= 0 ; j -- )
        {
                pwrTab [ j ] = i ;
                i = pc_mul_1b ( i, 3 ) ;
        }

        int done ;
        BENCHMARK(&start, BENCHMARK_TIME,
                done=pc_decode_data_avx512_gfni_1b(TEST_LEN(m), m, p, pwrTab, buffs, temp_buffs, 1));

        printf("polynomial_code_pss" TEST_TYPE_STR ": k=%d p=%d ", m, p );
        perf_print(start, (long long) (TEST_LEN(m)) * (m));

        printf ( "Length decoded = %x TEST_LEN(m) = %x\n", done, TEST_LEN(m) ) ;
        for (i = 0; i < p; i++) 
        {
                if (0 != memcmp(z0, temp_buffs[i], TEST_LEN(m) ) ) 
                {
                        printf("Fail zero compare (%d, %d, %d, %d) - ", m, k, p, i);
                        dump_u8xu8 ( z0, 1, 16 ) ;
                        dump_u8xu8 ( temp_buffs [ i ], 1, 256 ) ;
                        goto exit;
               }
        }

        if ( test_pgz_decoder_1b ( 0, m, p, pwrTab, buffs, temp_buffs, avx2 ) == 0 )
        {
                printf ( "Decoder failed\n" ) ;
                goto exit ;
        }
#ifndef NOPAPI
        int event_set = InitPAPI () ; //PAPI_NULL, event_code ;
        if ( ( ret = PAPI_start ( event_set ) ) != PAPI_OK ) 
        {
                handle_error(ret);
        }
        // Workload
        ec_encode_data_avx512_gfni ( TEST_LEN(m), k, p, g_tbls, buffs, temp_buffs ) ;

        long long values[ 2 ];
        if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
        {
                handle_error( ret );
        }

        double CPI;
        CPI = ( double ) values[ 0 ] / values[ 1 ] ;
        double BPC ;
        BPC = ( double ) ( TEST_LEN(m) * m ) / values [ 0 ] ;

        printf ( "EC_Encode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[0], values[1], CPI, BPC ) ;

        if ( ( ret = PAPI_start( event_set ) ) != PAPI_OK ) 
        {
                handle_error ( ret ) ;
        }
        // Workload
        pc_encode_data_avx512_gfni_1b( TEST_LEN(m), k, p, LFSRTab, buffs, &buffs [ k ] );
        if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
        {
                handle_error ( ret ) ;
        }

        CPI = ( double ) values [ 0 ] / values [ 1 ] ;
        BPC = ( double ) ( TEST_LEN(m) * m ) / values [ 0 ] ;

        printf ( "PC_Encode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[0], values[1], CPI, BPC ) ;

        if ((ret = PAPI_start(event_set)) != PAPI_OK) {
                handle_error(ret);
        }

        ec_encode_data_avx512_gfni( TEST_LEN(m), m, p, g_tbls, buffs, temp_buffs ); 

                if ((ret = PAPI_stop(event_set, values)) != PAPI_OK)
        {
                handle_error(ret);
        }

        CPI = (double) values[0]/values[1] ;
        BPC = ( double ) ( TEST_LEN(m) * m ) / values [ 0 ] ;

        printf ( "EC_decode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[0], values[1], CPI, BPC ) ;

        if ((ret = PAPI_start(event_set)) != PAPI_OK) 
        {
                handle_error(ret);
        }

        // Workload
        int complete = pc_decode_data_avx512_gfni_1b ( TEST_LEN(m), m, p, pwrTab, buffs, temp_buffs, 1 ) ;
        //gf_4vect_pss_avx512_gfni_2d ( TEST_LEN(m), m, g_tbls, buffs, temp_buffs, 0 ) ;

        if ( ( ret = PAPI_stop ( event_set, values ) ) != PAPI_OK )
        {
                handle_error ( ret ) ;
        }

        CPI = ( double ) values[ 0 ] / values[ 1 ] ;
        BPC = ( double ) ( TEST_LEN(m) * m ) / values [ 0 ] ;


        printf ( "PC_Decode_data %11lld cycles %11lld instructions CPI %.3lf BPC %.3lf\n", values[0], values[1], CPI, BPC ) ;

        printf ( "Completed = %d tot = %d\n", complete, TEST_LEN(m) ) ;

        PAPI_cleanup_eventset ( event_set ) ;
        PAPI_destroy_eventset ( &event_set ) ;
        PAPI_shutdown ();
#endif
        printf (" done all: Pass\n" ) ;
        fflush ( stdout ) ;

        ret = 0;
exit:
        aligned_free ( z0 ) ;
        free ( a ) ;
        for (i = 0; i < TEST_SOURCES; i++)
        {
                aligned_free ( buffs[ i ] ) ;
                aligned_free ( temp_buffs[ i ] ) ;
        }
        aligned_free ( g_tbls ) ;
        return ret;
}

