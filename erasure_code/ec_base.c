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

#include <limits.h>
#include <string.h> // for memset
#include <stdint.h>
#include "erasure_code.h"
#include "ec_base.h" // for GF tables
#ifndef __aarch64__
#include <immintrin.h>
#endif
//#ifdef NDEF
// Utility print routine
#include <stdio.h>
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
//#endif
void
ec_init_tables_base(int k, int rows, unsigned char *a, unsigned char *g_tbls)
{
        int i, j;

        for (i = 0; i < rows; i++) {
                for (j = 0; j < k; j++) {
                        gf_vect_mul_init(*a++, g_tbls);
                        g_tbls += 32;
                }
        }
}

unsigned char
gf_mul(unsigned char a, unsigned char b)
{
#ifndef GF_LARGE_TABLES
        int i;

        if ((a == 0) || (b == 0))
                return 0;

        return gff_base[(i = gflog_base[a] + gflog_base[b]) > 254 ? i - 255 : i];
#else
        return gf_mul_table_base[b * 256 + a];
#endif
}

unsigned char
gf_inv(unsigned char a)
{
#ifndef GF_LARGE_TABLES
        if (a == 0)
                return 0;

        return gff_base[255 - gflog_base[a]];
#else
        return gf_inv_table_base[a];
#endif
}

void
gf_gen_rs_matrix(unsigned char *a, int m, int k)
{
        int i, j;
        unsigned char p, gen = 1;

        memset(a, 0, k * m);
        for (i = 0; i < k; i++)
                a[k * i + i] = 1;

        for (i = k; i < m; i++) {
                p = 1;
                for (j = 0; j < k; j++) {
                        a[k * i + j] = p;
                        p = gf_mul(p, gen);
                }
                gen = gf_mul(gen, 2);
        }
}

void
gf_gen_rsr_matrix(unsigned char *a, int m, int k)
{
        int i, j;
        unsigned char p, gen = 1;

        // Create the identity matrix
        memset(a, 0, k * m);
        for (i = 0; i < k; i++)
        {
                a[k * i + i] = 1 ;
        }

        // Loop through rows and cols backward
        for (i = m-1; i>=k; i--)
        {
                p = 1;
                for (j = 0; j < k; j++)
                {
                        a[k * i + (k-j-1)] = p;
                        p = gf_mul(p, gen);
                }
                gen = gf_mul(gen, 2);
        }
}

void
gf_gen_cauchy1_matrix(unsigned char *a, int m, int k)
{
        int i, j;
        unsigned char *p;

        // Identity matrix in high position
        memset(a, 0, k * m);
        for (i = 0; i < k; i++)
                a[k * i + i] = 1;

        // For the rest choose 1/(i + j) | i != j
        p = &a[k * k];
        for (i = k; i < m; i++)
                for (j = 0; j < k; j++)
                        *p++ = gf_inv(i ^ j);
}

void
gf_gen_poly( unsigned char *p, int rank)
{
        int c, alpha, cr ; // Loop variables

        p [ 0 ] = 1 ; // Start with (x+1)
        alpha = 2 ;
        for ( cr = 1 ; cr < rank ; cr ++ ) // Loop rank-1 times
        {
                // Compute the last term of the polynomial by multiplying
                p [ cr ] = gf_mul ( p [ cr - 1 ], alpha ) ;

                // Pass the middle terms to produce multiply result
                for ( c = cr - 1 ; c > 0 ; c -- )
                {
                        p [ c ] ^= gf_mul ( p [ c - 1 ], alpha ) ;
                }

                // Compute the first term by adding in alphaI
                p [ 0 ] ^= alpha ;

                // Compute next alpha (power of 2)
                alpha = gf_mul ( alpha, 2 ) ;
        }
}

void
gf_gen_poly_matrix(unsigned char *a, int m, int k)
{
        int i, j, par, over, lpos ;
        unsigned char *p, taps [ 254 ], lfsr [ 254 ] ;

        // First compute the generator polynomial and initialize the taps
        par = m - k ;

        gf_gen_poly ( taps, par ) ;
        memcpy ( lfsr, taps, par ) ; // Initial value of LFSR is the taps

        // Now use an LFSR to build the values
        p = &a[k * k];
        for ( i = k - 1 ; i >= 0 ; i-- ) // Outer loop for each col
        {
                for (j = 0; j < par ; j++) // Each row
                {
                        // Copy in the current LFSR values
                        p [ ( j * k ) + i ] = lfsr [ j ] ;
                }
                // Now update values with LFSR - first compute overflow
                over = lfsr [ 0 ] ;

                // Loop through the MSB LFSR terms (not the LSB)
                for ( lpos = 0 ; lpos < par - 1 ; lpos ++ )
                {
                        lfsr [ lpos ] = gf_mul ( over, taps [ lpos ] ) ^ lfsr [ lpos + 1 ] ;
                }
                // Now do the LSB of the LFSR to finish
                lfsr [ par - 1 ] = gf_mul ( over, taps [ par - 1 ] ) ;
        }

        // Identity matrix in high position
        memset( a , 0, k * k)  ;
        for ( i = 0; i < k ; i++ )
        {
                a [ k * i + i ] = 1 ;
        }
}

int
gf_invert_matrix(unsigned char *in_mat, unsigned char *out_mat, const int n)
{
        int i, j, k;
        unsigned char temp;

        // Set out_mat[] to the identity matrix
        for (i = 0; i < n * n; i++) // memset(out_mat, 0, n*n)
                out_mat[i] = 0;

        for (i = 0; i < n; i++)
                out_mat[i * n + i] = 1;

        // Inverse
        for (i = 0; i < n; i++) {
                // Check for 0 in pivot element
                if (in_mat[i * n + i] == 0) {
                        // Find a row with non-zero in current column and swap
                        for (j = i + 1; j < n; j++)
                                if (in_mat[j * n + i])
                                        break;

                        if (j == n) // Couldn't find means it's singular
                                return -1;

                        for (k = 0; k < n; k++) { // Swap rows i,j
                                temp = in_mat[i * n + k];
                                in_mat[i * n + k] = in_mat[j * n + k];
                                in_mat[j * n + k] = temp;

                                temp = out_mat[i * n + k];
                                out_mat[i * n + k] = out_mat[j * n + k];
                                out_mat[j * n + k] = temp;
                        }
                }

                temp = gf_inv(in_mat[i * n + i]); // 1/pivot
                for (j = 0; j < n; j++) {         // Scale row i by 1/pivot
                        in_mat[i * n + j] = gf_mul(in_mat[i * n + j], temp);
                        out_mat[i * n + j] = gf_mul(out_mat[i * n + j], temp);
                }

                for (j = 0; j < n; j++) {
                        if (j == i)
                                continue;

                        temp = in_mat[j * n + i];
                        for (k = 0; k < n; k++) {
                                out_mat[j * n + k] ^= gf_mul(temp, out_mat[i * n + k]);
                                in_mat[j * n + k] ^= gf_mul(temp, in_mat[i * n + k]);
                        }
                }
        }
        return 0;
}



// Calculates const table gftbl in GF(2^8) from single input A
// gftbl(A) = {A{00}, A{01}, A{02}, ... , A{0f} }, {A{00}, A{10}, A{20}, ... , A{f0} }

void
gf_vect_mul_init(unsigned char c, unsigned char *tbl)
{
        unsigned char c2 = (c << 1) ^ ((c & 0x80) ? 0x1d : 0);   // Mult by GF{2}
        unsigned char c4 = (c2 << 1) ^ ((c2 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        unsigned char c8 = (c4 << 1) ^ ((c4 & 0x80) ? 0x1d : 0); // Mult by GF{2}

#if (__WORDSIZE == 64 || _WIN64 || __x86_64__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        unsigned long long v1, v2, v4, v8, *t;
        unsigned long long v10, v20, v40, v80;
        unsigned char c17, c18, c20, c24;

        t = (unsigned long long *) tbl;

        v1 = c * 0x0100010001000100ull;
        v2 = c2 * 0x0101000001010000ull;
        v4 = c4 * 0x0101010100000000ull;
        v8 = c8 * 0x0101010101010101ull;

        v4 = v1 ^ v2 ^ v4;
        t[0] = v4;
        t[1] = v8 ^ v4;

        c17 = (c8 << 1) ^ ((c8 & 0x80) ? 0x1d : 0);   // Mult by GF{2}
        c18 = (c17 << 1) ^ ((c17 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c20 = (c18 << 1) ^ ((c18 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c24 = (c20 << 1) ^ ((c20 & 0x80) ? 0x1d : 0); // Mult by GF{2}

        v10 = c17 * 0x0100010001000100ull;
        v20 = c18 * 0x0101000001010000ull;
        v40 = c20 * 0x0101010100000000ull;
        v80 = c24 * 0x0101010101010101ull;

        v40 = v10 ^ v20 ^ v40;
        t[2] = v40;
        t[3] = v80 ^ v40;

#else // 32-bit or other
        unsigned char c3, c5, c6, c7, c9, c10, c11, c12, c13, c14, c15;
        unsigned char c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31;

        c3 = c2 ^ c;
        c5 = c4 ^ c;
        c6 = c4 ^ c2;
        c7 = c4 ^ c3;

        c9 = c8 ^ c;
        c10 = c8 ^ c2;
        c11 = c8 ^ c3;
        c12 = c8 ^ c4;
        c13 = c8 ^ c5;
        c14 = c8 ^ c6;
        c15 = c8 ^ c7;

        tbl[0] = 0;
        tbl[1] = c;
        tbl[2] = c2;
        tbl[3] = c3;
        tbl[4] = c4;
        tbl[5] = c5;
        tbl[6] = c6;
        tbl[7] = c7;
        tbl[8] = c8;
        tbl[9] = c9;
        tbl[10] = c10;
        tbl[11] = c11;
        tbl[12] = c12;
        tbl[13] = c13;
        tbl[14] = c14;
        tbl[15] = c15;

        c17 = (c8 << 1) ^ ((c8 & 0x80) ? 0x1d : 0);   // Mult by GF{2}
        c18 = (c17 << 1) ^ ((c17 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c19 = c18 ^ c17;
        c20 = (c18 << 1) ^ ((c18 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c21 = c20 ^ c17;
        c22 = c20 ^ c18;
        c23 = c20 ^ c19;
        c24 = (c20 << 1) ^ ((c20 & 0x80) ? 0x1d : 0); // Mult by GF{2}
        c25 = c24 ^ c17;
        c26 = c24 ^ c18;
        c27 = c24 ^ c19;
        c28 = c24 ^ c20;
        c29 = c24 ^ c21;
        c30 = c24 ^ c22;
        c31 = c24 ^ c23;

        tbl[16] = 0;
        tbl[17] = c17;
        tbl[18] = c18;
        tbl[19] = c19;
        tbl[20] = c20;
        tbl[21] = c21;
        tbl[22] = c22;
        tbl[23] = c23;
        tbl[24] = c24;
        tbl[25] = c25;
        tbl[26] = c26;
        tbl[27] = c27;
        tbl[28] = c28;
        tbl[29] = c29;
        tbl[30] = c30;
        tbl[31] = c31;

#endif //__WORDSIZE == 64 || _WIN64 || __x86_64__
}

void
gf_vect_dot_prod_base(int len, int vlen, unsigned char *v, unsigned char **src, unsigned char *dest)
{
        int i, j;
        unsigned char s;
        for (i = 0; i < len; i++) {
                s = 0;
                for (j = 0; j < vlen; j++)
                        s ^= gf_mul(src[j][i], v[j * 32 + 1]);

                dest[i] = s;
        }
}

void
gf_vect_mad_base(int len, int vec, int vec_i, unsigned char *v, unsigned char *src,
                 unsigned char *dest)
{
        int i;
        unsigned char s;
        for (i = 0; i < len; i++) {
                s = dest[i];
                s ^= gf_mul(src[i], v[vec_i * 32 + 1]);
                dest[i] = s;
        }
}

void
ec_encode_data_base(int len, int srcs, int dests, unsigned char *v, unsigned char **src,
                    unsigned char **dest)
{
        int i, j, l;
        unsigned char s;

        for (l = 0; l < dests; l++) {
                for (i = 0; i < len; i++) {
                        s = 0;
                        for (j = 0; j < srcs; j++)
                                s ^= gf_mul(src[j][i], v[j * 32 + l * srcs * 32 + 1]);

                        dest[l][i] = s;
                }
        }
}

void
ec_encode_data_update_base(int len, int k, int rows, int vec_i, unsigned char *v,
                           unsigned char *data, unsigned char **dest)
{
        int i, l;
        unsigned char s;

        for (l = 0; l < rows; l++) {
                for (i = 0; i < len; i++) {
                        s = dest[l][i];
                        s ^= gf_mul(data[i], v[vec_i * 32 + l * k * 32 + 1]);

                        dest[l][i] = s;
                }
        }
}

int
gf_vect_mul_base(int len, unsigned char *a, unsigned char *src, unsigned char *dest)
{
        // 2nd element of table array is ref value used to fill it in
        unsigned char c = a[1];

        // Len must be aligned to 32B
        if ((len % 32) != 0) {
                return -1;
        }

        while (len-- > 0)
                *dest++ = gf_mul(c, *src++);
        return 0;
}

// Assume there is a single error and try to correct, see if syndromes match
int pc_verify_single_error ( unsigned char * S, unsigned char ** data, int k, int p, 
        int newPos, int offSet )
{
        // LSB has parity, for single error this equals error value
        unsigned char eVal = S [ 0 ] ;

        // Compute error location is log2(syndrome[1]/syndrome[0])
        unsigned char eLoc = S [ 1 ] ;
        unsigned char pVal = gf_mul ( eLoc, gf_inv ( eVal ) ) ;
        eLoc = ( gflog_base [ pVal ] ) % 255 ;

        // Verify error location is reasonable
        if ( eLoc >= k )
        {
                return 0 ;
        }

        // If more than 2 syndromes, verify we can produce them all
        if ( p > 2 )
        {
                // Now verify that the error can be used to produce the remaining syndromes
                for ( int i = 2 ; i < p ; i ++ )
                {
                        if ( gf_mul ( S [ i - 1 ], pVal ) != S [ i ] )
                        {
                                return 0 ;
                        }
                }
        }
        // Good correction
        data [ k - eLoc - 1 ] [ newPos + offSet ] ^= eVal ;
        return 1 ;
}

// Affine table from ec_base.h: 256 * 8-byte matrices for GF(256) multiplication
static const uint64_t gf_table_gfni[256];

#ifndef __aarch64__
// Inverts an n x n matrix in GF(256) using Gaussian elimination, with AVX-512 GFNI acceleration.
// in_mat: input matrix (n x n, row-major, modified in place).
// out_mat: output matrix (n x n, row-major, starts as identity, becomes inverse).
// n: matrix dimension (assumes n <= 64 for SIMD; larger n needs multiple vectors per row).
// Returns: 0 on success, -1 if matrix is singular.
int gf_invert_matrix_vec(unsigned char *in_mat, unsigned char *out_mat, const int n)
{
    int i, j;
    __m512i in_vecs[64];  // Store n rows of in_mat (n <= 64)
    __m512i out_vecs[64]; // Store n rows of out_mat

    // Copy in_mat and out_mat to __m512i arrays, set out_mat to identity
    unsigned char temp_row[64] = {0};
    for (i = 0; i < n; i++)
    {
        memcpy(temp_row, &in_mat[i * n], n);
        in_vecs[i] = _mm512_loadu_si512(temp_row);
        memset(temp_row, 0, 64);
        temp_row[i] = 1; // Identity matrix
        out_vecs[i] = _mm512_loadu_si512(temp_row);
    }

    // Gaussian elimination
    for (i = 0; i < n; i++)
    {
        // Check for 0 in pivot element
        if (((unsigned char*)&in_vecs[i])[i] == 0)
        {
            // Find a row with non-zero in current column and swap
            for (j = i + 1; j < n; j++)
            {
                if (((unsigned char*)&in_vecs[j])[i])
                {
                    break;
                }
            }
            if (j == n)
            {
                // Singular matrix
                return -1;
            }
            // Swap rows i,j in __m512i arrays
            __m512i temp_vec = in_vecs[i];
            in_vecs[i] = in_vecs[j];
            in_vecs[j] = temp_vec;
            temp_vec = out_vecs[i];
            out_vecs[i] = out_vecs[j];
            out_vecs[j] = temp_vec;
        }

        // Scale row i by 1/pivot
        unsigned char temp = gf_inv(((unsigned char*)&in_vecs[i])[i]);
        __m128i matrix_128 = _mm_set1_epi64x(gf_table_gfni[temp]);  // Load affine matrix for 1/pivot
        __m512i matrix = _mm512_broadcast_i32x2(matrix_128);  // Broadcast to 512-bit vector
        in_vecs[i] = _mm512_gf2p8affine_epi64_epi8(in_vecs[i], matrix, 0);
        out_vecs[i] = _mm512_gf2p8affine_epi64_epi8(out_vecs[i], matrix, 0);

        // Eliminate column i in other rows
        for (j = 0; j < n; j++)
        {
            if (j == i)
            {
                continue;
            }
            temp = ((unsigned char*)&in_vecs[j])[i];
            if (temp == 0)
            {
                // Skip if multiplier is 0
                continue;
            }
            matrix_128 = _mm_set1_epi64x(gf_table_gfni[temp]);
            matrix = _mm512_broadcast_i32x2(matrix_128);
            __m512i mul_in_res = _mm512_gf2p8affine_epi64_epi8(in_vecs[i], matrix, 0);
            __m512i mul_out_res = _mm512_gf2p8affine_epi64_epi8(out_vecs[i], matrix, 0);
            in_vecs[j] = _mm512_xor_si512(in_vecs[j], mul_in_res);
            out_vecs[j] = _mm512_xor_si512(out_vecs[j], mul_out_res);
        }
    }

    // Copy results back to in_mat and out_mat
    for (i = 0; i < n; i++)
    {
        _mm512_storeu_si512(temp_row, in_vecs[i]);
        memcpy(&in_mat[i * n], temp_row, n);
        _mm512_storeu_si512(temp_row, out_vecs[i]);
        memcpy(&out_mat[i * n], temp_row, n);
    }

    return 0;
}
#endif
#define PC_MAX_ERRS 32

// Identify roots from key equation
int find_roots ( unsigned char * keyEq, unsigned char * roots, int mSize )
{
        int rootCount = 0 ;
        unsigned char baseVal = 1, eVal ;

        // Check each possible root
        for ( int i = 0 ; i < 255 ; i ++ )
        {
                // Loop over the Key Equation terms and sum
                eVal = 1 ;
                for ( int j = 0 ; j < mSize ; j ++ )
                {
                        eVal = gf_mul ( eVal, baseVal ) ;
                        eVal = eVal ^ keyEq [ mSize - j - 1 ] ;
                }
                // Check for a good root
                if ( eVal == 0 )
                {
                        roots [ rootCount ] = i ;
                        rootCount ++ ;
                }
                // Next evaluation is at the next power of 2
                baseVal = gf_mul ( baseVal, 2 ) ;
        }
        return rootCount ;
}
#ifndef __aarch64__
int find_roots_vec_64(unsigned char *keyEq, unsigned char *roots, int mSize)
{
        static __m512i Vandermonde [ 16 ] [ 4 ] ;
        __m512i sum [ 4 ], temp, affineVal512 ;
        __m128i affineVal128 ;
        int i, j ;

        unsigned char * vVal = ( unsigned char *) Vandermonde ;
        // Check to see if Vandermonde has been initialized yet
        if ( vVal [ 0 ] == 0 )
        {
                unsigned char base = 2, cVal = 1 ;
                for (  i = 0 ; i < 16 ; i ++ )
                {
                        vVal = ( unsigned char * ) &Vandermonde [ i ] ;
                        for ( j = 0 ; j < 255 ; j ++ )
                        {
                                vVal [ j ] = cVal ;
                                cVal = gf_mul ( cVal, base ) ;
                        }
                        base = gf_mul ( base, 2 ) ;
                }
        }
        // Initialize our sum to the constant term, no need for multiply
        sum [ 0 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;
        sum [ 1 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;
        sum [ 2 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;
        sum [ 3 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;

        // Loop through each keyEq value, multiply it by Vandermonde and add it to sum
        for ( i = 1 ; i < mSize ; i ++ )
        {
                affineVal128 = _mm_set1_epi64x ( gf_table_gfni [ keyEq [ i ] ] ) ;
                affineVal512 = _mm512_broadcast_i32x2 ( affineVal128 ) ;
                // Remember that we did not build the first row of Vandermonde, so use i-1
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 0 ], affineVal512, 0 ) ;
                sum [ 0 ] = _mm512_xor_si512 ( sum [ 0 ], temp ) ;
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 1 ], affineVal512, 0 ) ;
                sum [ 1 ] = _mm512_xor_si512 ( sum [ 1 ], temp ) ;
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 2 ], affineVal512, 0 ) ;
                sum [ 2 ] = _mm512_xor_si512 ( sum [ 2 ], temp ) ;
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 3 ], affineVal512, 0 ) ;
                sum [ 3 ] = _mm512_xor_si512 ( sum [ 3 ], temp ) ;
        }
        // Add in the leading Vandermonde row, just assume it's a one so no multiply
        sum [ 0 ] = _mm512_xor_si512 ( sum [ 0 ], Vandermonde [ mSize - 1 ] [ 0 ] ) ;
        sum [ 1 ] = _mm512_xor_si512 ( sum [ 1 ], Vandermonde [ mSize - 1 ] [ 1 ] ) ;
        sum [ 2 ] = _mm512_xor_si512 ( sum [ 2 ], Vandermonde [ mSize - 1 ] [ 2 ] ) ;
        sum [ 3 ] = _mm512_xor_si512 ( sum [ 3 ], Vandermonde [ mSize - 1 ] [ 3 ] ) ;

        int rootCount = 0, idx = 0 ;
        // Create the list of roots
        for ( i = 0 ; i < 4 ; i ++ )
        {
                // Compare each byte to zero, generating a 64-bit mask
                __mmask64 mask = _mm512_cmpeq_epi8_mask( sum [ i ], _mm512_setzero_si512());

                // Count number of zeros (popcount of mask)
                rootCount += _popcnt64(mask);

                // Extract indices of set bits (zero bytes)
                while ( mask )
                {
                        // Find the next set bit (index of zero byte)
                        uint64_t pos = _tzcnt_u64(mask);
                        roots[idx++] = (uint8_t) pos + ( i * 64 ) ;
                        // Clear the lowest set bit
                        mask = _blsr_u64 ( mask ) ; // mask &= (mask - 1)
                }
        }
        return rootCount ;
}
#endif

// Compute base ^ Power
int pc_pow ( unsigned char base, unsigned char Power ) 
{
        // The first power is always 1
        if ( Power == 0 ) 
        {
                return 1 ;
        }

        // Otherwise compute the power of two for Power
        unsigned char computedPow = base ;
        for ( int i = 1 ; i < Power ; i ++ )
        {
                computedPow = gf_mul ( computedPow, base ) ;
        }
        return computedPow ;
}

// Compute error values using Vandermonde
int pc_compute_error_values ( int mSize, unsigned char * S, unsigned char * roots,
        unsigned char * errVal )
{
        int i, j ;
        unsigned char Mat [ PC_MAX_ERRS * PC_MAX_ERRS ] ;
        unsigned char Mat_inv [ PC_MAX_ERRS * PC_MAX_ERRS ] ;

        // Find error values by building and inverting Vandemonde
        for ( i = 0 ; i < mSize ; i ++ )
        {
                Mat [ i ] = 1 ;
        }
        unsigned char base = 2 ;
        for ( i = 1 ; i < mSize ; i ++ )
        {
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        Mat [ i * mSize + j ] = pc_pow ( base, roots [ j ] ) ;
                }
                base = gf_mul ( base, 2 ) ;
        }
        // Invert matrix and verify inversion
        if ( gf_invert_matrix ( Mat, Mat_inv, mSize ) != 0 )
        {
                return 0 ;
        }

        // Compute error values by summing Syndrome terms across inverted Vandermonde
        for ( i = 0 ; i < mSize ; i ++ )
        {
                errVal [ i ] = 0 ;
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        errVal [ i ] ^= gf_mul ( S [ j ], Mat_inv [ i * mSize + j ] ) ;
                }
        }
        return 1 ;
}

// Verify proposed data values and locations can generate syndromes
int pc_verify_syndromes ( unsigned char * S, int p, int mSize, unsigned char * roots,
        unsigned char * errVal )
{
        int i,j ;
        unsigned char sum = 0 ;

        // Verify syndromes across each power row
        unsigned char base = 1 ;
        for ( i = 0 ; i < p ; i++ )
        {
                sum = 0 ;
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        // Scale up the data value based on location
                        unsigned char termVal = gf_mul ( errVal [ j ], pc_pow ( base, roots [ j ] ) ) ;
                        sum ^= termVal ;
                }

                // Verify we reproduced the syndrome
                if ( sum != S [ i ] )
                {
                        return 0 ;
                }
                // Move to next syndrome
                base = gf_mul ( base, 2 ) ;
        }
        return 1 ;
}

unsigned char gf_div ( unsigned char a, unsigned char b )
{
        return gf_mul ( a, gf_inv ( b ) ) ;
}

// Assumes external gf_mul and gf_div functions for GF(256).
// syndromes: array of length 'length' (typically 2t), syndromes[0] = S1, [1] = S2, etc.
// lambda: caller-allocated array of size at least (length + 1), filled with locator poly coeffs.
// Returns: degree L of the error locator polynomial.
int berlekamp_massey(unsigned char *syndromes, int length, unsigned char *lambda) 
{
    unsigned char b[length + 1];
    unsigned char temp[length + 1];
    int L = 0;
    int m = 1;
    unsigned char old_d = 1;  // Initial previous discrepancy

    memset(lambda, 0, length + 1);
    lambda[0] = 1;
    memset(b, 0, length + 1);
    b[0] = 1;

    for (int r = 0; r < length; r++) 
    {
        unsigned char d = syndromes[r];
        for (int j = 1; j <= L; j++) 
        {
            if (r - j >= 0) 
            {
                d ^= gf_mul(lambda[j], syndromes[r - j]);
            }
        }

        if (d == 0) 
        {
            m++;
        } 
        else 
        {
            unsigned char q = gf_div(d, old_d);
            memcpy(temp, lambda, length + 1);

            // Update lambda: lambda += q * (x^m * b), addition is XOR
            for (int j = 0; j <= length - m; j++) 
            {  // Prevent overflow
                if (b[j] != 0) 
                {  // Optional optimization
                    lambda[j + m] ^= gf_mul(q, b[j]);
                }
            }

            if (2 * L <= r) 
            {
                L = r + 1 - L;
                memcpy(b, temp, length + 1);
                old_d = d;
                m = 1;
            } else 
            {
                m++;
            }
        }
    }

    return L;
}
// Affine table from ec_base.h: 256 * 8-byte matrices for GF(256) multiplication
static const uint64_t gf_table_gfni[256];  // Assume defined in ec_base.h

#ifndef __aarch64__
// syndromes: array of length 'length' (typically 2t), syndromes[0] = S1, [1] = S2, etc.
// lambda: caller-allocated array of size at least (length + 1 + 31), filled with locator poly coeffs. Padded for SIMD.
// Returns: degree L of the error locator polynomial.
// Note: Assumes length <= 32 for AVX-512 (32-byte vectors); extend loops for larger lengths.
int berlekamp_massey_vec(unsigned char *syndromes, int length, unsigned char *lambda) 
{
    unsigned char b[length + 1 + 31];  // Padded for AVX-512 (32-byte alignment)
    unsigned char temp[length + 1 + 31];
    int L = 0;
    int m = 1;
    unsigned char old_d = 1;  // Initial previous discrepancy

    memset(lambda, 0, length + 1 + 31);
    lambda[0] = 1;
    memset(b, 0, length + 1 + 31);
    b[0] = 1;

    for (int r = 0; r < length; r++) 
    {
        unsigned char d = syndromes[r];
        for (int j = 1; j <= L; j++) 
        {
            if (r - j >= 0) {
                d ^= gf_mul(lambda[j], syndromes[r - j]);
            }
        }

        if (d == 0) 
        {
            m++;
        } 
        else 
        {
            unsigned char q = gf_div(d, old_d);
            memcpy(temp, lambda, length + 1 + 31);

            // SIMD update: lambda[j + m] ^= gf_mul(q, b[j]) using AVX-512 GF2P8AFFINE
            // Load and broadcast 8-byte affine matrix for q
            __m128i matrix_128 = _mm_set1_epi64x(gf_table_gfni[q]);  // Load uint64_t from gf_table_gfni[q]
            __m256i matrix = _mm256_broadcast_i32x2(matrix_128);  // Broadcast to all 4 lanes
            __m256i b_vec = _mm256_loadu_si256((const __m256i *)b);
            // Perform GF(256) multiplication: result = affine(b_vec, matrix) + 0
            __m256i mul_res = _mm256_gf2p8affine_epi64_epi8(b_vec, matrix, 0);
            __m256i vec_lam = _mm256_loadu_si256((const __m256i *)&lambda[m]);
            vec_lam = _mm256_xor_si256(vec_lam, mul_res);
            _mm256_storeu_si256((__m256i *)&lambda[m], vec_lam);

            // Handle remainder scalarly (unlikely needed for length <= 32)
            for (int j = 32; j <= length - m; j++) 
            {
                if (b[j] != 0) 
                {
                    lambda[j + m] ^= gf_mul(q, b[j]);
                }
            }

            if (2 * L <= r) 
            {
                L = r + 1 - L;
                memcpy(b, temp, length + 1 + 31);
                old_d = d;
                m = 1;
            } 
            else 
            {
                m++;
            }
        }
    }

    return L;
}
#endif

// Attempt to detect multiple error locations and values
int pc_verify_multiple_errors ( unsigned char * S, unsigned char ** data, int mSize, int k, 
        int p, int newPos, int offSet, unsigned char * invMat )
{
        unsigned char keyEq [ PC_MAX_ERRS ] = {0}, roots [ PC_MAX_ERRS ] = {0} ;
        unsigned char errVal [ PC_MAX_ERRS ] ;
        //unsigned char lambda [ PC_MAX_ERRS + 1 ] ;
        int i, j ;

        // Compute the key equation terms
        for ( i = 0 ; i < mSize ; i ++ )
        {
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        keyEq [ i ] ^= gf_mul ( S [ mSize + j ], invMat [ i * mSize + j ] ) ;
                }
        }
#ifdef NDEF
        printf ( "PGZ key equation\n" ) ;
        dump_u8xu8 ( keyEq, 1, mSize ) ;
        unsigned char lambda [ 17 ], lambda2 [ 33 ] ;
        int l = berlekamp_massey ( S, p, lambda ) ;
        printf ( "berlekamp\n" ) ;
        dump_u8xu8 ( lambda, 1, l+1 ) ;
        l = berlekamp_massey_vec ( S, p, lambda2 ) ;
        printf ( "Berlekamp Vec\n" ) ;
        dump_u8xu8 ( lambda, 1, l+1 ) ;
#endif
        int nroots = find_roots ( keyEq, roots, mSize );
        // Find roots, exit if mismatch with expected roots
        if ( nroots != mSize )
        {
                printf ( "Bad roots expected %d got %d\n", mSize, nroots ) ;
                return 0 ;
        }
        //printf ( "RootsS\n" ) ;
        //dump_u8xu8 ( roots, 1, nroots ) ;
#ifndef __aarch64__
        unsigned char roots2 [ 17 ] ;
        int nroots2 = find_roots_vec_64 ( keyEq, roots2, mSize ) ;
        //printf ( "nroots = %d nroots2 = %d\n", nroots, nroots2 ) ;
        //dump_u8xu8 ( roots2, 1, nroots2 ) ;
        if ( nroots != nroots2 )
        {
                printf ( "Root count scalar %d doesn't match vector %d\n", nroots, nroots2 ) ;
                return 0 ;
        }
        if ( memcmp ( roots, roots2, nroots ) )
        {
                printf ( "Roots don't match\n" ) ;
                dump_u8xu8 ( roots, 1, nroots ) ;
                dump_u8xu8 ( roots2, 1, nroots2 ) ;
                return 0 ;
        }
#endif
        // Compute the error values
        if ( pc_compute_error_values ( mSize, S, roots, errVal ) == 0 )
        {
                return 0 ;
        }

        // Verify all syndromes are correct
        if ( pc_verify_syndromes ( S, p, mSize, roots, errVal ) == 0 )
        {
                return 0 ;
        }

        // Syndromes are OK, correct the user data
        for ( i = 0 ; i < mSize ; i ++ )
        {
                int sym = k - roots [ i ] - 1 ;
                data [ sym ] [ newPos + offSet ] ^= errVal [ i ] ;
        }
        // Good correction
        return 1 ;
}

// Syndromes are non-zero, try to calculate error location and data values
int pc_correct ( int newPos, int k, int p, unsigned char ** data, char ** coding, int vLen )
{
        int offSet = 0, i, j, mSize  ;
        unsigned char synZero = 0 ;
        unsigned char S [ PC_MAX_ERRS ] ;
        unsigned char SMat [ PC_MAX_ERRS * PC_MAX_ERRS ], SMat_inv [ PC_MAX_ERRS * PC_MAX_ERRS ] ;
        //unsigned char SMat2 [ PC_MAX_ERRS * PC_MAX_ERRS ], SMat_inv2 [ PC_MAX_ERRS * PC_MAX_ERRS ] ;

        while ( offSet < vLen ) 
        {
                // Scan for first non-zero byte in syndrome vectors
                for ( i = 0 ; i < ( p / 2 ) ; i ++ )
                {
                        synZero |= coding [ i ] [ offSet ] ;
                }
                if ( synZero != 0 )
                {
                        break ;
                }
                offSet ++ ;
        }
        // Verify we found a non-zero syndrome
        if ( offSet >= vLen )
        {
                return 0 ;
        }

        // Gather up the syndromes
        for ( i = 0 ; i < p ; i ++ )
        {
                S [ i ] = coding [ p - i - 1 ] [ offSet ] ;
        }

        // Check to see if a single error can be verified
        if ( pc_verify_single_error ( S, data, k, p, newPos, offSet ) )
        {
                return 1 ;
        }

        // Create and find Hankel matrix that will invert
        for ( mSize = ( p / 2 ) ; mSize >= 2 ; mSize -- )
        {
                for ( i = 0 ; i < mSize ; i ++ )
                {
                        for ( j = 0 ; j < mSize ; j ++ )
                        {
                                SMat [ i * mSize + j ] = S [ i + j ] ;
                                //SMat2 [ i * mSize + j ] = S [ i + j ] ;
                        }
                }
                if ( gf_invert_matrix ( SMat, SMat_inv, mSize ) == 0 )
                {
#ifdef NDEF
                        gf_invert_matrix_vec ( SMat2, SMat_inv2, mSize ) ;
                        printf ( "Smat_inv2\n" ) ;
                        dump_u8xu8 ( SMat_inv, mSize, mSize ) ;
                        dump_u8xu8 ( SMat_inv2, mSize, mSize ) ;
#endif
                        return pc_verify_multiple_errors ( S, data, mSize, k, p, newPos, offSet, SMat_inv ) ;
                }
        }
        return 0 ;
}

#ifdef NDEF
// Quick experiment for AVX2
int find_zero_bytes_avx2(const uint8_t *data, uint8_t *indices)
{
    // Load 64 bytes into two YMM registers
    __m256i ymm0 = _mm256_loadu_si256((const __m256i*)data);
    __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(data + 32));

    // Compare each byte to zero, generating two 32-bit masks
    __m256i zero = _mm256_setzero_si256();
    __m256i mask0 = _mm256_cmpeq_epi8(ymm0, zero);
    __m256i mask1 = _mm256_cmpeq_epi8(ymm1, zero);

    // Extract 32-bit masks
    uint32_t mask_low = (uint32_t)_mm256_movemask_epi8(mask0);
    uint32_t mask_high = (uint32_t)_mm256_movemask_epi8(mask1);

    // Combine into a 64-bit mask
    uint64_t mask = ((uint64_t)mask_high << 32) | mask_low;

    // Count zeros and extract indices
    int num_zeros = _popcnt64(mask);
    int idx = 0;
    while (mask)
    {
        uint64_t pos = _tzcnt_u64(mask);
        indices[idx++] = (uint8_t)pos;
        mask = _blsr_u64(mask); // Clear lowest set bit
    }

    return num_zeros;
}


#include <immintrin.h>
#include <string.h>

extern __m512i gf_affine_mul_table[256];

int gf_invert_matrix_vec2(unsigned char *in_mat, unsigned char *out_mat, const int n) {
    if (n > 32) return -1; // Assumption: n <= 32

    int i, j;
    __m512i aug_rows[32] __attribute__((aligned(64))); // Ensure 64-byte alignment
    unsigned char *matrix_mem = (unsigned char *)aug_rows; // Point to aug_rows memory

    // Initialize augmented matrix: [in_mat row | out_mat row | padding zeros]
    for (i = 0; i < n; i++) {
        memcpy(&matrix_mem[i * 64], &in_mat[i * n], n);
        memcpy(&matrix_mem[i * 64 + n], &out_mat[i * n], n);
        memset(&matrix_mem[i * 64 + 2 * n], 0, 64 - 2 * n);
        aug_rows[i] = _mm512_load_si512((__m512i*)&matrix_mem[i * 64]);
    }

    // Set identity matrix in out_mat and update augmented matrix
    for (i = 0; i < n * n; i++) {
        out_mat[i] = 0;
    }
    for (i = 0; i < n; i++) {
        out_mat[i * n + i] = 1;
        memcpy(&matrix_mem[i * 64 + n], &out_mat[i * n], n); // Update identity part
        aug_rows[i] = _mm512_load_si512((__m512i*)&matrix_mem[i * 64]); // Reload after identity update
    }

    // Inverse using Gaussian elimination
    for (i = 0; i < n; i++) {
        // Check for 0 in pivot element using matrix_mem
        if (matrix_mem[i * 64 + i] == 0) {
            // Find a row with non-zero in current column and swap
            for (j = i + 1; j < n; j++) {
                if (matrix_mem[j * 64 + i] != 0) {
                    break;
                }
            }
            if (j == n) { // Couldn't find means it's singular
                return -1;
            }
            // Swap rows i and j in ZMM registers
            __m512i temp_vec = aug_rows[i];
            aug_rows[i] = aug_rows[j];
            aug_rows[j] = temp_vec;
            // No store; compiler handles spills for matrix_mem access
        }

        // Get pivot and compute 1/pivot
        unsigned char pivot = matrix_mem[i * 64 + i];
        unsigned char temp_scalar = gf_inv(pivot);

        // Scale row i by 1/pivot using GFNI affine
        aug_rows[i] = _mm512_gf2p8affine_epi64_epi8(aug_rows[i], gf_affine_mul_table[temp_scalar], 0);

        // Eliminate in other rows
        for (j = 0; j < n; j++) {
            if (j == i) continue;
            unsigned char factor = matrix_mem[j * 64 + i];
            // Compute scaled pivot row: pivot_row * factor
            __m512i scaled = _mm512_gf2p8affine_epi64_epi8(aug_rows[i], gf_affine_mul_table[factor], 0);
            // row_j ^= scaled
            aug_rows[j] = _mm512_xor_si512(aug_rows[j], scaled);
        }
    }

    // Store results to memory for final copy
    for (i = 0; i < n; i++) {
        _mm512_store_si512((__m512i*)&matrix_mem[i * 64], aug_rows[i]);
    }

    // Copy back to in_mat and out_mat
    for (i = 0; i < n; i++) {
        memcpy(&in_mat[i * n], &matrix_mem[i * 64], n);
        memcpy(&out_mat[i * n], &matrix_mem[i * 64 + n], n);
    }

    return 0;
}

#endif
