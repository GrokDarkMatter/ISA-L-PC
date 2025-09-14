/**********************************************************************
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

Neither the name of Michael H. Anderson, nor the names
of his contributors may be used to endorse or promote products derived from
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
#define PC_MAX_ERRS 32

extern unsigned char pc_pow ( unsigned char base, unsigned char pow ) ;
extern unsigned char gf_div ( unsigned char a, unsigned char b ) ;
extern int pc_verify_single_error ( unsigned char * S, unsigned char ** data, int k, int p, int newPos, int offSet ) ;

int gf_invert_matrix_AVX512_GFNI(unsigned char *in_mat, unsigned char *out_mat, const int n)
{
        __m512i affineVal512 ;
        __m128i affineVal128 ;

        if (n > 32) return -1; // Assumption: n <= 32

        int i, j;
        __m512i aug_rows[32] ;                                 // Ensure 64-byte alignment
        unsigned char *matrix_mem = (unsigned char *)aug_rows; // Point to aug_rows memory

        // Initialize augmented matrix: [in_mat row | out_mat row | padding zeros]
        for (i = 0; i < n; i++)
        {
                memcpy ( &matrix_mem [ i * 64 ], &in_mat [ i * n ], n ) ;
                memset ( &matrix_mem [ i * 64 + n ], 0, n ) ;
                matrix_mem [ i * 64 + n + i ] = 1 ;
                //dump_u8xu8 ( &matrix_mem [ i * 64 + n ], 1, n ) ;
        }

        // Inverse using Gaussian elimination
        for ( i = 0; i < n; i++ )
        {
                // Check for 0 in pivot element using matrix_mem
                unsigned char pivot = matrix_mem [ i * 64 + i ] ;
                //printf ( "Pivot = %d\n", pivot ) ;
                if ( pivot == 0 )
                {
                        // Find a row with non-zero in current column and swap
                        for ( j = i + 1; j < n; j++ )
                        {
                                if ( matrix_mem [ j * 64 + i ] != 0 )
                                {
                                        break ;
                                }
                        }
                        if ( j == n )
                        {
                                // Couldn't find means it's singular
                                return -1;
                        }
                        // Swap rows i and j in ZMM registers
                        __m512i temp_vec = aug_rows [ i ] ;
                        aug_rows[i] = aug_rows [ j ] ;
                        aug_rows[j] = temp_vec ;
                }

                // Get pivot and compute 1/pivot
                pivot = matrix_mem [ i * 64 + i ]  ;
                //printf ( "Pivot2 = %d\n", pivot ) ;
                unsigned char temp_scalar = gf_inv ( pivot ) ;
                //printf ( "Scalar = %d\n", temp_scalar ) ;

                // Scale row i by 1/pivot using GFNI affine
                affineVal128 = _mm_set1_epi64x ( gf_table_gfni [ temp_scalar ] ) ;
                affineVal512 = _mm512_broadcast_i32x2 ( affineVal128 ) ;
                aug_rows [ i ]  = _mm512_gf2p8affine_epi64_epi8 ( aug_rows[ i ], affineVal512, 0 ) ;

                // Eliminate in other rows
                for ( j = 0; j < n; j++ )
                {
                        if ( j == i)  continue;
                        unsigned char factor = matrix_mem[j * 64 + i];
                        // Compute scaled pivot row: pivot_row * factor
                        affineVal128 = _mm_set1_epi64x ( gf_table_gfni [ factor ] ) ;
                        affineVal512 = _mm512_broadcast_i32x2 ( affineVal128 ) ;
                        __m512i scaled = _mm512_gf2p8affine_epi64_epi8 ( aug_rows [ i ], affineVal512, 0 ) ;
                        // row_j ^= scaled
                        aug_rows [ j ] = _mm512_xor_si512( aug_rows [ j ], scaled ) ;
                }
        }
        // Copy back to out_mat
        for ( i = 0; i < n; i++ )
        {
                //dump_u8xu8 ( &matrix_mem [ i * 64 + n ], 1, n ) ;
                memcpy ( &out_mat [ i * n ] , &matrix_mem [ i * 64 + n ], n )  ;
        }
        return 0 ;
}

int find_roots_AVX512_GFNI(unsigned char *keyEq, unsigned char *roots, int mSize)
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

// Compute error values using Vandermonde
int pc_compute_error_values_AVX512_GFNI ( int mSize, unsigned char * S, unsigned char * roots,
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
        if ( gf_invert_matrix_AVX512_GFNI ( Mat, Mat_inv, mSize ) != 0 )
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
int pc_verify_syndromes_AVX512_GFNI ( unsigned char * S, int p, int mSize, unsigned char * roots,
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

// Affine table from ec_base.h: 256 * 8-byte matrices for GF(256) multiplication
static const uint64_t gf_table_gfni[256];  // Assume defined in ec_base.h

// syndromes: array of length 'length' (typically 2t), syndromes[0] = S1, [1] = S2, etc.
// lambda: caller-allocated array of size at least (length + 1 + 31), filled with locator poly coeffs. Padded for SIMD.
// Returns: degree L of the error locator polynomial.
// Note: Assumes length <= 32 for AVX-512 (32-byte vectors); extend loops for larger lengths.
int berlekamp_massey_AVX512_GFNI(unsigned char *syndromes, int length, unsigned char *lambda)
{
    unsigned char b[PC_MAX_ERRS*2+1];  // Padded for AVX-512 (32-byte alignment)
    unsigned char temp[PC_MAX_ERRS*2+1];
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

// Attempt to detect multiple error locations and values
int pc_verify_multiple_errors_AVX512_GFNI ( unsigned char * S, unsigned char ** data, int mSize, int k,
        int p, int newPos, int offSet, unsigned char * keyEq )
{
        unsigned char roots [ PC_MAX_ERRS ] = {0} ;
        unsigned char errVal [ PC_MAX_ERRS ] ;

        // Find roots, exit if mismatch with expected roots
        int nroots = find_roots_AVX512_GFNI ( keyEq, roots, mSize );
        if ( nroots != mSize )
        {
                printf ( "Bad roots expected %d got %d\n", mSize, nroots ) ;
                return 0 ;
        }

        // Compute the error values
        if ( pc_compute_error_values_AVX512_GFNI ( mSize, S, roots, errVal ) == 0 )
        {
                return 0 ;
        }

        // Verify all syndromes are correct
        if ( pc_verify_syndromes_AVX512_GFNI ( S, p, mSize, roots, errVal ) == 0 )
        {
                return 0 ;
        }

        // Syndromes are OK, correct the user data
        for ( int i = 0 ; i < mSize ; i ++ )
        {
                int sym = k - roots [ i ] - 1 ;
                data [ sym ] [ newPos + offSet ] ^= errVal [ i ] ;
        }
        // Good correction
        return 1 ;
}

// PGZ decoding step 1, see if we can invert the matrix, if so, compute key equation
int PGZ_AVX512_GFNI ( unsigned char * S, int p, unsigned char * keyEq ) 
{
       unsigned char SMat [ PC_MAX_ERRS * PC_MAX_ERRS ], SMat_inv [ PC_MAX_ERRS * PC_MAX_ERRS ] ;
        int i,j ;

       // For each potential size, create and find Hankel matrix that will invert
        for ( int mSize = ( p / 2 ) ; mSize >= 2 ; mSize -- )
        {
                for ( i = 0 ; i < mSize ; i ++ )
                {
                        for ( j = 0 ; j < mSize ; j ++ )
                        {
                                SMat [ i * mSize + j ] = S [ i + j ] ;
                        }
                }
                // If good inversion then we know error count and can compute key equation
                if ( gf_invert_matrix_AVX512_GFNI ( SMat, SMat_inv, mSize ) == 0 )
                {
                        // Compute the key equation terms
                        for ( i = 0 ; i < mSize ; i ++ )
                        {
                                for ( j = 0 ; j < mSize ; j ++ )
                                {
                                        keyEq [ i ] ^= gf_mul ( S [ mSize + j ], SMat_inv [ i * mSize + j ] ) ;
                                }
                        }
                        return mSize ;
                }
        }
        return 0 ;

}

// Syndromes are non-zero, try to calculate error location and data values
int pc_correct_AVX512_GFNI ( int newPos, int k, int p,
    unsigned char ** data, unsigned char ** coding, int vLen )
{
        int i, mSize  ;
        unsigned char S [ PC_MAX_ERRS ], keyEq [ PC_MAX_ERRS + 1 ] = { 0 } ;

        __m512i vec, vec2 ;

        // Get a "or" of all the syndrome vectors
        vec = _mm512_load_si512( ( __m512i * )  coding [ 0 ] ) ;
        for ( i = 0 ; i < p ; i ++ )
        {
                vec2 = _mm512_load_si512 ( ( __m512i * ) coding [ i ] ) ;
                vec = _mm512_or_si512 ( vec,  vec2 ) ;
        }

        // Now find the first non-zero byte
        __mmask64 mask = _mm512_test_epi8_mask( vec, vec ) ;
        uint64_t offSet = _tzcnt_u64(mask);

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

        mSize = PGZ_AVX512_GFNI ( S, p, keyEq ) ;

        if ( mSize > 1 )
        {
                return pc_verify_multiple_errors_AVX512_GFNI ( S, data, mSize, k, p, newPos, offSet, keyEq ) ;
        }
        return 0 ;
}

int gf_2vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 2 ], taps [ 1 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_3vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 3 ], taps [ 2 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_4vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 4 ], taps [ 3 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_5vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 5 ], taps [ 4 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_6vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 6 ], taps [ 5 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_7vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 7 ], taps [ 6 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_8vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 8 ], taps [ 7 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_9vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 9 ], taps [ 8 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_10vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 10 ], taps [ 9 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_11vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 11 ], taps [ 10 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_12vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 12 ], taps [ 11 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_13vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 13 ], taps [ 12 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_14vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 14 ], taps [ 13 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_15vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 15 ], taps [ 14 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_16vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 16 ], taps [ 15 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_17vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 17 ], taps [ 16 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_18vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 18 ], taps [ 17 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_19vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 19 ], taps [ 18 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_20vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 20 ], taps [ 19 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_21vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 21 ], taps [ 20 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_22vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 22 ], taps [ 21 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_23vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 23 ], taps [ 22 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_24vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 24 ], taps [ 23 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_25vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 25 ], taps [ 24 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_26vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 26 ], taps [ 25 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_27vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 27 ], taps [ 26 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;
                parity [ 26 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 25 ], taps [ 25 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 26 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 26 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_28vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 28 ], taps [ 27 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;
                parity [ 26 ] = data_vec ;
                parity [ 27 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 25 ], taps [ 25 ], 0) ;
                        parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 26 ], taps [ 26 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 26 ], data_vec ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 27 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 26 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 27 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_29vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 29 ], taps [ 28 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;
                parity [ 26 ] = data_vec ;
                parity [ 27 ] = data_vec ;
                parity [ 28 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 25 ], taps [ 25 ], 0) ;
                        parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 26 ], taps [ 26 ], 0) ;
                        parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 27 ], taps [ 27 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 26 ], data_vec ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 27 ], data_vec ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 28 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 26 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 27 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 28 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_30vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 30 ], taps [ 29 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;
                parity [ 26 ] = data_vec ;
                parity [ 27 ] = data_vec ;
                parity [ 28 ] = data_vec ;
                parity [ 29 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 25 ], taps [ 25 ], 0) ;
                        parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 26 ], taps [ 26 ], 0) ;
                        parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 27 ], taps [ 27 ], 0) ;
                        parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 28 ], taps [ 28 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 26 ], data_vec ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 27 ], data_vec ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 28 ], data_vec ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 29 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 26 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 27 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 28 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 29 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 29 ] [ 0 ], parity [ 29 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_31vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 31 ], taps [ 30 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 29 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;
                parity [ 26 ] = data_vec ;
                parity [ 27 ] = data_vec ;
                parity [ 28 ] = data_vec ;
                parity [ 29 ] = data_vec ;
                parity [ 30 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 25 ], taps [ 25 ], 0) ;
                        parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 26 ], taps [ 26 ], 0) ;
                        parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 27 ], taps [ 27 ], 0) ;
                        parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 28 ], taps [ 28 ], 0) ;
                        parity [ 29 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 29 ], taps [ 29 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 26 ], data_vec ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 27 ], data_vec ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 28 ], data_vec ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 29 ], data_vec ) ;
                        parity [ 30 ] = _mm512_xor_si512 ( parity [ 30 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 26 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 27 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 28 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 29 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 30 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 29 ] [ 0 ], parity [ 29 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 30 ] [ 0 ], parity [ 30 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_32vect_pss_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 32 ], taps [ 31 ] ;            // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 29 * 8 ) ) );
        taps [ 30 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 30 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;
                parity [ 9 ] = data_vec ;
                parity [ 10 ] = data_vec ;
                parity [ 11 ] = data_vec ;
                parity [ 12 ] = data_vec ;
                parity [ 13 ] = data_vec ;
                parity [ 14 ] = data_vec ;
                parity [ 15 ] = data_vec ;
                parity [ 16 ] = data_vec ;
                parity [ 17 ] = data_vec ;
                parity [ 18 ] = data_vec ;
                parity [ 19 ] = data_vec ;
                parity [ 20 ] = data_vec ;
                parity [ 21 ] = data_vec ;
                parity [ 22 ] = data_vec ;
                parity [ 23 ] = data_vec ;
                parity [ 24 ] = data_vec ;
                parity [ 25 ] = data_vec ;
                parity [ 26 ] = data_vec ;
                parity [ 27 ] = data_vec ;
                parity [ 28 ] = data_vec ;
                parity [ 29 ] = data_vec ;
                parity [ 30 ] = data_vec ;
                parity [ 31 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 4 ], taps [ 4 ], 0) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 5 ], taps [ 5 ], 0) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 6 ], taps [ 6 ], 0) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 7 ], taps [ 7 ], 0) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 8 ], taps [ 8 ], 0) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 9 ], taps [ 9 ], 0) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 10 ], taps [ 10 ], 0) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 11 ], taps [ 11 ], 0) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 12 ], taps [ 12 ], 0) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 13 ], taps [ 13 ], 0) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 14 ], taps [ 14 ], 0) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 15 ], taps [ 15 ], 0) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 16 ], taps [ 16 ], 0) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 17 ], taps [ 17 ], 0) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 18 ], taps [ 18 ], 0) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 19 ], taps [ 19 ], 0) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 20 ], taps [ 20 ], 0) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 21 ], taps [ 21 ], 0) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 22 ], taps [ 22 ], 0) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 23 ], taps [ 23 ], 0) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 24 ], taps [ 24 ], 0) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 25 ], taps [ 25 ], 0) ;
                        parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 26 ], taps [ 26 ], 0) ;
                        parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 27 ], taps [ 27 ], 0) ;
                        parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 28 ], taps [ 28 ], 0) ;
                        parity [ 29 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 29 ], taps [ 29 ], 0) ;
                        parity [ 30 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 30 ], taps [ 30 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 4 ], data_vec ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 5 ], data_vec ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 6 ], data_vec ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 7 ], data_vec ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 8 ], data_vec ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 9 ], data_vec ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 10 ], data_vec ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 11 ], data_vec ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 12 ], data_vec ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 13 ], data_vec ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 14 ], data_vec ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 15 ], data_vec ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 16 ], data_vec ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 17 ], data_vec ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 18 ], data_vec ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 19 ], data_vec ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 20 ], data_vec ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 21 ], data_vec ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 22 ], data_vec ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 23 ], data_vec ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 24 ], data_vec ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 25 ], data_vec ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 26 ], data_vec ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 27 ], data_vec ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 28 ], data_vec ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 29 ], data_vec ) ;
                        parity [ 30 ] = _mm512_xor_si512 ( parity [ 30 ], data_vec ) ;
                        parity [ 31 ] = _mm512_xor_si512 ( parity [ 31 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 9 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 10 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 11 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 12 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 13 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 14 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 15 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 16 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 17 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 18 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 19 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 20 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 21 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 22 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 23 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 24 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 25 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 26 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 27 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 28 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 29 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 30 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 31 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 29 ] [ 0 ], parity [ 29 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 30 ] [ 0 ], parity [ 30 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 31 ] [ 0 ], parity [ 31 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}


int gf_2vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 2 ], taps [ 2 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
        }
        return ( curPos ) ;
}

int gf_3vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 3 ], taps [ 3 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
        }
        return ( curPos ) ;
}

int gf_4vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 4 ], taps [ 4 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
        }
        return ( curPos ) ;
}

int gf_5vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 5 ], taps [ 5 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
        }
        return ( curPos ) ;
}

int gf_6vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 6 ], taps [ 6 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
        }
        return ( curPos ) ;
}

int gf_7vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 7 ], taps [ 7 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
        }
        return ( curPos ) ;
}

int gf_8vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 8 ], taps [ 8 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
        }
        return ( curPos ) ;
}

int gf_9vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 9 ], taps [ 9 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
        }
        return ( curPos ) ;
}

int gf_10vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 10 ], taps [ 10 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
        }
        return ( curPos ) ;
}

int gf_11vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 11 ], taps [ 11 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
        }
        return ( curPos ) ;
}

int gf_12vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 12 ], taps [ 12 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
        }
        return ( curPos ) ;
}

int gf_13vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 13 ], taps [ 13 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
        }
        return ( curPos ) ;
}

int gf_14vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 14 ], taps [ 14 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
        }
        return ( curPos ) ;
}

int gf_15vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 15 ], taps [ 15 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
        }
        return ( curPos ) ;
}

int gf_16vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 16 ], taps [ 16 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
        }
        return ( curPos ) ;
}

int gf_17vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 17 ], taps [ 17 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
        }
        return ( curPos ) ;
}

int gf_18vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 18 ], taps [ 18 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
        }
        return ( curPos ) ;
}

int gf_19vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 19 ], taps [ 19 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
        }
        return ( curPos ) ;
}

int gf_20vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 20 ], taps [ 20 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
        }
        return ( curPos ) ;
}

int gf_21vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 21 ], taps [ 21 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
        }
        return ( curPos ) ;
}

int gf_22vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 22 ], taps [ 22 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
        }
        return ( curPos ) ;
}

int gf_23vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 23 ], taps [ 23 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
        }
        return ( curPos ) ;
}

int gf_24vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 24 ], taps [ 24 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
        }
        return ( curPos ) ;
}

int gf_25vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 25 ], taps [ 25 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
        }
        return ( curPos ) ;
}

int gf_26vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 26 ], taps [ 26 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
        }
        return ( curPos ) ;
}

int gf_27vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 27 ], taps [ 27 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
        }
        return ( curPos ) ;
}

int gf_28vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 28 ], taps [ 28 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
        }
        return ( curPos ) ;
}

int gf_29vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 29 ], taps [ 29 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
        }
        return ( curPos ) ;
}

int gf_30vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 30 ], taps [ 30 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 29 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity [ 29 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity [ 29 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 29 ] [ curPos ], parity [ 29 ] ) ;
        }
        return ( curPos ) ;
}

int gf_31vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 31 ], taps [ 31 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 29 * 8 ) ) );
        taps [ 30 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 30 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity [ 29 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                parity [ 30 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0 )  ) ;
                        parity [ 30 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 29 ] [ curPos ], parity [ 29 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 30 ] [ curPos ], parity [ 30 ] ) ;
        }
        return ( curPos ) ;
}

int gf_32vect_pls_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 32 ], taps [ 32 ] ;          // Parity registers
        __m512i data_vec ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 29 * 8 ) ) );
        taps [ 30 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 30 * 8 ) ) );
        taps [ 31 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 31 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity [ 4 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity [ 5 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity [ 6 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity [ 7 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity [ 8 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity [ 9 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity [ 10 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity [ 11 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity [ 12 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity [ 13 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity [ 14 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity [ 15 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity [ 16 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity [ 17 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity [ 18 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity [ 19 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity [ 20 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity [ 21 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity [ 22 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity [ 23 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity [ 24 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity [ 25 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity [ 26 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity [ 27 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity [ 28 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity [ 29 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                parity [ 30 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                parity [ 31 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 31 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0 )  ) ;
                        parity [ 30 ] = _mm512_xor_si512 ( parity [ 31 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0 )  ) ;
                        parity [ 31 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 31 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 29 ] [ curPos ], parity [ 29 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 30 ] [ curPos ], parity [ 30 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 31 ] [ curPos ], parity [ 31 ] ) ;
        }
        return ( curPos ) ;
}

void pc_encode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        switch (rows) {
        case 2: gf_2vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 3: gf_3vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 4: gf_4vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 5: gf_5vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 6: gf_6vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 7: gf_7vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 8: gf_8vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 9: gf_9vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 10: gf_10vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 11: gf_11vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 12: gf_12vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 13: gf_13vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 14: gf_14vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 15: gf_15vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 16: gf_16vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 17: gf_17vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 18: gf_18vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 19: gf_19vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 20: gf_20vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 21: gf_21vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 22: gf_22vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 23: gf_23vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 24: gf_24vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 25: gf_25vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 26: gf_26vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 27: gf_27vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 28: gf_28vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 29: gf_29vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 30: gf_30vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 31: gf_31vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 32: gf_32vect_pls_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        }
}
int pc_decode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding, int retries)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < retries ) )
        {

                switch (rows) {
                case 2: newPos = gf_2vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 3: newPos = gf_3vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 4: newPos = gf_4vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 5: newPos = gf_5vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 6: newPos = gf_6vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 7: newPos = gf_7vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 8: newPos = gf_8vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 9: newPos = gf_9vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 10: newPos = gf_10vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 11: newPos = gf_11vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 12: newPos = gf_12vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 13: newPos = gf_13vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 14: newPos = gf_14vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 15: newPos = gf_15vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 16: newPos = gf_16vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 17: newPos = gf_17vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 18: newPos = gf_18vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 19: newPos = gf_19vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 20: newPos = gf_20vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 21: newPos = gf_21vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 22: newPos = gf_22vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 23: newPos = gf_23vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 24: newPos = gf_24vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 25: newPos = gf_25vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 26: newPos = gf_26vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 27: newPos = gf_27vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 28: newPos = gf_28vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 29: newPos = gf_29vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 30: newPos = gf_30vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 31: newPos = gf_31vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 32: newPos = gf_32vect_pss_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                }
                if ( newPos < len )
                {
                        if ( pc_correct_AVX512_GFNI ( newPos, k, rows, data, coding, 64 ) )
                        {
                                return ( newPos ) ;
                        }

                }
        }
        return ( newPos ) ;
}
