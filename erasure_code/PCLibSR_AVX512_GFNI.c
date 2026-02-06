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

unsigned char
pcsr_div_AVX512_GFNI (unsigned char a, unsigned char b)
{
    return gf_mul (a, gf_inv (b));
}

// Code generation functions
// Compute base ^ Power
int
pcsr_pow_AVX512_GFNI (unsigned char base, unsigned char Power)
{
    // The first power is always 1
    if (Power == 0)
    {
        return 1;
    }

    // Otherwise compute the power of two for Power
    unsigned char computedPow = base;
    for (int i = 1; i < Power; i++)
    {
        computedPow = gf_mul (computedPow, base);
    }
    return computedPow;
}

// Assume there is a single error and try to correct, see if syndromes match
int
pcsr_verify_single_error_AVX512_GFNI (unsigned char *S, unsigned char **data, int k, int p,
                                    int newPos, int offSet)
{
    // LSB has parity, for single error this equals error value
    unsigned char eVal = S[ 0 ];

    // Compute error location is log2(syndrome[1]/syndrome[0])
    unsigned char eLoc = S[ 1 ];
    unsigned char pVal = gf_mul (eLoc, gf_inv (eVal));
    int first ;
    //printf ( "Syndrome is:\n" ) ;
    //dump_u8xu8 ( S, 1, p ) ;
    eLoc = (gflog_base[ pVal ]) % PC_FIELD_SIZE;
    //printf ( "eLoc = %d pVal = %d\n", eLoc, pVal ) ;

    int start = p / 2 ; 
    if ( p & 1 )
    {
        first = 255 - start ;
    }
    else
    {
        first = 128 - start ;
    }
    int base = gff_base [ first ] ;
    //printf ( "First = %d Base = %d\n",first, base ) ;

    // Verify error location is reasonable
    if (eLoc >= k)
    {
        //printf ( "Ret1\n" ) ;
        return 0;
    }

    // If more than 2 syndromes, verify we can produce them all
    if (p > 2)
    {
        // Now verify that the error can be used to produce the remaining syndromes
        for (int i = 2; i < p; i++)
        {
            if (gf_mul (S[ i - 1 ], pVal) != S[ i ])
            {
                //printf ( "Ret2\n" ) ;
                return 0;
            }
        }
    }
    // Good correction - compute actual location
    //int actLoc = k - eLoc - 1 ;
    //printf ( "eVal = %d Base = %d\n", eVal, base ) ;
    unsigned char div = pcsr_pow_AVX512_GFNI (base, eLoc ) ;
    //printf ( "div = %d\n", div ) ;
    eVal = gf_mul ( eVal, gf_inv ( div ) ) ;
    //printf ( "Actual location is %d powerPosition is %d\n", actLoc, eLoc ) ;
    //printf ( "Applying to data [%d] value %d\n", actLoc, eVal ) ;
    data[ k - eLoc - 1 ][ newPos + offSet ] ^= eVal;
    return 1;
}

// Invert matrix with vector assist
int
pcsr_invert_matrix_AVX512_GFNI (unsigned char *in_mat, unsigned char *out_mat, const int n)
{
    __m512i affineVal512;
    __m128i affineVal128;

    if (n > PC_MAX_PAR)
        return -1; // Assumption: n <= 32

    int i, j;
    __m512i aug_rows[ PC_MAX_PAR ];                         // Ensure 64-byte alignment
    unsigned char *matrix_mem = (unsigned char *) aug_rows; // Point to aug_rows memory

    // Initialize augmented matrix: [in_mat row | out_mat row | padding zeros]
    for (i = 0; i < n; i++)
    {
        memcpy (&matrix_mem[ i * 64 ], &in_mat[ i * n ], n);
        memset (&matrix_mem[ i * 64 + n ], 0, n);
        matrix_mem[ i * 64 + n + i ] = 1;
        // dump_u8xu8 ( &matrix_mem [ i * 64 + n ], 1, n ) ;
    }

    // Inverse using Gaussian elimination
    for (i = 0; i < n; i++)
    {
        // Check for 0 in pivot element using matrix_mem
        unsigned char pivot = matrix_mem[ i * PC_STRIDE + i ];
        // printf ( "Pivot = %d\n", pivot ) ;
        if (pivot == 0)
        {
            // Find a row with non-zero in current column and swap
            for (j = i + 1; j < n; j++)
            {
                if (matrix_mem[ j * PC_STRIDE + i ] != 0)
                {
                    break;
                }
            }
            if (j == n)
            {
                // Couldn't find means it's singular
                return -1;
            }
            // Swap rows i and j in ZMM registers
            __m512i temp_vec = aug_rows[ i ];
            aug_rows[ i ] = aug_rows[ j ];
            aug_rows[ j ] = temp_vec;
        }

        // Get pivot and compute 1/pivot
        pivot = matrix_mem[ i * 64 + i ];
        // printf ( "Pivot2 = %d\n", pivot ) ;
        unsigned char temp_scalar = gf_inv (pivot);
        // printf ( "Scalar = %d\n", temp_scalar ) ;

        // Scale row i by 1/pivot using GFNI affine
        affineVal128 = _mm_set1_epi64x (gf_table_gfni[ temp_scalar ]);
        affineVal512 = _mm512_broadcast_i32x2 (affineVal128);
        aug_rows[ i ] = _mm512_gf2p8affine_epi64_epi8 (aug_rows[ i ], affineVal512, 0);

        // Eliminate in other rows
        for (j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            unsigned char factor = matrix_mem[ j * 64 + i ];
            // Compute scaled pivot row: pivot_row * factor
            affineVal128 = _mm_set1_epi64x (gf_table_gfni[ factor ]);
            affineVal512 = _mm512_broadcast_i32x2 (affineVal128);
            __m512i scaled = _mm512_gf2p8affine_epi64_epi8 (aug_rows[ i ], affineVal512, 0);
            // row_j ^= scaled
            aug_rows[ j ] = _mm512_xor_si512 (aug_rows[ j ], scaled);
        }
    }
    // Copy back to out_mat
    for (i = 0; i < n; i++)
    {
        // dump_u8xu8 ( &matrix_mem [ i * 64 + n ], 1, n ) ;
        memcpy (&out_mat[ i * n ], &matrix_mem[ i * 64 + n ], n);
    }
    return 0;
}

// Find rots with vector assist
int
pcsr_find_roots_AVX512_GFNI (unsigned char *keyEq, unsigned char *roots, int mSize)
{
    static __m512i Vandermonde[ PC_MAX_PAR ][ 4 ]; // 4 64 byte registers cover the
    __m512i sum[ 4 ], temp, affineVal512;          // whole field to search
    __m128i affineVal128;
    int i, j;

    unsigned char *vVal = (unsigned char *) Vandermonde;
    // Check to see if Vandermonde has been initialized yet
    if (vVal[ 0 ] == 0)
    {
        unsigned char base = PC_GEN_x11d, cVal = 1;
        for (i = 0; i < 16; i++)
        {
            vVal = (unsigned char *) &Vandermonde[ i ];
            for (j = 0; j < PC_FIELD_SIZE; j++)
            {
                vVal[ j ] = cVal;
                cVal = gf_mul (cVal, base);
            }
            base = gf_mul (base, PC_GEN_x11d);
        }
    }
    // Initialize our sum to the constant term, no need for multiply
    sum[ 0 ] = _mm512_set1_epi8 (keyEq[ 0 ]);
    sum[ 1 ] = _mm512_set1_epi8 (keyEq[ 0 ]);
    sum[ 2 ] = _mm512_set1_epi8 (keyEq[ 0 ]);
    sum[ 3 ] = _mm512_set1_epi8 (keyEq[ 0 ]);

    // Loop through each keyEq value, multiply it by Vandermonde and add it to sum
    for (i = 1; i < mSize; i++)
    {
        affineVal128 = _mm_set1_epi64x (gf_table_gfni[ keyEq[ i ] ]);
        affineVal512 = _mm512_broadcast_i32x2 (affineVal128);
        // Remember that we did not build the first row of Vandermonde, so use i-1
        temp = _mm512_gf2p8affine_epi64_epi8 (Vandermonde[ i - 1 ][ 0 ], affineVal512, 0);
        sum[ 0 ] = _mm512_xor_si512 (sum[ 0 ], temp);
        temp = _mm512_gf2p8affine_epi64_epi8 (Vandermonde[ i - 1 ][ 1 ], affineVal512, 0);
        sum[ 1 ] = _mm512_xor_si512 (sum[ 1 ], temp);
        temp = _mm512_gf2p8affine_epi64_epi8 (Vandermonde[ i - 1 ][ 2 ], affineVal512, 0);
        sum[ 2 ] = _mm512_xor_si512 (sum[ 2 ], temp);
        temp = _mm512_gf2p8affine_epi64_epi8 (Vandermonde[ i - 1 ][ 3 ], affineVal512, 0);
        sum[ 3 ] = _mm512_xor_si512 (sum[ 3 ], temp);
    }
    // Add in the leading Vandermonde row, just assume it's a one so no multiply
    sum[ 0 ] = _mm512_xor_si512 (sum[ 0 ], Vandermonde[ mSize - 1 ][ 0 ]);
    sum[ 1 ] = _mm512_xor_si512 (sum[ 1 ], Vandermonde[ mSize - 1 ][ 1 ]);
    sum[ 2 ] = _mm512_xor_si512 (sum[ 2 ], Vandermonde[ mSize - 1 ][ 2 ]);
    sum[ 3 ] = _mm512_xor_si512 (sum[ 3 ], Vandermonde[ mSize - 1 ][ 3 ]);

    int rootCount = 0, idx = 0;
    // Create the list of roots
    for (i = 0; i < PC_L1PAR; i++)
    {
        // Compare each byte to zero, generating a 64-bit mask
        __mmask64 mask = _mm512_cmpeq_epi8_mask (sum[ i ], _mm512_setzero_si512 ());

        // Count number of zeros (popcount of mask)
        rootCount += _mm_popcnt_u64 (mask);

        // Extract indices of set bits (zero bytes)
        while (mask)
        {
            // Find the next set bit (index of zero byte)
            uint64_t pos = _tzcnt_u64 (mask);
            roots[ idx++ ] = (uint8_t) pos + (i * PC_STRIDE);
            // Clear the lowest set bit
            mask = _blsr_u64 (mask); // mask &= (mask - 1)
        }
    }
    return rootCount;
}

// Compute error values using Vandermonde
int
pcsr_compute_error_values_AVX512_GFNI (int mSize, unsigned char *S, unsigned char *roots,
                                     unsigned char *errVal, int bVal )
{
    int i, j;
    unsigned char Mat[ PC_MAX_PAR * PC_MAX_PAR ];
    unsigned char Mat_inv[ PC_MAX_PAR * PC_MAX_PAR ];

    // Find error values by building and inverting Vandemonde

    //unsigned char base = PC_GEN_x11d; // *****************************************this needs to be updated with offset for SR polynomial instead of 2
    int base = bVal ;
    for (i = 0; i < mSize; i++)
    {
        for (j = 0; j < mSize; j++)
        {
            Mat[ i * mSize + j ] = pc_pow_AVX512_GFNI (base, roots[ j ]);
        }
        base = gf_mul (base, PC_GEN_x11d);
    }
    //printf ( "Before invert\n" ) ;
    //dump_u8xu8 ( Mat, mSize, mSize ) ;
    // Invert matrix and verify inversion
    if (gf_invert_matrix_AVX512_GFNI (Mat, Mat_inv, mSize) != 0)
    {
        return 0;
    }

    // Compute error values by summing Syndrome terms across inverted Vandermonde
    for (i = 0; i < mSize; i++)
    {
        errVal[ i ] = 0;
        for (j = 0; j < mSize; j++)
        {
            errVal[ i ] ^= gf_mul (S[ j ], Mat_inv[ i * mSize + j ]);
        }
        //printf ( "ErrVal [ %d ] = %d\n", i, errVal [ i ] ) ;
    }
    return 1;
}

// Verify proposed data values and locations can generate syndromes
int
pcsr_verify_syndromes_AVX512_GFNI (unsigned char *S, int p, int mSize, unsigned char *roots,
                                 unsigned char *errVal, int baseVal )
{
    int i, j;
    unsigned char sum = 0;

    // Verify syndromes across each power row
    unsigned char base = baseVal;
    for (i = 0; i < p; i++)
    {
        sum = 0;
        for (j = 0; j < mSize; j++)
        {
            // Scale up the data value based on location
            unsigned char termVal = gf_mul (errVal[ j ], pc_pow_AVX512_GFNI (base, roots[ j ]));
            sum ^= termVal;
        }

        // Verify we reproduced the syndrome
        if (sum != S[ i ])
        {
            return 0;
        }
        // Move to next syndrome
        base = gf_mul (base, PC_GEN_x11d);
    }
    return 1;
}

// Affine table from ec_base.h: 256 * 8-byte matrices for GF(256) multiplication
static const uint64_t gf_table_gfni[ 256 ]; // Assume defined in ec_base.h

// syndromes: array of length 'length' (typically 2t), syndromes[0] = S1, [1] = S2, etc.
// lambda: caller-allocated array of size at least (length + 1 + 31), filled with locator poly
// coeffs. Padded for SIMD. Returns: degree L of the error locator polynomial. Note: Assumes length
// <= 32 for AVX-512 (32-byte vectors); extend loops for larger lengths.
int
pcsr_berlekamp_massey_AVX512_GFNI (unsigned char *syndromes, int length, unsigned char *lambda)
{
    unsigned char b[ PC_MAX_PAR * 2 + 1 ]; // Padded for AVX-512 (32-byte alignment)
    unsigned char temp[ PC_MAX_PAR * 2 + 1 ];
    int L = 0;
    int m = 1;
    unsigned char old_d = 1; // Initial previous discrepancy

    memset (lambda, 0, length + 1 + 31);
    lambda[ 0 ] = 1;
    memset (b, 0, length + 1 + 31);
    b[ 0 ] = 1;

    for (int r = 0; r < length; r++)
    {
        unsigned char d = syndromes[ r ];
        for (int j = 1; j <= L; j++)
        {
            if (r - j >= 0)
            {
                d ^= gf_mul (lambda[ j ], syndromes[ r - j ]);
            }
        }

        if (d == 0)
        {
            m++;
        }
        else
        {
            unsigned char q = gf_div_AVX512_GFNI (d, old_d);
            memcpy (temp, lambda, length + 1 + 31);

            // SIMD update: lambda[j + m] ^= gf_mul(q, b[j]) using AVX-512 GF2P8AFFINE
            // Load and broadcast 8-byte affine matrix for q
            __m128i matrix_128 =
                    _mm_set1_epi64x (gf_table_gfni[ q ]); // Load uint64_t from gf_table_gfni[q]
            __m256i matrix = _mm256_broadcast_i32x2 (matrix_128); // Broadcast to all 4 lanes
            __m256i b_vec = _mm256_loadu_si256 ((const __m256i *) b);
            // Perform GF(256) multiplication: result = affine(b_vec, matrix) + 0
            __m256i mul_res = _mm256_gf2p8affine_epi64_epi8 (b_vec, matrix, 0);
            __m256i vec_lam = _mm256_loadu_si256 ((const __m256i *) &lambda[ m ]);
            vec_lam = _mm256_xor_si256 (vec_lam, mul_res);
            _mm256_storeu_si256 ((__m256i *) &lambda[ m ], vec_lam);

            // Handle remainder scalarly (unlikely needed for length <= 32)
            for (int j = 32; j <= length - m; j++)
            {
                if (b[ j ] != 0)
                {
                    lambda[ j + m ] ^= gf_mul (q, b[ j ]);
                }
            }

            if (2 * L <= r)
            {
                L = r + 1 - L;
                memcpy (b, temp, length + 1 + 31);
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
int
pcsr_verify_multiple_errors_AVX512_GFNI (unsigned char *S, unsigned char **data, int mSize, int k,
                                       int p, int newPos, int offSet, unsigned char *keyEq)
{
    unsigned char roots[ PC_MAX_PAR ] = { 0 };
    unsigned char errVal[ PC_MAX_PAR ];

    // Find roots, exit if mismatch with expected roots
    int nroots = pcsr_find_roots_AVX512_GFNI (keyEq, roots, mSize);
    if (nroots != mSize)
    {
        printf ("Bad roots expected %d got %d\n", mSize, nroots);
        return 0;
    }

    //printf ("Roots found = %d\n", nroots ) ;
    //dump_u8xu8 ( roots, 1, mSize ) ;

    int first ;
    int start = p / 2 ; 
    if ( p & 1 )
    {
        first = 255 - start ;
    }
    else
    {
        first = 128 - start ;
    }

    int base = gff_base [ first ] ;

    // Compute the error values
    if (pcsr_compute_error_values_AVX512_GFNI (mSize, S, roots, errVal, base) == 0)
    {
        return 0;
    }

    // Verify all syndromes are correct
    if (pcsr_verify_syndromes_AVX512_GFNI (S, p, mSize, roots, errVal, base) == 0)
    {
        printf ( "Verify failed\n" ) ;
        return 0;
    }

    // Syndromes are OK, correct the user data
    for (int i = 0; i < mSize; i++)
    {
        int sym = k - roots[ i ] - 1;
        //printf ( "Correcting data [%d] [%d] with errVal %d\n", sym, newPos + offSet, errVal [ i ] ) ;
        data[ sym ][ newPos + offSet ] ^= errVal[ i ];
    }
    // Good correction
    return 1;
}

// PGZ decoding step 1, see if we can invert the matrix, if so, compute key equation
int
pcsr_PGZ_AVX512_GFNI (unsigned char *S, int p, unsigned char *keyEq)
{
    unsigned char SMat[ PC_MAX_PAR * PC_MAX_PAR ], SMat_inv[ PC_MAX_PAR * PC_MAX_PAR ];
    int i, j;

    // For each potential size, create and find Hankel matrix that will invert
    for (int mSize = (p / 2); mSize >= 2; mSize--)
    {
        for (i = 0; i < mSize; i++)
        {
            for (j = 0; j < mSize; j++)
            {
                SMat[ i * mSize + j ] = S[ i + j ];
            }
        }
        // If good inversion then we know error count and can compute key equation
        if (gf_invert_matrix_AVX512_GFNI (SMat, SMat_inv, mSize) == 0)
        {
            //printf ( "PGZ Good Inversion size %d\n", mSize ) ;
            // Compute the key equation terms
            for (i = 0; i < mSize; i++)
            {
                for (j = 0; j < mSize; j++)
                {
                    keyEq[ i ] ^= gf_mul (S[ mSize + j ], SMat_inv[ i * mSize + j ]);
                }
            }
            //printf ( "Error locator\n" ) ;
            //dump_u8xu8 ( keyEq, 1, mSize ) ;
            return mSize;
        }
    }
    return 0;
}

// Syndromes are non-zero, try to calculate error location and data values
int
pcsr_correct_AVX512_GFNI (int newPos, int k, int p, unsigned char **data, unsigned char **coding,
                        int vLen)
{
    int i, mSize;
    unsigned char S[ PC_MAX_PAR ], keyEq[ PC_MAX_PAR + 1 ] = { 0 };

    __m512i vec, vec2;

    // Get a "or" of all the syndrome vectors
    vec = _mm512_load_si512 ((__m512i *) coding[ 0 ]);
    for (i = 0; i < p; i++)
    {
        vec2 = _mm512_load_si512 ((__m512i *) coding[ i ]);
        vec = _mm512_or_si512 (vec, vec2);
    }

    // Now find the first non-zero byte
    __mmask64 mask = _mm512_test_epi8_mask (vec, vec);
    uint64_t offSet = _tzcnt_u64 (mask);

    // Verify we found a non-zero syndrome
    if (offSet >= vLen)
    {
        return 0;
    }

    // Gather up the syndromes
    for (i = 0; i < p; i++)
    {
        S[ i ] = coding[ p - i - 1 ][ offSet ];
    }

    // Check to see if a single error can be verified
    if (pcsr_verify_single_error_AVX512_GFNI (S, data, k, p, newPos, offSet))
    {
        return 1;
    }

    mSize = pcsr_PGZ_AVX512_GFNI (S, p, keyEq);

    if (mSize > 1)
    {
        return pcsr_verify_multiple_errors_AVX512_GFNI (S, data, mSize, k, p, newPos, offSet, keyEq);
    }
    return 0;
}

void pcsr_recon_1p ( unsigned char ** source, unsigned char ** dest, int len, int k, int e )
{
    unsigned char **curS ;
    __m512i sum ;

    for ( int curPos = 0 ; curPos < len ; curPos += 64 )
    {
        curS = source ;
        curS ++ ;
        sum = _mm512_stream_load_si512( *curS + curPos ) ;
        for ( int curK = 1 ; curK < k ; curK ++ )
        {
            sum = _mm512_xor_si512 ( sum, *( __m512i *)curS ) ;
            curS ++ ;
        }
        _mm512_stream_si512 ( (__m512i * ) ( *dest + curPos), sum ) ;
    }
}
void pcsr_recon_1m ( unsigned char ** source, unsigned char ** dest, int len, int k, int e )
{
    unsigned char **curS ;
    __m512i sum, affineval ;

    affineval = _mm512_stream_load_si512 ( *source) ;

    for ( int curPos = 0 ; curPos < len ; curPos += 64 )
    {
        curS = source ;
        curS ++ ;
        sum = _mm512_stream_load_si512( *curS + curPos ) ;
        for ( int curK = 1 ; curK < k ; curK ++ )
        {
            sum = _mm512_gf2p8affine_epi64_epi8 (sum, affineval, 0 ) ;
            sum = _mm512_xor_si512 ( sum, *( __m512i *)curS ) ;
            curS ++ ;
        }
        _mm512_stream_si512 ( (__m512i * ) ( *dest + curPos), sum ) ;
    }
}
int
pcsr_gen_poly (unsigned char *p, int rank )
{
    int c, alpha, cr, first, retVal; // Loop variables

    //p[ 0 ] = 1; // Start with (x+1)
    int start = rank / 2 ; // = first;
    if ( rank & 1 )
    {
        first = 255 - start ;
    }
    else
    {
        first = 128 - start ;
    }
    //printf ( "Rank = %d, first = %d\n", rank, first ) ;
    first = pcsr_pow_AVX512_GFNI (2, first);
    retVal = first ;
    //printf ("Poly power First = %d  ", first);
    p[ 0 ] = first; // Start with (x+1)
    //alpha = 2;
    alpha = gf_mul (first, 2);
    for (cr = 1; cr < rank; cr++) // Loop rank-1 times
    {
        // Compute the last term of the polynomial by multiplying
        p[ cr ] = gf_mul (p[ cr - 1 ], alpha);

        // Pass the middle terms to produce multiply result
        for (c = cr - 1; c > 0; c--)
        {
            p[ c ] ^= gf_mul (p[ c - 1 ], alpha);
        }

        // Compute the first term by adding in alphaI
        p[ 0 ] ^= alpha;

        // Compute next alpha (power of 2)
        alpha = gf_mul (alpha, 2);
    }

    //printf ("FinPoly: %d ", start);
    //dump_u8xu8 (p, 1, rank);
    return retVal ;
}

int
pcsr_gen_poly_matrix (unsigned char *a, int m, int k)
{
    int i, j, par, over, lpos, retVal;
    unsigned char *p, taps[ 254 ], lfsr[ 254 ];

    // First compute the generator polynomial and initialize the taps
    par = m - k;

    retVal = pcsr_gen_poly (taps, par);
    memcpy (lfsr, taps, par); // Initial value of LFSR is the taps

    // Now use an LFSR to build the values
    p = &a[ k * k ];
    for (i = k - 1; i >= 0; i--) // Outer loop for each col
    {
        for (j = 0; j < par; j++) // Each row
        {
            // Copy in the current LFSR values
            p[ (j * k) + i ] = lfsr[ j ];
        }
        // Now update values with LFSR - first compute overflow
        over = lfsr[ 0 ];

        // Loop through the MSB LFSR terms (not the LSB)
        for (lpos = 0; lpos < par - 1; lpos++)
        {
            lfsr[ lpos ] = gf_mul (over, taps[ lpos ]) ^ lfsr[ lpos + 1 ];
        }
        // Now do the LSB of the LFSR to finish
        lfsr[ par - 1 ] = gf_mul (over, taps[ par - 1 ]);
    }

    // Identity matrix in high position
    memset (a, 0, k * k);
    for (i = 0; i < k; i++)
    {
        a[ k * i + i ] = 1;
    }
    return retVal ;
}

void
pcsr_gen_rsr_matrix (unsigned char *a, int m, int k, int first)
{
    int i, j;
    unsigned char p, gen = first;

    // Create the identity matrix
    memset (a, 0, k * m);
    for (i = 0; i < k; i++)
    {
        a[ k * i + i ] = 1;
    }

    // Loop through rows and cols backward
    for (i = m - 1; i >= k; i--)
    {
        p = 1;
        for (j = 0; j < k; j++)
        {
            a[ k * i + (k - j - 1) ] = p;
            p = gf_mul (p, gen);
        }
        gen = gf_mul (gen, 2);
    }
}

// Parallel Syndrome Sequencer SR for P = 2 Codewords
int pcsr_2vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 2 ], taps [ 2 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

// Parallel Syndrome Sequencer SR for P = 3 Codewords
int pcsr_3vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 3 ], taps [ 3 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 4 Codewords
int pcsr_4vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 4 ], taps [ 4 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 3 ], taps [ 3 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 5 Codewords
int pcsr_5vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 5 ], taps [ 5 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 6 Codewords
int pcsr_6vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 6 ], taps [ 6 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 7 Codewords
int pcsr_7vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 7 ], taps [ 7 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 8 Codewords
int pcsr_8vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 8 ], taps [ 8 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 9 Codewords
int pcsr_9vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 9 ], taps [ 9 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;
                parity [ 7 ] = data_vec ;
                parity [ 8 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 4 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 5 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 6 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 7 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 8 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 10 Codewords
int pcsr_10vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 10 ], taps [ 10 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 11 Codewords
int pcsr_11vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 11 ], taps [ 11 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 12 Codewords
int pcsr_12vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 12 ], taps [ 12 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 13 Codewords
int pcsr_13vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 13 ], taps [ 13 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 14 Codewords
int pcsr_14vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 14 ], taps [ 14 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 15 Codewords
int pcsr_15vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 15 ], taps [ 15 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 16 Codewords
int pcsr_16vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 16 ], taps [ 16 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 17 Codewords
int pcsr_17vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 17 ], taps [ 17 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 18 Codewords
int pcsr_18vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 18 ], taps [ 18 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 19 Codewords
int pcsr_19vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 19 ], taps [ 19 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 20 Codewords
int pcsr_20vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 20 ], taps [ 20 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 21 Codewords
int pcsr_21vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 21 ], taps [ 21 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 22 Codewords
int pcsr_22vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 22 ], taps [ 22 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 23 Codewords
int pcsr_23vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 23 ], taps [ 23 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 24 Codewords
int pcsr_24vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 24 ], taps [ 24 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 25 Codewords
int pcsr_25vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 25 ], taps [ 25 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 26 Codewords
int pcsr_26vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 26 ], taps [ 26 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 27 Codewords
int pcsr_27vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 27 ], taps [ 27 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 26 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 28 Codewords
int pcsr_28vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 28 ], taps [ 28 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 27 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 29 Codewords
int pcsr_29vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 29 ], taps [ 29 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 28 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 30 Codewords
int pcsr_30vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 30 ], taps [ 30 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 29 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 31 Codewords
int pcsr_31vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 31 ], taps [ 31 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 29 * 8 ) ) );
        taps [ 30 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 30 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                }

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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

// Parallel Syndrome Sequencer SR for P = 32 Codewords
int pcsr_32vect_pss_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 32 ], taps [ 32 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char ** dp ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );
        taps [ 16 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 16 * 8 ) ) );
        taps [ 17 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 17 * 8 ) ) );
        taps [ 18 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 18 * 8 ) ) );
        taps [ 19 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 19 * 8 ) ) );
        taps [ 20 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 20 * 8 ) ) );
        taps [ 21 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 21 * 8 ) ) );
        taps [ 22 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 22 * 8 ) ) );
        taps [ 23 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 23 * 8 ) ) );
        taps [ 24 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 24 * 8 ) ) );
        taps [ 25 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 25 * 8 ) ) );
        taps [ 26 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 26 * 8 ) ) );
        taps [ 27 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 27 * 8 ) ) );
        taps [ 28 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 28 * 8 ) ) );
        taps [ 29 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 29 * 8 ) ) );
        taps [ 30 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 30 * 8 ) ) );
        taps [ 31 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 31 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                dp = data ;
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( *dp + curPos ) ;
              __builtin_prefetch ( *dp + curPos + 64 , 0, 3 ) ;
                // Move data vector pointer to next symbol
                // Initialize parity values to Symbol 0
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

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        dp ++ ;
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( *dp + curPos ) ;
                      __builtin_prefetch ( *dp + curPos + 64, 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
                        parity [ 31 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 31 ], taps [ 31 ], 0) ;
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

                // Verify Syndromes are zero
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
                // Store syndromes and exit function on non-zero syndromes
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


// Parallel LFSR_SR Sequencer for P = 2 Codewords
int pcsr_2vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 2 ], taps [ 1 ] ;          // Parity registers
        __m512i data_vec, temp [ 1 ] ;
        unsigned char **sPnt, *pPnt [ 2 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 3 Codewords
int pcsr_3vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 3 ], taps [ 1 ] ;          // Parity registers
        __m512i data_vec, temp [ 1 ] ;
        unsigned char **sPnt, *pPnt [ 3 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 0 ] ) ;
                        parity [ 2 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 4 Codewords
int pcsr_4vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 4 ], taps [ 2 ] ;          // Parity registers
        __m512i data_vec, temp [ 2 ] ;
        unsigned char **sPnt, *pPnt [ 4 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 0 ] ) ;
                        parity [ 3 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 5 Codewords
int pcsr_5vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 5 ], taps [ 2 ] ;          // Parity registers
        __m512i data_vec, temp [ 2 ] ;
        unsigned char **sPnt, *pPnt [ 5 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 1 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 0 ] ) ;
                        parity [ 4 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 6 Codewords
int pcsr_6vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 6 ], taps [ 3 ] ;          // Parity registers
        __m512i data_vec, temp [ 3 ] ;
        unsigned char **sPnt, *pPnt [ 6 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 1 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 0 ] ) ;
                        parity [ 5 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 7 Codewords
int pcsr_7vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 7 ], taps [ 3 ] ;          // Parity registers
        __m512i data_vec, temp [ 3 ] ;
        unsigned char **sPnt, *pPnt [ 7 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 2 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 1 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 0 ] ) ;
                        parity [ 6 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 8 Codewords
int pcsr_8vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 8 ], taps [ 4 ] ;          // Parity registers
        __m512i data_vec, temp [ 4 ] ;
        unsigned char **sPnt, *pPnt [ 8 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 2 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 1 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 0 ] ) ;
                        parity [ 7 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 9 Codewords
int pcsr_9vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 9 ], taps [ 4 ] ;          // Parity registers
        __m512i data_vec, temp [ 4 ] ;
        unsigned char **sPnt, *pPnt [ 9 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 3 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 2 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 1 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 0 ] ) ;
                        parity [ 8 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 10 Codewords
int pcsr_10vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 10 ], taps [ 5 ] ;          // Parity registers
        __m512i data_vec, temp [ 5 ] ;
        unsigned char **sPnt, *pPnt [ 10 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 3 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 2 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 1 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 0 ] ) ;
                        parity [ 9 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 11 Codewords
int pcsr_11vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 11 ], taps [ 5 ] ;          // Parity registers
        __m512i data_vec, temp [ 5 ] ;
        unsigned char **sPnt, *pPnt [ 11 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 4 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 3 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 2 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 1 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 0 ] ) ;
                        parity [ 10 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 12 Codewords
int pcsr_12vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 12 ], taps [ 6 ] ;          // Parity registers
        __m512i data_vec, temp [ 6 ] ;
        unsigned char **sPnt, *pPnt [ 12 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 4 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 3 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 2 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 1 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 0 ] ) ;
                        parity [ 11 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 13 Codewords
int pcsr_13vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 13 ], taps [ 6 ] ;          // Parity registers
        __m512i data_vec, temp [ 6 ] ;
        unsigned char **sPnt, *pPnt [ 13 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 5 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 4 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 3 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 2 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 1 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 0 ] ) ;
                        parity [ 12 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 14 Codewords
int pcsr_14vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 14 ], taps [ 7 ] ;          // Parity registers
        __m512i data_vec, temp [ 7 ] ;
        unsigned char **sPnt, *pPnt [ 14 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 5 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 4 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 3 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 2 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 1 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 0 ] ) ;
                        parity [ 13 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 15 Codewords
int pcsr_15vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 15 ], taps [ 7 ] ;          // Parity registers
        __m512i data_vec, temp [ 7 ] ;
        unsigned char **sPnt, *pPnt [ 15 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 6 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 5 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 4 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 3 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 2 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 1 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 0 ] ) ;
                        parity [ 14 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 16 Codewords
int pcsr_16vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 16 ], taps [ 8 ] ;          // Parity registers
        __m512i data_vec, temp [ 8 ] ;
        unsigned char **sPnt, *pPnt [ 16 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 6 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 5 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 4 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 3 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 2 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 1 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 0 ] ) ;
                        parity [ 15 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 17 Codewords
int pcsr_17vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 17 ], taps [ 8 ] ;          // Parity registers
        __m512i data_vec, temp [ 8 ] ;
        unsigned char **sPnt, *pPnt [ 17 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 7 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 6 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 5 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 4 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 3 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 2 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 1 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 0 ] ) ;
                        parity [ 16 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 18 Codewords
int pcsr_18vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 18 ], taps [ 9 ] ;          // Parity registers
        __m512i data_vec, temp [ 9 ] ;
        unsigned char **sPnt, *pPnt [ 18 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 7 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 6 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 5 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 4 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 3 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 2 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 1 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 0 ] ) ;
                        parity [ 17 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 19 Codewords
int pcsr_19vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 19 ], taps [ 9 ] ;          // Parity registers
        __m512i data_vec, temp [ 9 ] ;
        unsigned char **sPnt, *pPnt [ 19 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 8 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 7 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 6 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 5 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 4 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 3 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 2 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 1 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 0 ] ) ;
                        parity [ 18 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 20 Codewords
int pcsr_20vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 20 ], taps [ 10 ] ;          // Parity registers
        __m512i data_vec, temp [ 10 ] ;
        unsigned char **sPnt, *pPnt [ 20 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 8 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 7 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 6 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 5 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 4 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 3 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 2 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 1 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 0 ] ) ;
                        parity [ 19 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 21 Codewords
int pcsr_21vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 21 ], taps [ 10 ] ;          // Parity registers
        __m512i data_vec, temp [ 10 ] ;
        unsigned char **sPnt, *pPnt [ 21 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 9 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 8 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 7 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 6 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 5 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 4 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 3 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 2 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 1 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 0 ] ) ;
                        parity [ 20 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 22 Codewords
int pcsr_22vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 22 ], taps [ 11 ] ;          // Parity registers
        __m512i data_vec, temp [ 11 ] ;
        unsigned char **sPnt, *pPnt [ 22 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 9 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 8 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 7 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 6 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 5 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 4 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 3 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 2 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 1 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 0 ] ) ;
                        parity [ 21 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 23 Codewords
int pcsr_23vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 23 ], taps [ 11 ] ;          // Parity registers
        __m512i data_vec, temp [ 11 ] ;
        unsigned char **sPnt, *pPnt [ 23 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 10 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 9 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 8 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 7 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 6 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 5 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 4 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 3 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 2 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 1 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 0 ] ) ;
                        parity [ 22 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 24 Codewords
int pcsr_24vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 24 ], taps [ 12 ] ;          // Parity registers
        __m512i data_vec, temp [ 12 ] ;
        unsigned char **sPnt, *pPnt [ 24 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 10 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 9 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 8 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 7 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 6 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 5 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 4 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 3 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 2 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 1 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 0 ] ) ;
                        parity [ 23 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 25 Codewords
int pcsr_25vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 25 ], taps [ 12 ] ;          // Parity registers
        __m512i data_vec, temp [ 12 ] ;
        unsigned char **sPnt, *pPnt [ 25 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 11 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 10 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 9 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 8 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 7 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 6 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 5 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 4 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 3 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 2 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 1 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 0 ] ) ;
                        parity [ 24 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 26 Codewords
int pcsr_26vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 26 ], taps [ 13 ] ;          // Parity registers
        __m512i data_vec, temp [ 13 ] ;
        unsigned char **sPnt, *pPnt [ 26 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 11 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 10 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 9 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 8 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 7 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 6 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 5 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 4 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 3 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 2 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 1 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 0 ] ) ;
                        parity [ 25 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 27 Codewords
int pcsr_27vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 27 ], taps [ 13 ] ;          // Parity registers
        __m512i data_vec, temp [ 13 ] ;
        unsigned char **sPnt, *pPnt [ 27 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;
        pPnt [ 26 ] = dest [ 26 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;
                parity [ 26 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 12 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 11 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 10 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 9 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 8 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 7 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 6 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 5 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 4 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 3 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 2 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 1 ] ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 0 ] ) ;
                        parity [ 26 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 26 ] + curPos), parity [ 26 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 28 Codewords
int pcsr_28vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 28 ], taps [ 14 ] ;          // Parity registers
        __m512i data_vec, temp [ 14 ] ;
        unsigned char **sPnt, *pPnt [ 28 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;
        pPnt [ 26 ] = dest [ 26 ] ;
        pPnt [ 27 ] = dest [ 27 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;
                parity [ 26 ] = _mm512_setzero_si512 () ;
                parity [ 27 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 12 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 11 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 10 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 9 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 8 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 7 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 6 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 5 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 4 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 3 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 2 ] ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 1 ] ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 0 ] ) ;
                        parity [ 27 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 26 ] + curPos), parity [ 26 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 27 ] + curPos), parity [ 27 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 29 Codewords
int pcsr_29vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 29 ], taps [ 14 ] ;          // Parity registers
        __m512i data_vec, temp [ 14 ] ;
        unsigned char **sPnt, *pPnt [ 29 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;
        pPnt [ 26 ] = dest [ 26 ] ;
        pPnt [ 27 ] = dest [ 27 ] ;
        pPnt [ 28 ] = dest [ 28 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;
                parity [ 26 ] = _mm512_setzero_si512 () ;
                parity [ 27 ] = _mm512_setzero_si512 () ;
                parity [ 28 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 13 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 12 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 11 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 10 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 9 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 8 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 7 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 6 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 5 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 4 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 3 ] ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 2 ] ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 1 ] ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 0 ] ) ;
                        parity [ 28 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 26 ] + curPos), parity [ 26 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 27 ] + curPos), parity [ 27 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 28 ] + curPos), parity [ 28 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 30 Codewords
int pcsr_30vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 30 ], taps [ 15 ] ;          // Parity registers
        __m512i data_vec, temp [ 15 ] ;
        unsigned char **sPnt, *pPnt [ 30 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;
        pPnt [ 26 ] = dest [ 26 ] ;
        pPnt [ 27 ] = dest [ 27 ] ;
        pPnt [ 28 ] = dest [ 28 ] ;
        pPnt [ 29 ] = dest [ 29 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;
                parity [ 26 ] = _mm512_setzero_si512 () ;
                parity [ 27 ] = _mm512_setzero_si512 () ;
                parity [ 28 ] = _mm512_setzero_si512 () ;
                parity [ 29 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
                        temp [ 14 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 14 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 14 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 13 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 12 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 11 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 10 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 9 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 8 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 7 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 6 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 5 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 4 ] ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 3 ] ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 2 ] ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 1 ] ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ], temp [ 0 ] ) ;
                        parity [ 29 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 26 ] + curPos), parity [ 26 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 27 ] + curPos), parity [ 27 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 28 ] + curPos), parity [ 28 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 29 ] + curPos), parity [ 29 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 31 Codewords
int pcsr_31vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 31 ], taps [ 15 ] ;          // Parity registers
        __m512i data_vec, temp [ 15 ] ;
        unsigned char **sPnt, *pPnt [ 31 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;
        pPnt [ 26 ] = dest [ 26 ] ;
        pPnt [ 27 ] = dest [ 27 ] ;
        pPnt [ 28 ] = dest [ 28 ] ;
        pPnt [ 29 ] = dest [ 29 ] ;
        pPnt [ 30 ] = dest [ 30 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;
                parity [ 26 ] = _mm512_setzero_si512 () ;
                parity [ 27 ] = _mm512_setzero_si512 () ;
                parity [ 28 ] = _mm512_setzero_si512 () ;
                parity [ 29 ] = _mm512_setzero_si512 () ;
                parity [ 30 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
                        temp [ 14 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 14 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 14 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 14 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 13 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 12 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 11 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 10 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 9 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 8 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 7 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 6 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 5 ] ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 4 ] ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 3 ] ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 2 ] ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ], temp [ 1 ] ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ], temp [ 0 ] ) ;
                        parity [ 30 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 26 ] + curPos), parity [ 26 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 27 ] + curPos), parity [ 27 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 28 ] + curPos), parity [ 28 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 29 ] + curPos), parity [ 29 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 30 ] + curPos), parity [ 30 ] ) ;
        }
        return ( curPos ) ;
}

// Parallel LFSR_SR Sequencer for P = 32 Codewords
int pcsr_32vect_pls_avx512_gfni(int len, int k, unsigned char *afftab, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 32 ], taps [ 16 ] ;          // Parity registers
        __m512i data_vec, temp [ 16 ] ;
        unsigned char **sPnt, *pPnt [ 32 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;
        pPnt [ 2 ] = dest [ 2 ] ;
        pPnt [ 3 ] = dest [ 3 ] ;
        pPnt [ 4 ] = dest [ 4 ] ;
        pPnt [ 5 ] = dest [ 5 ] ;
        pPnt [ 6 ] = dest [ 6 ] ;
        pPnt [ 7 ] = dest [ 7 ] ;
        pPnt [ 8 ] = dest [ 8 ] ;
        pPnt [ 9 ] = dest [ 9 ] ;
        pPnt [ 10 ] = dest [ 10 ] ;
        pPnt [ 11 ] = dest [ 11 ] ;
        pPnt [ 12 ] = dest [ 12 ] ;
        pPnt [ 13 ] = dest [ 13 ] ;
        pPnt [ 14 ] = dest [ 14 ] ;
        pPnt [ 15 ] = dest [ 15 ] ;
        pPnt [ 16 ] = dest [ 16 ] ;
        pPnt [ 17 ] = dest [ 17 ] ;
        pPnt [ 18 ] = dest [ 18 ] ;
        pPnt [ 19 ] = dest [ 19 ] ;
        pPnt [ 20 ] = dest [ 20 ] ;
        pPnt [ 21 ] = dest [ 21 ] ;
        pPnt [ 22 ] = dest [ 22 ] ;
        pPnt [ 23 ] = dest [ 23 ] ;
        pPnt [ 24 ] = dest [ 24 ] ;
        pPnt [ 25 ] = dest [ 25 ] ;
        pPnt [ 26 ] = dest [ 26 ] ;
        pPnt [ 27 ] = dest [ 27 ] ;
        pPnt [ 28 ] = dest [ 28 ] ;
        pPnt [ 29 ] = dest [ 29 ] ;
        pPnt [ 30 ] = dest [ 30 ] ;
        pPnt [ 31 ] = dest [ 31 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 8 * 8 ) ) );
        taps [ 9 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 9 * 8 ) ) );
        taps [ 10 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 10 * 8 ) ) );
        taps [ 11 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 11 * 8 ) ) );
        taps [ 12 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 12 * 8 ) ) );
        taps [ 13 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 13 * 8 ) ) );
        taps [ 14 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 14 * 8 ) ) );
        taps [ 15 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( afftab + ( 15 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                sPnt = data ;
                parity [ 0 ] = _mm512_setzero_si512 () ;
                parity [ 1 ] = _mm512_setzero_si512 () ;
                parity [ 2 ] = _mm512_setzero_si512 () ;
                parity [ 3 ] = _mm512_setzero_si512 () ;
                parity [ 4 ] = _mm512_setzero_si512 () ;
                parity [ 5 ] = _mm512_setzero_si512 () ;
                parity [ 6 ] = _mm512_setzero_si512 () ;
                parity [ 7 ] = _mm512_setzero_si512 () ;
                parity [ 8 ] = _mm512_setzero_si512 () ;
                parity [ 9 ] = _mm512_setzero_si512 () ;
                parity [ 10 ] = _mm512_setzero_si512 () ;
                parity [ 11 ] = _mm512_setzero_si512 () ;
                parity [ 12 ] = _mm512_setzero_si512 () ;
                parity [ 13 ] = _mm512_setzero_si512 () ;
                parity [ 14 ] = _mm512_setzero_si512 () ;
                parity [ 15 ] = _mm512_setzero_si512 () ;
                parity [ 16 ] = _mm512_setzero_si512 () ;
                parity [ 17 ] = _mm512_setzero_si512 () ;
                parity [ 18 ] = _mm512_setzero_si512 () ;
                parity [ 19 ] = _mm512_setzero_si512 () ;
                parity [ 20 ] = _mm512_setzero_si512 () ;
                parity [ 21 ] = _mm512_setzero_si512 () ;
                parity [ 22 ] = _mm512_setzero_si512 () ;
                parity [ 23 ] = _mm512_setzero_si512 () ;
                parity [ 24 ] = _mm512_setzero_si512 () ;
                parity [ 25 ] = _mm512_setzero_si512 () ;
                parity [ 26 ] = _mm512_setzero_si512 () ;
                parity [ 27 ] = _mm512_setzero_si512 () ;
                parity [ 28 ] = _mm512_setzero_si512 () ;
                parity [ 29 ] = _mm512_setzero_si512 () ;
                parity [ 30 ] = _mm512_setzero_si512 () ;
                parity [ 31 ] = _mm512_setzero_si512 () ;

                // Loop through all the 0..k symbols
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_stream_load_si512( *sPnt + curPos ) ;
                      __builtin_prefetch ( *sPnt + curPos + 64, 0, 3 ) ;
                        sPnt ++ ;
                        // Add incoming data to MSB of parity, then update parities using Parallel Multiplier
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        temp [ 0 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 0 ], 0 ) ;
                        temp [ 1 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 1 ], 0 ) ;
                        temp [ 2 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 2 ], 0 ) ;
                        temp [ 3 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 3 ], 0 ) ;
                        temp [ 4 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 4 ], 0 ) ;
                        temp [ 5 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 5 ], 0 ) ;
                        temp [ 6 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 6 ], 0 ) ;
                        temp [ 7 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 7 ], 0 ) ;
                        temp [ 8 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 8 ], 0 ) ;
                        temp [ 9 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 9 ], 0 ) ;
                        temp [ 10 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 10 ], 0 ) ;
                        temp [ 11 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 11 ], 0 ) ;
                        temp [ 12 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 12 ], 0 ) ;
                        temp [ 13 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 13 ], 0 ) ;
                        temp [ 14 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 14 ], 0 ) ;
                        temp [ 15 ] = _mm512_gf2p8affine_epi64_epi8 ( data_vec, taps [ 15 ], 0 ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ], temp [ 0 ] ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ], temp [ 1 ] ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ], temp [ 2 ] ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ], temp [ 3 ] ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ], temp [ 4 ] ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ], temp [ 5 ] ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ], temp [ 6 ] ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ], temp [ 7 ] ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ], temp [ 8 ] ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ], temp [ 9 ] ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ], temp [ 10 ] ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ], temp [ 11 ] ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ], temp [ 12 ] ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ], temp [ 13 ] ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ], temp [ 14 ] ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ], temp [ 15 ] ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ], temp [ 14 ] ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ], temp [ 13 ] ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ], temp [ 12 ] ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ], temp [ 11 ] ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ], temp [ 10 ] ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ], temp [ 9 ] ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ], temp [ 8 ] ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ], temp [ 7 ] ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ], temp [ 6 ] ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ], temp [ 5 ] ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ], temp [ 4 ] ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ], temp [ 3 ] ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ], temp [ 2 ] ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ], temp [ 1 ] ) ;
                        parity [ 30 ] = _mm512_xor_si512 ( parity [ 31 ], temp [ 0 ] ) ;
                        parity [ 31 ] = data_vec ;
                }

                 // Store parity back to memory
                _mm512_stream_si512( (__m512i *) (pPnt [ 0 ] + curPos), parity [ 0 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 1 ] + curPos), parity [ 1 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 2 ] + curPos), parity [ 2 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 3 ] + curPos), parity [ 3 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 4 ] + curPos), parity [ 4 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 5 ] + curPos), parity [ 5 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 6 ] + curPos), parity [ 6 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 7 ] + curPos), parity [ 7 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 8 ] + curPos), parity [ 8 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 9 ] + curPos), parity [ 9 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 10 ] + curPos), parity [ 10 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 11 ] + curPos), parity [ 11 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 12 ] + curPos), parity [ 12 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 13 ] + curPos), parity [ 13 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 14 ] + curPos), parity [ 14 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 15 ] + curPos), parity [ 15 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 16 ] + curPos), parity [ 16 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 17 ] + curPos), parity [ 17 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 18 ] + curPos), parity [ 18 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 19 ] + curPos), parity [ 19 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 20 ] + curPos), parity [ 20 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 21 ] + curPos), parity [ 21 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 22 ] + curPos), parity [ 22 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 23 ] + curPos), parity [ 23 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 24 ] + curPos), parity [ 24 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 25 ] + curPos), parity [ 25 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 26 ] + curPos), parity [ 26 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 27 ] + curPos), parity [ 27 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 28 ] + curPos), parity [ 28 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 29 ] + curPos), parity [ 29 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 30 ] + curPos), parity [ 30 ] ) ;
                _mm512_stream_si512( (__m512i *) (pPnt [ 31 ] + curPos), parity [ 31 ] ) ;
        }
        return ( curPos ) ;
}

// Single function to access each unrolled Encode
void pcsr_encode_data_avx512_gfni(int len, int k, int rows, unsigned char *afftab, unsigned char **data,
        unsigned char **coding)
{
        switch (rows) {
        case 2: pcsr_2vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 3: pcsr_3vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 4: pcsr_4vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 5: pcsr_5vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 6: pcsr_6vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 7: pcsr_7vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 8: pcsr_8vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 9: pcsr_9vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 10: pcsr_10vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 11: pcsr_11vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 12: pcsr_12vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 13: pcsr_13vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 14: pcsr_14vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 15: pcsr_15vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 16: pcsr_16vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 17: pcsr_17vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 18: pcsr_18vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 19: pcsr_19vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 20: pcsr_20vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 21: pcsr_21vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 22: pcsr_22vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 23: pcsr_23vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 24: pcsr_24vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 25: pcsr_25vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 26: pcsr_26vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 27: pcsr_27vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 28: pcsr_28vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 29: pcsr_29vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 30: pcsr_30vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 31: pcsr_31vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        case 32: pcsr_32vect_pls_avx512_gfni(len, k, afftab, data, coding);
                 break ;
        }
}
// Single function to access each unrolled Decode
int pcsr_decode_data_avx512_gfni(int len, int k, int rows, unsigned char *afftab, unsigned char **data,
        unsigned char **coding, int retries)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < retries ) )
        {

                switch (rows) {
                case 2: newPos = pcsr_2vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 3: newPos = pcsr_3vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 4: newPos = pcsr_4vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 5: newPos = pcsr_5vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 6: newPos = pcsr_6vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 7: newPos = pcsr_7vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 8: newPos = pcsr_8vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 9: newPos = pcsr_9vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 10: newPos = pcsr_10vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 11: newPos = pcsr_11vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 12: newPos = pcsr_12vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 13: newPos = pcsr_13vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 14: newPos = pcsr_14vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 15: newPos = pcsr_15vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 16: newPos = pcsr_16vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 17: newPos = pcsr_17vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 18: newPos = pcsr_18vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 19: newPos = pcsr_19vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 20: newPos = pcsr_20vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 21: newPos = pcsr_21vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 22: newPos = pcsr_22vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 23: newPos = pcsr_23vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 24: newPos = pcsr_24vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 25: newPos = pcsr_25vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 26: newPos = pcsr_26vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 27: newPos = pcsr_27vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 28: newPos = pcsr_28vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 29: newPos = pcsr_29vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 30: newPos = pcsr_30vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 31: newPos = pcsr_31vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                case 32: newPos = pcsr_32vect_pss_avx512_gfni(len, k, afftab, data, coding, newPos);
                         break ;
                }
                if ( newPos < len )
                {
                        if ( pcsr_correct_AVX512_GFNI ( newPos, k, rows, data, coding, 64 ) )
                        {
                                return ( newPos ) ;
                        }

                }
        }
        return ( newPos ) ;
}
