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

static unsigned char pc_ptab_1b [ 256 ], pc_ltab_1b [ 256 ], pc_itab_1b [ 256 ] ;
static __m512i EncMat [ 255 ] [ 4 ] ;
static __m512i Vand1b [ 255 ] [ 4 ] ;

#define L1Enc(vec,p,pvec) \
for ( int curP = 0 ; curP < p ; curP ++ ) \
{ \
        matVec = _mm512_load_si512 ( &EncMat [ curP ] [ 3 ]) ; \
        vreg = _mm512_gf2p8mul_epi8 ( vec, matVec ) ; \
        __m256i low = _mm512_castsi512_si256(vreg); \
        __m256i high = _mm512_extracti64x4_epi64(vreg, 1); \
        __m256i xored = _mm256_xor_si256(low, high); \
        __m128i low128 = _mm256_castsi256_si128(xored); \
        __m128i high128 = _mm256_extracti128_si256(xored, 1); \
        __m128i xored128 = _mm_xor_si128(low128, high128); \
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2)); \
        __m128i xored64 = _mm_xor_si128(xored128, perm); \
        uint64_t result_64 = _mm_cvtsi128_si64(xored64); \
        result_64 ^= result_64 >> 32 ; \
        result_64 ^= result_64 >> 16 ; \
        result_64 ^= result_64 >> 8 ; \
        pp [ curP ] = ( unsigned char ) result_64 ; \
} \
vec = _mm512_mask_blend_epi32 ( 0x8000, vec, pvec ) ; 

#define L1Dec(vec,p,err,syn) \
for ( int curP = 0 ; curP < p ; curP ++ ) \
{ \
        matVec = _mm512_load_si512 ( &Vand1b [ curP ] [ 3 ]) ; \
        vreg = _mm512_gf2p8mul_epi8 ( vec, matVec ) ; \
        __m256i low = _mm512_castsi512_si256(vreg); \
        __m256i high = _mm512_extracti64x4_epi64(vreg, 1); \
        __m256i xored = _mm256_xor_si256(low, high); \
        __m128i low128 = _mm256_castsi256_si128(xored); \
        __m128i high128 = _mm256_extracti128_si256(xored, 1); \
        __m128i xored128 = _mm_xor_si128(low128, high128); \
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2)); \
        __m128i xored64 = _mm_xor_si128(xored128, perm); \
        uint64_t result_64 = _mm_cvtsi128_si64(xored64); \
        result_64 ^= result_64 >> 32 ; \
        result_64 ^= result_64 >> 16 ; \
        result_64 ^= result_64 >> 8 ; \
        syn [ curP ] = ( unsigned char ) result_64 ; \
        if ( syn [ curP ] != 0 ) err = 1 ; \
} 

// Multiply two bytes using the hardware GF multiply
unsigned char pc_mul_1b ( unsigned char a, unsigned char b) 
{
    __m128i va, vb ;

    unsigned char * veca = ( unsigned char * ) &va ;
    unsigned char * vecb = ( unsigned char * ) &vb ;
    *veca = a ;
    *vecb = b ;

    va = _mm_gf2p8mul_epi8 ( va, vb ) ;
    return *veca ;
}

// pc_bpow - Build a table of power values
void pc_bpow_1b ( unsigned char Gen )
{
    int i ;


    // A positive integer raised to the power 0 is one
    pc_ptab_1b [ 0 ] = 1 ;

    // Two is a good generator for 0x1d, three is a good generator for 0x1b
    for ( i = 1 ; i < 256 ; i ++ )
    {
        pc_ptab_1b[ i ] = pc_mul_1b ( pc_ptab_1b [ i - 1 ], Gen ) ;
    }
}

// pc_blog - Use the power table to build the log table
void pc_blog_1b ( void ) 
{
    int i ;


    // Use the power table to index into the log table and store log value
    for ( i = 0 ; i < 256 ; i ++ )
    {
        pc_ltab_1b [ pc_ptab_1b [ i ] ] = i ;
    }
}

// pc_linv - Calculate the inverse of a number, that is, 1/Number
void  pc_binv_1b ( void ) 
{
    int i ;
    for ( i = 0 ; i < 256 ; i ++ )
    {
        pc_itab_1b [ i ] = pc_ptab_1b [ 255 - pc_ltab_1b [ i ] ] ;
    }
}

void pc_gen_rsr_matrix_1b(unsigned char *a, int k)
{
        int i, j;
        unsigned char p, gen = 1;

        // Loop through rows and cols backward
        for (i = k-1; i>=0; i--)
        {
                p = 1;
                for ( j = 0; j < 255 ; j++ )
                {
                        int idx = ( 255 * i ) + ( 255 - j - 1 ) ;
                        a [ idx ] = p;
                        p = pc_mul_1b(p, gen);
                }
                gen = pc_mul_1b(gen, 3);
                //printf ( "Vand row %d\n", i ) ;
                //dump_u8xu8 ( &a[ 255*i ], 16, 16 ) ;
        }
}

// Initialize encoding matrix for encoding
void pc_bmat_1b ( unsigned char * vals, int p )
{
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        unsigned char * eDest = ( unsigned char * ) &EncMat [ curP ] ;
        eDest ++ ;
        memcpy ( ( unsigned char * ) eDest, &vals [ curP * ( 255 - p ) ], 255 - p ) ;
        unsigned char * extra = ( unsigned char * ) &EncMat [ curP ] ;
        memset ( extra + 256 - p, 0, p ) ;
        //printf ( "Encmat %d\n", curP ) ;
        //dump_u8xu8 ( ( unsigned char * ) &EncMat [ curP ] [ 3 ], 4, 16 ) ;
    }
}

// Initialize vandermonde matrix for decoding
void pc_bvan_1b ( unsigned char * vals, int p )
{
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        unsigned char * eDest = ( unsigned char * ) &Vand1b [ curP ] ;
        *eDest = 0 ;
        eDest ++ ;
        memcpy (  ( unsigned char * ) eDest, &vals [ curP * ( 255 ) ], 255 ) ;
        //printf ( "Vand1b %d\n", curP ) ;
        //dump_u8xu8 ( ( unsigned char * ) &Vand1b [ curP ] [ 3 ], 4, 16 ) ;
    }
}

// Encode using the Vandermonde matrix
void pc_encoder1b ( unsigned char * codeWord, unsigned char * par, int p ) 
{
    __m512i codeWordvec [ 4 ], encMatvec [ 4 ], vreg [ 4 ] ;

    // Load the entire codeword into 4 vector registers
    codeWordvec [ 0 ] = _mm512_loadu_si512 ( codeWord + 0 * 64 ) ;
    codeWordvec [ 1 ] = _mm512_loadu_si512 ( codeWord + 1 * 64 ) ;
    codeWordvec [ 2 ] = _mm512_loadu_si512 ( codeWord + 2 * 64 ) ;
    codeWordvec [ 3 ] = _mm512_loadu_si512 ( codeWord + 3 * 64 ) ;
    //printf ( "Codeword\n" ) ;
    //dump_u8xu8 ( ( unsigned char * ) &codeWordvec [ 0 ], 1, 255 ) ;

    // Now loop and compute each parity using the encoding matrix
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        //printf ( "Encmat\n" ) ;
        //dump_u8xu8 ( ( unsigned char * ) &EncMat [ curP ] [ 0 ], 1, 255 ) ;

        // Load one row of the encoding matrix into vector registers
        encMatvec [ 0 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 0 ] ) ;
        encMatvec [ 1 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 1 ] ) ;
        encMatvec [ 2 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 2 ] ) ;
        encMatvec [ 3 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 3 ] ) ;

        // Multiply the codeword by the encoding matrix
        vreg [ 0 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 0 ], encMatvec [ 0 ] ) ;
        vreg [ 1 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 1 ], encMatvec [ 1 ] ) ;
        vreg [ 2 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 2 ], encMatvec [ 2 ] ) ;
        vreg [ 3 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 3 ], encMatvec [ 3 ] ) ;

        // Now collapse the 255 symbols down to 1
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 1 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 2 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 3 ] ) ;

        // Shuffle and XOR 512-bit to 256-bit
        __m256i low = _mm512_castsi512_si256(vreg [ 0 ] );
        __m256i high = _mm512_extracti64x4_epi64(vreg [ 0 ], 1);
        __m256i xored = _mm256_xor_si256(low, high);

        // Shuffle and XOR 256-bit to 128-bit
        __m128i low128 = _mm256_castsi256_si128(xored);
        __m128i high128 = _mm256_extracti128_si256(xored, 1);
        __m128i xored128 = _mm_xor_si128(low128, high128);

        // Shuffle 128-bit to 64-bit using permute
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i xored64 = _mm_xor_si128(xored128, perm);

        // Reduce 64-bit to 32-bit
        uint64_t result_64 = _mm_cvtsi128_si64(xored64);
        result_64 ^= result_64 >> 32 ;
        result_64 ^= result_64 >> 16 ;
        result_64 ^= result_64 >> 8 ;
        par [ curP ] = ( unsigned char ) result_64 ;
        //printf ( "Par [ %d ] = %d\n", curP, par [ curP ] ) ;
    }
}

// Encode using the Vandermonde matrix
void pc_decoder1b ( unsigned char * codeWord, unsigned char * syn, int p ) 
{
    __m512i codeWordvec [ 4 ], vanMatvec [ 4 ], vreg [ 4 ] ;

    // Load the whole codeword into vector registers
    codeWordvec [ 0 ] = _mm512_loadu_si512 ( codeWord + 0 * 64 ) ;
    codeWordvec [ 1 ] = _mm512_loadu_si512 ( codeWord + 1 * 64 ) ;
    codeWordvec [ 2 ] = _mm512_loadu_si512 ( codeWord + 2 * 64 ) ;
    codeWordvec [ 3 ] = _mm512_loadu_si512 ( codeWord + 3 * 64 ) ;
    //printf ( "Codeword LSB\n" ) ;
    //dump_u8xu8 ( ( unsigned char * ) &codeWordvec [ 0 ], 1, 255 ) ;

    // Loop through each decoding vector of Vandermonde
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        //printf ( "curP = %d\n", curP ) ;
        //printf ( "Vandermonde\n" ) ;
        //dump_u8xu8 ( ( unsigned char * ) &Vand1b [ curP ] [ 0 ], 1, 255 ) ;

        // Load an entire row from the Vandermonde matrix
        vanMatvec [ 0 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 0 ]) ;
        vanMatvec [ 1 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 1 ] ) ;
        vanMatvec [ 2 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 2 ] ) ;
        vanMatvec [ 3 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 3 ] ) ;

        // Multiply the codeword by the entire row
        vreg [ 0 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 0 ], vanMatvec [ 0 ] ) ;
        vreg [ 1 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 1 ], vanMatvec [ 1 ] ) ;
        vreg [ 2 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 2 ], vanMatvec [ 2 ] ) ;
        vreg [ 3 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 3 ], vanMatvec [ 3 ] ) ;
        //printf ( "VReg\n" ) ;
        //dump_u8xu8 ( (unsigned char *) vreg, 1, 255 ) ;

        // Now collapse the 255 symbols down to 1
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 1 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 2 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 3 ] ) ;

        // Shuffle and XOR 512-bit to 256-bit
        __m256i low = _mm512_castsi512_si256(vreg [ 0 ] );
        __m256i high = _mm512_extracti64x4_epi64(vreg [ 0 ], 1);
        __m256i xored = _mm256_xor_si256(low, high);

        // Shuffle and XOR 256-bit to 128-bit
        __m128i low128 = _mm256_castsi256_si128(xored);
        __m128i high128 = _mm256_extracti128_si256(xored, 1);
        __m128i xored128 = _mm_xor_si128(low128, high128);

        // Shuffle 128-bit to 64-bit using permute
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i xored64 = _mm_xor_si128(xored128, perm);

        // Reduce 64-bit to 32-bit
        uint64_t result_64 = _mm_cvtsi128_si64(xored64);
        result_64 ^= result_64 >> 32 ;
        result_64 ^= result_64 >> 16 ;
        result_64 ^= result_64 >> 8 ;
        syn [ curP ] = ( unsigned char ) result_64 ;
    }
}

// Identify roots from key equation
int find_roots_1b ( unsigned char * keyEq, unsigned char * roots, int mSize )
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
                        eVal = pc_mul_1b ( eVal, baseVal ) ;
                        eVal = eVal ^ keyEq [ mSize - j - 1 ] ;
                }
                // Check for a good root
                if ( eVal == 0 )
                {
                        roots [ rootCount ] = i ;
                        rootCount ++ ;
                }
                // Next evaluation is at the next power of 3
                baseVal = gf_mul ( baseVal, 3 ) ;
        }
        return rootCount ;
}


void pc_gen_poly_1b( unsigned char *p, int rank)
{
        int c, alpha, cr ; // Loop variables

        p [ 0 ] = 1 ; // Start with (x+1)
        alpha = 3 ;
        for ( cr = 1 ; cr < rank ; cr ++ ) // Loop rank-1 times
        {
                // Compute the last term of the polynomial by multiplying
                p [ cr ] = pc_mul_1b ( p [ cr - 1 ], alpha ) ;

                // Pass the middle terms to produce multiply result
                for ( c = cr - 1 ; c > 0 ; c -- )
                {
                        p [ c ] ^= pc_mul_1b ( p [ c - 1 ], alpha ) ;
                }

                // Compute the first term by adding in alphaI
                p [ 0 ] ^= alpha ;

                // Compute next alpha (power of 2)
                alpha = pc_mul_1b ( alpha, 3 ) ;
        }
}

void pc_gen_poly_matrix_1b(unsigned char *a, int m, int k)
{
        int i, j, par, over, lpos ;
        unsigned char *p, taps [ 254 ], lfsr [ 254 ] ;

        // First compute the generator polynomial and initialize the taps
        par = m - k ;

        pc_gen_poly_1b ( taps, par ) ;

        memcpy ( lfsr, taps, par ) ; // Initial value of LFSR is the taps

        // Now use an LFSR to build the values
        p = a ;
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
                        lfsr [ lpos ] = pc_mul_1b ( over, taps [ lpos ] ) ^ lfsr [ lpos + 1 ] ;
                }
                // Now do the LSB of the LFSR to finish
                lfsr [ par - 1 ] = pc_mul_1b ( over, taps [ par - 1 ] ) ;
        }
    }

unsigned char gf_div_1b_AVX512_GFNI ( unsigned char a, unsigned char b )
{
        return pc_mul_1b ( a, pc_itab_1b [ b ] ) ;
}

// Compute base ^ Power
int pc_pow_1b_AVX512_GFNI ( unsigned char base, unsigned char Power )
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
                computedPow = pc_mul_1b ( computedPow, base ) ;
        }
        return computedPow ;
}

// Assume there is a single error and try to correct, see if syndromes match
int pc_verify_single_error_1b_AVX512_GFNI ( unsigned char * S, unsigned char ** data, int k, int p,
        int newPos, int offSet )
{
        // LSB has parity, for single error this equals error value
        unsigned char eVal = S [ 0 ] ;

        // Compute error location is log2(syndrome[1]/syndrome[0])
        unsigned char eLoc = S [ 1 ] ;
        unsigned char pVal = pc_mul_1b ( eLoc, pc_itab_1b [ eVal ] ) ;
        eLoc = pc_ltab_1b [ pVal ] % 255 ;

        //printf ( "Eloc = %d\n", eLoc ) ;

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
                        if ( pc_mul_1b ( S [ i - 1 ], pVal ) != S [ i ] )
                        {
                                return 0 ;
                        }
                }
        }
        // Good correction
        data [ k - eLoc - 1 ] [ newPos + offSet ] ^= eVal ;
        return 1 ;
}

int gf_invert_matrix_1b_AVX512_GFNI(unsigned char *in_mat, unsigned char *out_mat, const int n)
{
        __m512i multVal512 ;

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
                                //printf ( "Singular\n" ) ;
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
                unsigned char temp_scalar = pc_itab_1b [ pivot ] ;
                //printf ( "Scalar = %d\n", temp_scalar ) ;

                // Scale row i by 1/pivot using GFNI affine
                multVal512 = _mm512_set1_epi8 ( temp_scalar ) ;
                aug_rows [ i ]  = _mm512_gf2p8mul_epi8 ( aug_rows[ i ], multVal512 ) ;

                // Eliminate in other rows
                for ( j = 0; j < n; j++ )
                {
                        if ( j == i)  continue;
                        unsigned char factor = matrix_mem[j * 64 + i];
                        // Compute scaled pivot row: pivot_row * factor
                        multVal512 = _mm512_set1_epi8 ( factor ) ;
                        __m512i scaled = _mm512_gf2p8mul_epi8 ( aug_rows [ i ], multVal512 ) ;
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

int find_roots_1b_AVX512_GFNI(unsigned char *keyEq, unsigned char *roots, int mSize)
{
        static __m512i Vandermonde [ 16 ] [ 4 ] ;
        __m512i sum [ 4 ], temp, multVal512 ;
        int i, j ;

        unsigned char * vVal = ( unsigned char *) Vandermonde ;
        // Check to see if Vandermonde has been initialized yet
        if ( vVal [ 0 ] == 0 )
        {
                unsigned char base = 3, cVal = 1 ;
                for (  i = 0 ; i < 16 ; i ++ )
                {
                        vVal = ( unsigned char * ) &Vandermonde [ i ] ;
                        for ( j = 0 ; j < 255 ; j ++ )
                        {
                                vVal [ j ] = cVal ;
                                cVal = pc_mul_1b ( cVal, base ) ;
                        }
                        base = pc_mul_1b ( base, 3 ) ;
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
                multVal512 = _mm512_set1_epi8 ( keyEq [ i ] ) ;
                // Remember that we did not build the first row of Vandermonde, so use i-1
                temp = _mm512_gf2p8mul_epi8 ( Vandermonde [ i-1 ] [ 0 ], multVal512 ) ;
                sum [ 0 ] = _mm512_xor_si512 ( sum [ 0 ], temp ) ;
                temp = _mm512_gf2p8mul_epi8 ( Vandermonde [ i-1 ] [ 1 ], multVal512 ) ;
                sum [ 1 ] = _mm512_xor_si512 ( sum [ 1 ], temp ) ;
                temp = _mm512_gf2p8mul_epi8 ( Vandermonde [ i-1 ] [ 2 ], multVal512 ) ;
                sum [ 2 ] = _mm512_xor_si512 ( sum [ 2 ], temp ) ;
                temp = _mm512_gf2p8mul_epi8 ( Vandermonde [ i-1 ] [ 3 ], multVal512 ) ;
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
                rootCount += _mm_popcnt_u64(mask);

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
int pc_compute_error_values_1b_AVX512_GFNI ( int mSize, unsigned char * S, unsigned char * roots,
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
        unsigned char base = 3 ;
        for ( i = 1 ; i < mSize ; i ++ )
        {
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        Mat [ i * mSize + j ] = pc_pow_1b_AVX512_GFNI ( base, roots [ j ] ) ;
                }
                base = pc_mul_1b ( base, 3 ) ;
        }
        // Invert matrix and verify inversion
        if ( gf_invert_matrix_1b_AVX512_GFNI ( Mat, Mat_inv, mSize ) != 0 )
        {
                return 0 ;
        }

        // Compute error values by summing Syndrome terms across inverted Vandermonde
        for ( i = 0 ; i < mSize ; i ++ )
        {
                errVal [ i ] = 0 ;
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        errVal [ i ] ^= pc_mul_1b ( S [ j ], Mat_inv [ i * mSize + j ] ) ;
                }
        }
        return 1 ;
}

// Verify proposed data values and locations can generate syndromes
int pc_verify_syndromes_1b_AVX512_GFNI ( unsigned char * S, int p, int mSize, unsigned char * roots,
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
                        unsigned char termVal = pc_mul_1b ( errVal [ j ], pc_pow_1b_AVX512_GFNI ( base, roots [ j ] ) ) ;
                        sum ^= termVal ;
                }

                // Verify we reproduced the syndrome
                if ( sum != S [ i ] )
                {
                        printf ( "S\n" ) ;
                        dump_u8xu8 ( S, 1, mSize ) ;
                        printf ( "ErrVals\n" ) ;
                        dump_u8xu8 ( errVal, 1, mSize ) ;
                        printf ( "Roots\n" ) ;
                        dump_u8xu8 ( roots, 1, mSize ) ;
                        printf ( "Sum = %x S [ %d ] = %x\n", sum, i, S [ i ] ) ;
                        return 0 ;
                }
                // Move to next syndrome
                base = pc_mul_1b ( base, 3 ) ;
        }
        return 1 ;
}

// Affine table from ec_base.h: 256 * 8-byte matrices for GF(256) multiplication
static const uint64_t gf_table_gfni[256];  // Assume defined in ec_base.h

// syndromes: array of length 'length' (typically 2t), syndromes[0] = S1, [1] = S2, etc.
// lambda: caller-allocated array of size at least (length + 1 + 31), filled with locator poly coeffs. Padded for SIMD.
// Returns: degree L of the error locator polynomial.
// Note: Assumes length <= 32 for AVX-512 (32-byte vectors); extend loops for larger lengths.
int berlekamp_massey_1b_AVX512_GFNI(unsigned char *syndromes, int length, unsigned char *lambda)
{
    unsigned char b[PC_MAX_ERRS*2 + 1];  // Padded for AVX-512 (32-byte alignment)
    unsigned char temp[PC_MAX_ERRS*2 + 1];
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
                d ^= pc_mul_1b(lambda[j], syndromes[r - j]);
            }
        }

        if (d == 0)
        {
            m++;
        }
        else
        {
            unsigned char q = gf_div_1b_AVX512_GFNI(d, old_d);
            memcpy(temp, lambda, length + 1 + 31);

            // SIMD update: lambda[j + m] ^= gf_mul(q, b[j]) using AVX-512 GF2P8AFFINE
            // Load and broadcast 8-byte affine matrix for q
            __m256i matrix = _mm256_set1_epi8 (q);  // Broadcast to all 4 lanes
            __m256i b_vec = _mm256_loadu_si256((const __m256i *)b);
            // Perform GF(256) multiplication: result = affine(b_vec, matrix) + 0
            __m256i mul_res = _mm256_gf2p8mul_epi8(b_vec, matrix);
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
int pc_verify_multiple_errors_1b_AVX512_GFNI ( unsigned char * S, unsigned char ** data, int mSize, int k,
        int p, int newPos, int offSet, unsigned char * keyEq )
{
        unsigned char roots [ PC_MAX_ERRS ] = {0} ;
        unsigned char errVal [ PC_MAX_ERRS ] ;

        // Find roots, exit if mismatch with expected roots
        int nroots = find_roots_1b_AVX512_GFNI ( keyEq, roots, mSize );
        if ( nroots != mSize )
        {
                printf ( "Bad roots expected %d got %d\n", mSize, nroots ) ;
                return 0 ;
        }

        // Compute the error values
        if ( pc_compute_error_values_1b_AVX512_GFNI ( mSize, S, roots, errVal ) == 0 )
        {
                printf ( "Error values exit\n" ) ;
                return 0 ;
        }

        // Verify all syndromes are correct
        if ( pc_verify_syndromes_1b_AVX512_GFNI ( S, p, mSize, roots, errVal ) == 0 )
        {
                printf ( "Syndromes exit\n" ) ;
                return 0 ;
        }

        //printf ( "Before correction, roots, errval\n" ) ;
        //dump_u8xu8 ( roots, 1, 16 ) ;
        //dump_u8xu8 ( errVal, 1, 16 ) ;

        // Syndromes are OK, correct the user data
        for ( int i = 0 ; i < mSize ; i ++ )
        {
                int sym = k - roots [ i ] - 1 ;
                if ( ( sym < 0 ) || ( sym >= k ) )
                {
                    printf ( "Error correction k = %d Sym = %d newPos = %d offset = %d\n", k, sym, newPos, offSet ) ;
                    return 0 ;
                }
                data [ sym ] [ newPos + offSet ] ^= errVal [ i ] ;
        }
        // Good correction
        return 1 ;
}

// PGZ decoding step 1, see if we can invert the matrix, if so, compute key equation
int PGZ_1b_AVX512_GFNI ( unsigned char * S, int p, unsigned char * keyEq )
{
       unsigned char SMat [ PC_MAX_ERRS * PC_MAX_ERRS ], SMat_inv [ PC_MAX_ERRS * PC_MAX_ERRS ] ;
        int i,j ;

        memset ( keyEq, 0, p / 2 ) ;

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
                if ( gf_invert_matrix_1b_AVX512_GFNI ( SMat, SMat_inv, mSize ) == 0 )
                {
                        // Compute the key equation terms
                        for ( i = 0 ; i < mSize ; i ++ )
                        {
                                for ( j = 0 ; j < mSize ; j ++ )
                                {
                                        keyEq [ i ] ^= pc_mul_1b ( S [ mSize + j ], SMat_inv [ i * mSize + j ] ) ;
                                }
                        }
                        //printf ( "Found good matrix size %d\n", mSize ) ;
                        return mSize ;
                }
        }
        return 0 ;

}

// Syndromes are non-zero, try to calculate error location and data values
int pc_correct_AVX512_GFNI_1b ( int newPos, int k, int p,
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
        if ( pc_verify_single_error_1b_AVX512_GFNI ( S, data, k, p, newPos, offSet ) )
        {
                return 1 ;
        }

        mSize = PGZ_1b_AVX512_GFNI( S, p, keyEq ) ;

        if ( mSize > 1 )
        {
                return pc_verify_multiple_errors_1b_AVX512_GFNI ( S, data, mSize, k, p, newPos, offSet, keyEq ) ;
        }
        return 0 ;
}

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
int gf_2vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 2 ], taps [ 1 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
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

int gf_3vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 3 ], taps [ 2 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
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

int gf_4vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 4 ], taps [ 3 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
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

int gf_5vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 5 ], taps [ 4 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
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

int gf_6vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 6 ], taps [ 5 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
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

int gf_7vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 7 ], taps [ 6 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
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

int gf_8vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 8 ], taps [ 7 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
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

int gf_9vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 9 ], taps [ 8 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
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

int gf_10vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 10 ], taps [ 9 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
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

int gf_11vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 11 ], taps [ 10 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
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

int gf_12vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 12 ], taps [ 11 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
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

int gf_13vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 13 ], taps [ 12 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
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

int gf_14vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 14 ], taps [ 13 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
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

int gf_15vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 15 ], taps [ 14 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
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

int gf_16vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 16 ], taps [ 15 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
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

int gf_17vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 17 ], taps [ 16 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
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

int gf_18vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 18 ], taps [ 17 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
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

int gf_19vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 19 ], taps [ 18 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
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

int gf_20vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 20 ], taps [ 19 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
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

int gf_21vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 21 ], taps [ 20 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
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

int gf_22vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 22 ], taps [ 21 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
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

int gf_23vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 23 ], taps [ 22 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
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

int gf_24vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 24 ], taps [ 23 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
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

int gf_25vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 25 ], taps [ 24 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
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

int gf_26vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 26 ], taps [ 25 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
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

int gf_27vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 27 ], taps [ 26 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(parity [ 25 ], taps [ 25 ]) ;
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

int gf_28vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 28 ], taps [ 27 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(parity [ 25 ], taps [ 25 ]) ;
                        parity [ 26 ] = _mm512_gf2p8mul_epi8(parity [ 26 ], taps [ 26 ]) ;
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

int gf_29vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 29 ], taps [ 28 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(parity [ 25 ], taps [ 25 ]) ;
                        parity [ 26 ] = _mm512_gf2p8mul_epi8(parity [ 26 ], taps [ 26 ]) ;
                        parity [ 27 ] = _mm512_gf2p8mul_epi8(parity [ 27 ], taps [ 27 ]) ;
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

int gf_30vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 30 ], taps [ 29 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(parity [ 25 ], taps [ 25 ]) ;
                        parity [ 26 ] = _mm512_gf2p8mul_epi8(parity [ 26 ], taps [ 26 ]) ;
                        parity [ 27 ] = _mm512_gf2p8mul_epi8(parity [ 27 ], taps [ 27 ]) ;
                        parity [ 28 ] = _mm512_gf2p8mul_epi8(parity [ 28 ], taps [ 28 ]) ;
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

int gf_31vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 31 ], taps [ 30 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;
        taps [ 29 ] = _mm512_set1_epi8 ( g_tbls [ 29 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(parity [ 25 ], taps [ 25 ]) ;
                        parity [ 26 ] = _mm512_gf2p8mul_epi8(parity [ 26 ], taps [ 26 ]) ;
                        parity [ 27 ] = _mm512_gf2p8mul_epi8(parity [ 27 ], taps [ 27 ]) ;
                        parity [ 28 ] = _mm512_gf2p8mul_epi8(parity [ 28 ], taps [ 28 ]) ;
                        parity [ 29 ] = _mm512_gf2p8mul_epi8(parity [ 29 ], taps [ 29 ]) ;
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

int gf_32vect_pss_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        unsigned char err = 0, syn [ 4 ] ;
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 32 ], taps [ 31 ], matVec, vreg ;
        __m512i data_vec ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;
        taps [ 29 ] = _mm512_set1_epi8 ( g_tbls [ 29 ] ) ;
        taps [ 30 ] = _mm512_set1_epi8 ( g_tbls [ 30 ] ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 ) return curPos ;
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

                        L1Dec(data_vec, 4, err, syn ) ;
                        if ( err != 0 ) return curPos ;

                        parity [ 0 ] = _mm512_gf2p8mul_epi8(parity [ 0 ], taps [ 0 ]) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(parity [ 1 ], taps [ 1 ]) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(parity [ 2 ], taps [ 2 ]) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(parity [ 3 ], taps [ 3 ]) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(parity [ 4 ], taps [ 4 ]) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(parity [ 5 ], taps [ 5 ]) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(parity [ 6 ], taps [ 6 ]) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(parity [ 7 ], taps [ 7 ]) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(parity [ 8 ], taps [ 8 ]) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(parity [ 9 ], taps [ 9 ]) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(parity [ 10 ], taps [ 10 ]) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(parity [ 11 ], taps [ 11 ]) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(parity [ 12 ], taps [ 12 ]) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(parity [ 13 ], taps [ 13 ]) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(parity [ 14 ], taps [ 14 ]) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(parity [ 15 ], taps [ 15 ]) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(parity [ 16 ], taps [ 16 ]) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(parity [ 17 ], taps [ 17 ]) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(parity [ 18 ], taps [ 18 ]) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(parity [ 19 ], taps [ 19 ]) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(parity [ 20 ], taps [ 20 ]) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(parity [ 21 ], taps [ 21 ]) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(parity [ 22 ], taps [ 22 ]) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(parity [ 23 ], taps [ 23 ]) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(parity [ 24 ], taps [ 24 ]) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(parity [ 25 ], taps [ 25 ]) ;
                        parity [ 26 ] = _mm512_gf2p8mul_epi8(parity [ 26 ], taps [ 26 ]) ;
                        parity [ 27 ] = _mm512_gf2p8mul_epi8(parity [ 27 ], taps [ 27 ]) ;
                        parity [ 28 ] = _mm512_gf2p8mul_epi8(parity [ 28 ], taps [ 28 ]) ;
                        parity [ 29 ] = _mm512_gf2p8mul_epi8(parity [ 29 ], taps [ 29 ]) ;
                        parity [ 30 ] = _mm512_gf2p8mul_epi8(parity [ 30 ], taps [ 30 ]) ;
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


int gf_2vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 2 ], taps [ 2 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
        }
        return ( curPos ) ;
}

int gf_3vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 3 ], taps [ 3 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
        }
        return ( curPos ) ;
}

int gf_4vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 4 ], taps [ 4 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
        }
        return ( curPos ) ;
}

int gf_5vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 5 ], taps [ 5 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
        }
        return ( curPos ) ;
}

int gf_6vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 6 ], taps [ 6 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
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

int gf_7vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 7 ], taps [ 7 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
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

int gf_8vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 8 ], taps [ 8 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
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

int gf_9vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 9 ], taps [ 9 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
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

int gf_10vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 10 ], taps [ 10 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
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

int gf_11vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 11 ], taps [ 11 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
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

int gf_12vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 12 ], taps [ 12 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
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

int gf_13vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 13 ], taps [ 13 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
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

int gf_14vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 14 ], taps [ 14 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
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

int gf_15vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 15 ], taps [ 15 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
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

int gf_16vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 16 ], taps [ 16 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
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

int gf_17vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 17 ], taps [ 17 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
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

int gf_18vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 18 ], taps [ 18 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
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

int gf_19vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 19 ], taps [ 19 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
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

int gf_20vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 20 ], taps [ 20 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
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

int gf_21vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 21 ], taps [ 21 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
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

int gf_22vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 22 ], taps [ 22 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
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

int gf_23vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 23 ], taps [ 23 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
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

int gf_24vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 24 ], taps [ 24 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
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

int gf_25vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 25 ], taps [ 25 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
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

int gf_26vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 26 ], taps [ 26 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
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

int gf_27vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 27 ], taps [ 27 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
                parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ] ) ) ;
                        parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;
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

int gf_28vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 28 ], taps [ 28 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
                parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;
                parity [ 27 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ] ) ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ] ) ) ;
                        parity [ 27 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ]) ;
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

int gf_29vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 29 ], taps [ 29 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
                parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;
                parity [ 27 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ]) ;
                parity [ 28 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ] ) ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ] ) ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ] ) ) ;
                        parity [ 28 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ]) ;
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

int gf_30vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 30 ], taps [ 30 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;
        taps [ 29 ] = _mm512_set1_epi8 ( g_tbls [ 29 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
                parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;
                parity [ 27 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ]) ;
                parity [ 28 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ]) ;
                parity [ 29 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 29 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ] ) ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ] ) ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ] ) ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ] ) ) ;
                        parity [ 29 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 29 ]) ;
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

int gf_31vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 31 ], taps [ 31 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;
        taps [ 29 ] = _mm512_set1_epi8 ( g_tbls [ 29 ] ) ;
        taps [ 30 ] = _mm512_set1_epi8 ( g_tbls [ 30 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
                parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;
                parity [ 27 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ]) ;
                parity [ 28 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ]) ;
                parity [ 29 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 29 ]) ;
                parity [ 30 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 30 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ] ) ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ] ) ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ] ) ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ] ) ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 29 ] ) ) ;
                        parity [ 30 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 30 ]) ;
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

int gf_32vect_pls_avx512_gfni_1b(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 32 ], taps [ 32 ], par_vec ;
        __m512i data_vec, matVec, vreg ;
        unsigned char * pp = ( unsigned char * ) &par_vec ; 
        pp += 60 ;

        taps [ 0 ] = _mm512_set1_epi8 ( g_tbls [ 0 ] ) ;
        taps [ 1 ] = _mm512_set1_epi8 ( g_tbls [ 1 ] ) ;
        taps [ 2 ] = _mm512_set1_epi8 ( g_tbls [ 2 ] ) ;
        taps [ 3 ] = _mm512_set1_epi8 ( g_tbls [ 3 ] ) ;
        taps [ 4 ] = _mm512_set1_epi8 ( g_tbls [ 4 ] ) ;
        taps [ 5 ] = _mm512_set1_epi8 ( g_tbls [ 5 ] ) ;
        taps [ 6 ] = _mm512_set1_epi8 ( g_tbls [ 6 ] ) ;
        taps [ 7 ] = _mm512_set1_epi8 ( g_tbls [ 7 ] ) ;
        taps [ 8 ] = _mm512_set1_epi8 ( g_tbls [ 8 ] ) ;
        taps [ 9 ] = _mm512_set1_epi8 ( g_tbls [ 9 ] ) ;
        taps [ 10 ] = _mm512_set1_epi8 ( g_tbls [ 10 ] ) ;
        taps [ 11 ] = _mm512_set1_epi8 ( g_tbls [ 11 ] ) ;
        taps [ 12 ] = _mm512_set1_epi8 ( g_tbls [ 12 ] ) ;
        taps [ 13 ] = _mm512_set1_epi8 ( g_tbls [ 13 ] ) ;
        taps [ 14 ] = _mm512_set1_epi8 ( g_tbls [ 14 ] ) ;
        taps [ 15 ] = _mm512_set1_epi8 ( g_tbls [ 15 ] ) ;
        taps [ 16 ] = _mm512_set1_epi8 ( g_tbls [ 16 ] ) ;
        taps [ 17 ] = _mm512_set1_epi8 ( g_tbls [ 17 ] ) ;
        taps [ 18 ] = _mm512_set1_epi8 ( g_tbls [ 18 ] ) ;
        taps [ 19 ] = _mm512_set1_epi8 ( g_tbls [ 19 ] ) ;
        taps [ 20 ] = _mm512_set1_epi8 ( g_tbls [ 20 ] ) ;
        taps [ 21 ] = _mm512_set1_epi8 ( g_tbls [ 21 ] ) ;
        taps [ 22 ] = _mm512_set1_epi8 ( g_tbls [ 22 ] ) ;
        taps [ 23 ] = _mm512_set1_epi8 ( g_tbls [ 23 ] ) ;
        taps [ 24 ] = _mm512_set1_epi8 ( g_tbls [ 24 ] ) ;
        taps [ 25 ] = _mm512_set1_epi8 ( g_tbls [ 25 ] ) ;
        taps [ 26 ] = _mm512_set1_epi8 ( g_tbls [ 26 ] ) ;
        taps [ 27 ] = _mm512_set1_epi8 ( g_tbls [ 27 ] ) ;
        taps [ 28 ] = _mm512_set1_epi8 ( g_tbls [ 28 ] ) ;
        taps [ 29 ] = _mm512_set1_epi8 ( g_tbls [ 29 ] ) ;
        taps [ 30 ] = _mm512_set1_epi8 ( g_tbls [ 30 ] ) ;
        taps [ 31 ] = _mm512_set1_epi8 ( g_tbls [ 31 ] ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
               _mm512_mask_storeu_epi32 ( &data [ 0 ] [ curPos ], 0x8000, data_vec ) ;
                parity [ 0 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ]) ;
                parity [ 1 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ]) ;
                parity [ 2 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ]) ;
                parity [ 3 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ]) ;
                parity [ 4 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ]) ;
                parity [ 5 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ]) ;
                parity [ 6 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ]) ;
                parity [ 7 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ]) ;
                parity [ 8 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ]) ;
                parity [ 9 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ]) ;
                parity [ 10 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ]) ;
                parity [ 11 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ]) ;
                parity [ 12 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ]) ;
                parity [ 13 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ]) ;
                parity [ 14 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ]) ;
                parity [ 15 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ]) ;
                parity [ 16 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ]) ;
                parity [ 17 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ]) ;
                parity [ 18 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ]) ;
                parity [ 19 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ]) ;
                parity [ 20 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ]) ;
                parity [ 21 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ]) ;
                parity [ 22 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ]) ;
                parity [ 23 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ]) ;
                parity [ 24 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ]) ;
                parity [ 25 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ]) ;
                parity [ 26 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ]) ;
                parity [ 27 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ]) ;
                parity [ 28 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ]) ;
                parity [ 29 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 29 ]) ;
                parity [ 30 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 30 ]) ;
                parity [ 31 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 31 ]) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                       _mm512_mask_storeu_epi32 ( &data [ curSym ] [ curPos ], 0x8000, data_vec ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 0 ] ) ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 1 ] ) ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 2 ] ) ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 4 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 3 ] ) ) ;
                        parity [ 4 ] = _mm512_xor_si512 ( parity [ 5 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 4 ] ) ) ;
                        parity [ 5 ] = _mm512_xor_si512 ( parity [ 6 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 5 ] ) ) ;
                        parity [ 6 ] = _mm512_xor_si512 ( parity [ 7 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 6 ] ) ) ;
                        parity [ 7 ] = _mm512_xor_si512 ( parity [ 8 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 7 ] ) ) ;
                        parity [ 8 ] = _mm512_xor_si512 ( parity [ 9 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 8 ] ) ) ;
                        parity [ 9 ] = _mm512_xor_si512 ( parity [ 10 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 9 ] ) ) ;
                        parity [ 10 ] = _mm512_xor_si512 ( parity [ 11 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 10 ] ) ) ;
                        parity [ 11 ] = _mm512_xor_si512 ( parity [ 12 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 11 ] ) ) ;
                        parity [ 12 ] = _mm512_xor_si512 ( parity [ 13 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 12 ] ) ) ;
                        parity [ 13 ] = _mm512_xor_si512 ( parity [ 14 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 13 ] ) ) ;
                        parity [ 14 ] = _mm512_xor_si512 ( parity [ 15 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 14 ] ) ) ;
                        parity [ 15 ] = _mm512_xor_si512 ( parity [ 16 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 15 ] ) ) ;
                        parity [ 16 ] = _mm512_xor_si512 ( parity [ 17 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 16 ] ) ) ;
                        parity [ 17 ] = _mm512_xor_si512 ( parity [ 18 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 17 ] ) ) ;
                        parity [ 18 ] = _mm512_xor_si512 ( parity [ 19 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 18 ] ) ) ;
                        parity [ 19 ] = _mm512_xor_si512 ( parity [ 20 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 19 ] ) ) ;
                        parity [ 20 ] = _mm512_xor_si512 ( parity [ 21 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 20 ] ) ) ;
                        parity [ 21 ] = _mm512_xor_si512 ( parity [ 22 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 21 ] ) ) ;
                        parity [ 22 ] = _mm512_xor_si512 ( parity [ 23 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 22 ] ) ) ;
                        parity [ 23 ] = _mm512_xor_si512 ( parity [ 24 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 23 ] ) ) ;
                        parity [ 24 ] = _mm512_xor_si512 ( parity [ 25 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 24 ] ) ) ;
                        parity [ 25 ] = _mm512_xor_si512 ( parity [ 26 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 25 ] ) ) ;
                        parity [ 26 ] = _mm512_xor_si512 ( parity [ 27 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 26 ] ) ) ;
                        parity [ 27 ] = _mm512_xor_si512 ( parity [ 28 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 27 ] ) ) ;
                        parity [ 28 ] = _mm512_xor_si512 ( parity [ 29 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 28 ] ) ) ;
                        parity [ 29 ] = _mm512_xor_si512 ( parity [ 30 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 29 ] ) ) ;
                        parity [ 30 ] = _mm512_xor_si512 ( parity [ 31 ],
                                       _mm512_gf2p8mul_epi8(data_vec, taps [ 30 ] ) ) ;
                        parity [ 31 ] = _mm512_gf2p8mul_epi8(data_vec, taps [ 31 ]) ;
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

void pc_encode_data_avx512_gfni_1b(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        switch (rows) {
        case 2: gf_2vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 3: gf_3vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 4: gf_4vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 5: gf_5vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 6: gf_6vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 7: gf_7vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 8: gf_8vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 9: gf_9vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 10: gf_10vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 11: gf_11vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 12: gf_12vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 13: gf_13vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 14: gf_14vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 15: gf_15vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 16: gf_16vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 17: gf_17vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 18: gf_18vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 19: gf_19vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 20: gf_20vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 21: gf_21vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 22: gf_22vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 23: gf_23vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 24: gf_24vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 25: gf_25vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 26: gf_26vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 27: gf_27vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 28: gf_28vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 29: gf_29vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 30: gf_30vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 31: gf_31vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        case 32: gf_32vect_pls_avx512_gfni_1b(len, k, g_tbls, data, coding);
                 break ;
        }
}
int pc_decode_data_avx512_gfni_1b(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding, int retries)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < retries ) )
        {

                switch (rows) {
                case 2: newPos = gf_2vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 3: newPos = gf_3vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 4: newPos = gf_4vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 5: newPos = gf_5vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 6: newPos = gf_6vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 7: newPos = gf_7vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 8: newPos = gf_8vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 9: newPos = gf_9vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 10: newPos = gf_10vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 11: newPos = gf_11vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 12: newPos = gf_12vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 13: newPos = gf_13vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 14: newPos = gf_14vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 15: newPos = gf_15vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 16: newPos = gf_16vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 17: newPos = gf_17vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 18: newPos = gf_18vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 19: newPos = gf_19vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 20: newPos = gf_20vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 21: newPos = gf_21vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 22: newPos = gf_22vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 23: newPos = gf_23vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 24: newPos = gf_24vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 25: newPos = gf_25vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 26: newPos = gf_26vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 27: newPos = gf_27vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 28: newPos = gf_28vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 29: newPos = gf_29vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 30: newPos = gf_30vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 31: newPos = gf_31vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 32: newPos = gf_32vect_pss_avx512_gfni_1b(len, k, g_tbls, data, coding, newPos);
                         break ;
                }
                if ( newPos < len )
                {
                        if ( pc_correct_AVX512_GFNI_1b ( newPos, k, rows, data, coding, 64 ) )
                        {
                                return ( newPos ) ;
                        }

                }
        }
        return ( newPos ) ;
}
