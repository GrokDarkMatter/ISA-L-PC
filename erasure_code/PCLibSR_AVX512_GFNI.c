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
    printf ( "Rank = %d, first = %d\n", rank, first ) ;
    first = pcsr_pow_AVX512_GFNI (2, first);
    retVal = first ;
    printf ("Poly power First = %d  ", first);
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

    printf ("FinPoly: %d ", start);
    dump_u8xu8 (p, 1, rank);
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
int gf_2vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 2 ], taps [ 1 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
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
int gf_3vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 3 ], taps [ 2 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
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
int gf_4vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 4 ], taps [ 3 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
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
int gf_5vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 5 ], taps [ 4 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                // Initialize parity values to Symbol 0
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;

                // Loop through all the 1..k symbols
                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

                        // Update parity values using power values and Parallel Multiplier
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
int gf_6vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 6 ], taps [ 5 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_7vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 7 ], taps [ 6 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_8vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 8 ], taps [ 7 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_9vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 9 ], taps [ 8 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_10vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 10 ], taps [ 9 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_11vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 11 ], taps [ 10 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_12vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 12 ], taps [ 11 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_13vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 13 ], taps [ 12 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_14vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 14 ], taps [ 13 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_15vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 15 ], taps [ 14 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_16vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 16 ], taps [ 15 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_17vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 17 ], taps [ 16 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_18vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 18 ], taps [ 17 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_19vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 19 ], taps [ 18 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_20vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 20 ], taps [ 19 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_21vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 21 ], taps [ 20 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_22vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 22 ], taps [ 21 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_23vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 23 ], taps [ 22 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_24vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 24 ], taps [ 23 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_25vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 25 ], taps [ 24 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_26vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 26 ], taps [ 25 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_27vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 27 ], taps [ 26 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_28vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 28 ], taps [ 27 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_29vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 29 ], taps [ 28 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_30vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 30 ], taps [ 29 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_31vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 31 ], taps [ 30 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_32vect_pss_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 32 ], taps [ 31 ] ;            // Parity registers
        __m512i data_vec ;
        // Initialize the taps to the passed in power values to create parallel multiplier
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

        // Loop through each 64 byte codeword
        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                // Load 64 bytes of Original Data
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
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
                        // Load next 64 bytes of Original Data
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;

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
int gf_2vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                        // Loop counters
        __m512i parity [ 2 ], taps [ 1 ] ;          // Parity registers
        __m512i data_vec, temp [ 1 ] ;
        unsigned char **sPnt, *pPnt [ 2 ];                       // Data lookup pointers

        pPnt [ 0 ] = dest [ 0 ] ;
        pPnt [ 1 ] = dest [ 1 ] ;

        // Initialize the taps to the passed in Generator Polynomial values to create Parallel Multiplier
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );

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
int gf_3vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );

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
int gf_4vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );

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
int gf_5vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );

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
int gf_6vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );

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
int gf_7vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );

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
int gf_8vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );

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
int gf_9vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );

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
int gf_10vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );

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
int gf_11vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );

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
int gf_12vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );

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
int gf_13vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );

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
int gf_14vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );

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
int gf_15vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );

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
int gf_16vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );

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
int gf_17vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );

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
int gf_18vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );

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
int gf_19vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );
        taps [ 4 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 4 * 8 ) ) );
        taps [ 5 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 5 * 8 ) ) );
        taps [ 6 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 6 * 8 ) ) );
        taps [ 7 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 7 * 8 ) ) );
        taps [ 8 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 8 * 8 ) ) );

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
int gf_20vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_21vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_22vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_23vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_24vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_25vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_26vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_27vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_28vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_29vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_30vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_31vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
int gf_32vect_pls_sr_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
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
void pc_encode_data_sr_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        switch (rows) {
        case 2: gf_2vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 3: gf_3vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 4: gf_4vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 5: gf_5vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 6: gf_6vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 7: gf_7vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 8: gf_8vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 9: gf_9vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 10: gf_10vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 11: gf_11vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 12: gf_12vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 13: gf_13vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 14: gf_14vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 15: gf_15vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 16: gf_16vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 17: gf_17vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 18: gf_18vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 19: gf_19vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 20: gf_20vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 21: gf_21vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 22: gf_22vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 23: gf_23vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 24: gf_24vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 25: gf_25vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 26: gf_26vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 27: gf_27vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 28: gf_28vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 29: gf_29vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 30: gf_30vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 31: gf_31vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 32: gf_32vect_pls_sr_avx512_gfni(len, k, g_tbls, data, coding);
                 break ;
        }
}
// Single function to access each unrolled Decode
int pc_decode_data_sr_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding, int retries)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < retries ) )
        {

                switch (rows) {
                case 2: newPos = gf_2vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 3: newPos = gf_3vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 4: newPos = gf_4vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 5: newPos = gf_5vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 6: newPos = gf_6vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 7: newPos = gf_7vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 8: newPos = gf_8vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 9: newPos = gf_9vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 10: newPos = gf_10vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 11: newPos = gf_11vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 12: newPos = gf_12vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 13: newPos = gf_13vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 14: newPos = gf_14vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 15: newPos = gf_15vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 16: newPos = gf_16vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 17: newPos = gf_17vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 18: newPos = gf_18vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 19: newPos = gf_19vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 20: newPos = gf_20vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 21: newPos = gf_21vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 22: newPos = gf_22vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 23: newPos = gf_23vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 24: newPos = gf_24vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 25: newPos = gf_25vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 26: newPos = gf_26vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 27: newPos = gf_27vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 28: newPos = gf_28vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 29: newPos = gf_29vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 30: newPos = gf_30vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 31: newPos = gf_31vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 32: newPos = gf_32vect_pss_sr_avx512_gfni(len, k, g_tbls, data, coding, newPos);
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
