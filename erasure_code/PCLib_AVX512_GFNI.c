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
#ifndef MAX_PC_RETRY
#define MAX_PC_RETRY 2
extern int pc_correct ( int newPos, int k, int rows, unsigned char ** data, char ** coding, int vLen ) ;
#endif

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
        unsigned char **coding)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < MAX_PC_RETRY ) )
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
                        if ( pc_correct ( newPos, k, rows, data, coding, 64 ) )
                        {
                                return ( newPos ) ;
                        }

                }
        }
        return ( newPos ) ;
}
