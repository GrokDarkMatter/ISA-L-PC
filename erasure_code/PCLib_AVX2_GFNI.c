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
extern int pc_correct ( int newPos, int k, int rows, unsigned char ** data, int vLen ) ;
#endif

int gf_2vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 2 ], taps [ 1 ] ;
        __m256i parity1 [ 2 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_3vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 3 ], taps [ 2 ] ;
        __m256i parity1 [ 3 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_4vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 4 ], taps [ 3 ] ;
        __m256i parity1 [ 4 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_5vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 5 ], taps [ 4 ] ;
        __m256i parity1 [ 5 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_6vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 6 ], taps [ 5 ] ;
        __m256i parity1 [ 6 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_7vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 7 ], taps [ 6 ] ;
        __m256i parity1 [ 7 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_8vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 8 ], taps [ 7 ] ;
        __m256i parity1 [ 8 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_9vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 9 ], taps [ 8 ] ;
        __m256i parity1 [ 9 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_10vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 10 ], taps [ 9 ] ;
        __m256i parity1 [ 10 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_11vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 11 ], taps [ 10 ] ;
        __m256i parity1 [ 11 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_12vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 12 ], taps [ 11 ] ;
        __m256i parity1 [ 12 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_13vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 13 ], taps [ 12 ] ;
        __m256i parity1 [ 13 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_14vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 14 ], taps [ 13 ] ;
        __m256i parity1 [ 14 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_15vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 15 ], taps [ 14 ] ;
        __m256i parity1 [ 15 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_16vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 16 ], taps [ 15 ] ;
        __m256i parity1 [ 16 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_17vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 17 ], taps [ 16 ] ;
        __m256i parity1 [ 17 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_18vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 18 ], taps [ 17 ] ;
        __m256i parity1 [ 18 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_19vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 19 ], taps [ 18 ] ;
        __m256i parity1 [ 19 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_20vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 20 ], taps [ 19 ] ;
        __m256i parity1 [ 20 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_21vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 21 ], taps [ 20 ] ;
        __m256i parity1 [ 21 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_22vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 22 ], taps [ 21 ] ;
        __m256i parity1 [ 22 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_23vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 23 ], taps [ 22 ] ;
        __m256i parity1 [ 23 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_24vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 24 ], taps [ 23 ] ;
        __m256i parity1 [ 24 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_25vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 25 ], taps [ 24 ] ;
        __m256i parity1 [ 25 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_26vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 26 ], taps [ 25 ] ;
        __m256i parity1 [ 26 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_27vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 27 ], taps [ 26 ] ;
        __m256i parity1 [ 27 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;
                parity0 [ 26 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;
                parity1 [ 26 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 25 ], taps [ 25 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 26 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 25 ], taps [ 25 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 26 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 0 * 32 ], parity0 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 1 * 32 ], parity1 [ 26 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_28vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 28 ], taps [ 27 ] ;
        __m256i parity1 [ 28 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;
                parity0 [ 26 ] = data_vec ;
                parity0 [ 27 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;
                parity1 [ 26 ] = data_vec ;
                parity1 [ 27 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 25 ], taps [ 25 ], 0) ;
                        parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 26 ], taps [ 26 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 26 ], data_vec ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 27 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 25 ], taps [ 25 ], 0) ;
                        parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 26 ], taps [ 26 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 26 ], data_vec ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 27 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 0 * 32 ], parity0 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 1 * 32 ], parity1 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 0 * 32 ], parity0 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 1 * 32 ], parity1 [ 27 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_29vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 29 ], taps [ 28 ] ;
        __m256i parity1 [ 29 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;
                parity0 [ 26 ] = data_vec ;
                parity0 [ 27 ] = data_vec ;
                parity0 [ 28 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;
                parity1 [ 26 ] = data_vec ;
                parity1 [ 27 ] = data_vec ;
                parity1 [ 28 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 25 ], taps [ 25 ], 0) ;
                        parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 26 ], taps [ 26 ], 0) ;
                        parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 27 ], taps [ 27 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 26 ], data_vec ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 27 ], data_vec ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 28 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 25 ], taps [ 25 ], 0) ;
                        parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 26 ], taps [ 26 ], 0) ;
                        parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 27 ], taps [ 27 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 26 ], data_vec ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 27 ], data_vec ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 28 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 0 * 32 ], parity0 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 1 * 32 ], parity1 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 0 * 32 ], parity0 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 1 * 32 ], parity1 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 0 * 32 ], parity0 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 1 * 32 ], parity1 [ 28 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_30vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 30 ], taps [ 29 ] ;
        __m256i parity1 [ 30 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;
                parity0 [ 26 ] = data_vec ;
                parity0 [ 27 ] = data_vec ;
                parity0 [ 28 ] = data_vec ;
                parity0 [ 29 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;
                parity1 [ 26 ] = data_vec ;
                parity1 [ 27 ] = data_vec ;
                parity1 [ 28 ] = data_vec ;
                parity1 [ 29 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 25 ], taps [ 25 ], 0) ;
                        parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 26 ], taps [ 26 ], 0) ;
                        parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 27 ], taps [ 27 ], 0) ;
                        parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 28 ], taps [ 28 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 26 ], data_vec ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 27 ], data_vec ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 28 ], data_vec ) ;
                        parity0 [ 29 ] = _mm256_xor_si256 ( parity0 [ 29 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 25 ], taps [ 25 ], 0) ;
                        parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 26 ], taps [ 26 ], 0) ;
                        parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 27 ], taps [ 27 ], 0) ;
                        parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 28 ], taps [ 28 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 26 ], data_vec ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 27 ], data_vec ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 28 ], data_vec ) ;
                        parity1 [ 29 ] = _mm256_xor_si256 ( parity1 [ 29 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 0 * 32 ], parity0 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 1 * 32 ], parity1 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 0 * 32 ], parity0 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 1 * 32 ], parity1 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 0 * 32 ], parity0 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 1 * 32 ], parity1 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 29 ] [ 0 * 32 ], parity0 [ 29 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 29 ] [ 1 * 32 ], parity1 [ 29 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_31vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 31 ], taps [ 30 ] ;
        __m256i parity1 [ 31 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;
        taps [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;
                parity0 [ 26 ] = data_vec ;
                parity0 [ 27 ] = data_vec ;
                parity0 [ 28 ] = data_vec ;
                parity0 [ 29 ] = data_vec ;
                parity0 [ 30 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;
                parity1 [ 26 ] = data_vec ;
                parity1 [ 27 ] = data_vec ;
                parity1 [ 28 ] = data_vec ;
                parity1 [ 29 ] = data_vec ;
                parity1 [ 30 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 25 ], taps [ 25 ], 0) ;
                        parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 26 ], taps [ 26 ], 0) ;
                        parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 27 ], taps [ 27 ], 0) ;
                        parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 28 ], taps [ 28 ], 0) ;
                        parity0 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 29 ], taps [ 29 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 26 ], data_vec ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 27 ], data_vec ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 28 ], data_vec ) ;
                        parity0 [ 29 ] = _mm256_xor_si256 ( parity0 [ 29 ], data_vec ) ;
                        parity0 [ 30 ] = _mm256_xor_si256 ( parity0 [ 30 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 25 ], taps [ 25 ], 0) ;
                        parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 26 ], taps [ 26 ], 0) ;
                        parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 27 ], taps [ 27 ], 0) ;
                        parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 28 ], taps [ 28 ], 0) ;
                        parity1 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 29 ], taps [ 29 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 26 ], data_vec ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 27 ], data_vec ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 28 ], data_vec ) ;
                        parity1 [ 29 ] = _mm256_xor_si256 ( parity1 [ 29 ], data_vec ) ;
                        parity1 [ 30 ] = _mm256_xor_si256 ( parity1 [ 30 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 0 * 32 ], parity0 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 1 * 32 ], parity1 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 0 * 32 ], parity0 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 1 * 32 ], parity1 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 0 * 32 ], parity0 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 1 * 32 ], parity1 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 29 ] [ 0 * 32 ], parity0 [ 29 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 29 ] [ 1 * 32 ], parity1 [ 29 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 30 ] [ 0 * 32 ], parity0 [ 30 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 30 ] [ 1 * 32 ], parity1 [ 30 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_32vect_pss_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;
        __mmask8 mask ;
        __m256i parity0 [ 32 ], taps [ 31 ] ;
        __m256i parity1 [ 32 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;
        taps [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 8 ) ) ) ;
        taps [ 30 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 30 * 8 ) ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = data_vec ;
                parity0 [ 1 ] = data_vec ;
                parity0 [ 2 ] = data_vec ;
                parity0 [ 3 ] = data_vec ;
                parity0 [ 4 ] = data_vec ;
                parity0 [ 5 ] = data_vec ;
                parity0 [ 6 ] = data_vec ;
                parity0 [ 7 ] = data_vec ;
                parity0 [ 8 ] = data_vec ;
                parity0 [ 9 ] = data_vec ;
                parity0 [ 10 ] = data_vec ;
                parity0 [ 11 ] = data_vec ;
                parity0 [ 12 ] = data_vec ;
                parity0 [ 13 ] = data_vec ;
                parity0 [ 14 ] = data_vec ;
                parity0 [ 15 ] = data_vec ;
                parity0 [ 16 ] = data_vec ;
                parity0 [ 17 ] = data_vec ;
                parity0 [ 18 ] = data_vec ;
                parity0 [ 19 ] = data_vec ;
                parity0 [ 20 ] = data_vec ;
                parity0 [ 21 ] = data_vec ;
                parity0 [ 22 ] = data_vec ;
                parity0 [ 23 ] = data_vec ;
                parity0 [ 24 ] = data_vec ;
                parity0 [ 25 ] = data_vec ;
                parity0 [ 26 ] = data_vec ;
                parity0 [ 27 ] = data_vec ;
                parity0 [ 28 ] = data_vec ;
                parity0 [ 29 ] = data_vec ;
                parity0 [ 30 ] = data_vec ;
                parity0 [ 31 ] = data_vec ;

                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = data_vec ;
                parity1 [ 1 ] = data_vec ;
                parity1 [ 2 ] = data_vec ;
                parity1 [ 3 ] = data_vec ;
                parity1 [ 4 ] = data_vec ;
                parity1 [ 5 ] = data_vec ;
                parity1 [ 6 ] = data_vec ;
                parity1 [ 7 ] = data_vec ;
                parity1 [ 8 ] = data_vec ;
                parity1 [ 9 ] = data_vec ;
                parity1 [ 10 ] = data_vec ;
                parity1 [ 11 ] = data_vec ;
                parity1 [ 12 ] = data_vec ;
                parity1 [ 13 ] = data_vec ;
                parity1 [ 14 ] = data_vec ;
                parity1 [ 15 ] = data_vec ;
                parity1 [ 16 ] = data_vec ;
                parity1 [ 17 ] = data_vec ;
                parity1 [ 18 ] = data_vec ;
                parity1 [ 19 ] = data_vec ;
                parity1 [ 20 ] = data_vec ;
                parity1 [ 21 ] = data_vec ;
                parity1 [ 22 ] = data_vec ;
                parity1 [ 23 ] = data_vec ;
                parity1 [ 24 ] = data_vec ;
                parity1 [ 25 ] = data_vec ;
                parity1 [ 26 ] = data_vec ;
                parity1 [ 27 ] = data_vec ;
                parity1 [ 28 ] = data_vec ;
                parity1 [ 29 ] = data_vec ;
                parity1 [ 30 ] = data_vec ;
                parity1 [ 31 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 0 ], taps [ 0 ], 0) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 1 ], taps [ 1 ], 0) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 2 ], taps [ 2 ], 0) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 3 ], taps [ 3 ], 0) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 4 ], taps [ 4 ], 0) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 5 ], taps [ 5 ], 0) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 6 ], taps [ 6 ], 0) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 7 ], taps [ 7 ], 0) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 8 ], taps [ 8 ], 0) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 9 ], taps [ 9 ], 0) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 10 ], taps [ 10 ], 0) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 11 ], taps [ 11 ], 0) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 12 ], taps [ 12 ], 0) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 13 ], taps [ 13 ], 0) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 14 ], taps [ 14 ], 0) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 15 ], taps [ 15 ], 0) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 16 ], taps [ 16 ], 0) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 17 ], taps [ 17 ], 0) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 18 ], taps [ 18 ], 0) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 19 ], taps [ 19 ], 0) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 20 ], taps [ 20 ], 0) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 21 ], taps [ 21 ], 0) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 22 ], taps [ 22 ], 0) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 23 ], taps [ 23 ], 0) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 24 ], taps [ 24 ], 0) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 25 ], taps [ 25 ], 0) ;
                        parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 26 ], taps [ 26 ], 0) ;
                        parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 27 ], taps [ 27 ], 0) ;
                        parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 28 ], taps [ 28 ], 0) ;
                        parity0 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 29 ], taps [ 29 ], 0) ;
                        parity0 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(parity0 [ 30 ], taps [ 30 ], 0) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 0 ], data_vec ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 1 ], data_vec ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 2 ], data_vec ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 3 ], data_vec ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 4 ], data_vec ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 5 ], data_vec ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 6 ], data_vec ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 7 ], data_vec ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 8 ], data_vec ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 9 ], data_vec ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 10 ], data_vec ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 11 ], data_vec ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 12 ], data_vec ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 13 ], data_vec ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 14 ], data_vec ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 15 ], data_vec ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 16 ], data_vec ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 17 ], data_vec ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 18 ], data_vec ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 19 ], data_vec ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 20 ], data_vec ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 21 ], data_vec ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 22 ], data_vec ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 23 ], data_vec ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 24 ], data_vec ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 25 ], data_vec ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 26 ], data_vec ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 27 ], data_vec ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 28 ], data_vec ) ;
                        parity0 [ 29 ] = _mm256_xor_si256 ( parity0 [ 29 ], data_vec ) ;
                        parity0 [ 30 ] = _mm256_xor_si256 ( parity0 [ 30 ], data_vec ) ;
                        parity0 [ 31 ] = _mm256_xor_si256 ( parity0 [ 31 ], data_vec ) ;

                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 0 ], taps [ 0 ], 0) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 1 ], taps [ 1 ], 0) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 2 ], taps [ 2 ], 0) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 3 ], taps [ 3 ], 0) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 4 ], taps [ 4 ], 0) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 5 ], taps [ 5 ], 0) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 6 ], taps [ 6 ], 0) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 7 ], taps [ 7 ], 0) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 8 ], taps [ 8 ], 0) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 9 ], taps [ 9 ], 0) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 10 ], taps [ 10 ], 0) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 11 ], taps [ 11 ], 0) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 12 ], taps [ 12 ], 0) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 13 ], taps [ 13 ], 0) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 14 ], taps [ 14 ], 0) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 15 ], taps [ 15 ], 0) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 16 ], taps [ 16 ], 0) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 17 ], taps [ 17 ], 0) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 18 ], taps [ 18 ], 0) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 19 ], taps [ 19 ], 0) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 20 ], taps [ 20 ], 0) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 21 ], taps [ 21 ], 0) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 22 ], taps [ 22 ], 0) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 23 ], taps [ 23 ], 0) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 24 ], taps [ 24 ], 0) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 25 ], taps [ 25 ], 0) ;
                        parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 26 ], taps [ 26 ], 0) ;
                        parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 27 ], taps [ 27 ], 0) ;
                        parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 28 ], taps [ 28 ], 0) ;
                        parity1 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 29 ], taps [ 29 ], 0) ;
                        parity1 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(parity1 [ 30 ], taps [ 30 ], 0) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 0 ], data_vec ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 1 ], data_vec ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 2 ], data_vec ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 3 ], data_vec ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 4 ], data_vec ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 5 ], data_vec ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 6 ], data_vec ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 7 ], data_vec ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 8 ], data_vec ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 9 ], data_vec ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 10 ], data_vec ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 11 ], data_vec ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 12 ], data_vec ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 13 ], data_vec ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 14 ], data_vec ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 15 ], data_vec ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 16 ], data_vec ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 17 ], data_vec ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 18 ], data_vec ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 19 ], data_vec ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 20 ], data_vec ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 21 ], data_vec ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 22 ], data_vec ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 23 ], data_vec ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 24 ], data_vec ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 25 ], data_vec ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 26 ], data_vec ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 27 ], data_vec ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 28 ], data_vec ) ;
                        parity1 [ 29 ] = _mm256_xor_si256 ( parity1 [ 29 ], data_vec ) ;
                        parity1 [ 30 ] = _mm256_xor_si256 ( parity1 [ 30 ], data_vec ) ;
                        parity1 [ 31 ] = _mm256_xor_si256 ( parity1 [ 31 ], data_vec ) ;

                }

                data_vec = _mm256_or_si256 ( parity0 [ 0 ], parity1 [ 0 ] ) ;
                mask = _mm256_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 0 * 32 ], parity0 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 0 ] [ 1 * 32 ], parity1 [ 0 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 0 * 32 ], parity0 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 1 ] [ 1 * 32 ], parity1 [ 1 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 0 * 32 ], parity0 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 2 ] [ 1 * 32 ], parity1 [ 2 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 0 * 32 ], parity0 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 3 ] [ 1 * 32 ], parity1 [ 3 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 0 * 32 ], parity0 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 4 ] [ 1 * 32 ], parity1 [ 4 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 0 * 32 ], parity0 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 5 ] [ 1 * 32 ], parity1 [ 5 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 0 * 32 ], parity0 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 6 ] [ 1 * 32 ], parity1 [ 6 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 0 * 32 ], parity0 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 7 ] [ 1 * 32 ], parity1 [ 7 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 0 * 32 ], parity0 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 8 ] [ 1 * 32 ], parity1 [ 8 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 0 * 32 ], parity0 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 9 ] [ 1 * 32 ], parity1 [ 9 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 0 * 32 ], parity0 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 10 ] [ 1 * 32 ], parity1 [ 10 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 0 * 32 ], parity0 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 11 ] [ 1 * 32 ], parity1 [ 11 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 0 * 32 ], parity0 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 12 ] [ 1 * 32 ], parity1 [ 12 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 0 * 32 ], parity0 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 13 ] [ 1 * 32 ], parity1 [ 13 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 0 * 32 ], parity0 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 14 ] [ 1 * 32 ], parity1 [ 14 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 0 * 32 ], parity0 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 15 ] [ 1 * 32 ], parity1 [ 15 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 0 * 32 ], parity0 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 16 ] [ 1 * 32 ], parity1 [ 16 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 0 * 32 ], parity0 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 17 ] [ 1 * 32 ], parity1 [ 17 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 0 * 32 ], parity0 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 18 ] [ 1 * 32 ], parity1 [ 18 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 0 * 32 ], parity0 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 19 ] [ 1 * 32 ], parity1 [ 19 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 0 * 32 ], parity0 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 20 ] [ 1 * 32 ], parity1 [ 20 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 0 * 32 ], parity0 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 21 ] [ 1 * 32 ], parity1 [ 21 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 0 * 32 ], parity0 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 22 ] [ 1 * 32 ], parity1 [ 22 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 0 * 32 ], parity0 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 23 ] [ 1 * 32 ], parity1 [ 23 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 0 * 32 ], parity0 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 24 ] [ 1 * 32 ], parity1 [ 24 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 0 * 32 ], parity0 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 25 ] [ 1 * 32 ], parity1 [ 25 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 0 * 32 ], parity0 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 26 ] [ 1 * 32 ], parity1 [ 26 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 0 * 32 ], parity0 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 27 ] [ 1 * 32 ], parity1 [ 27 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 0 * 32 ], parity0 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 28 ] [ 1 * 32 ], parity1 [ 28 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 29 ] [ 0 * 32 ], parity0 [ 29 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 29 ] [ 1 * 32 ], parity1 [ 29 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 30 ] [ 0 * 32 ], parity0 [ 30 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 30 ] [ 1 * 32 ], parity1 [ 30 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 31 ] [ 0 * 32 ], parity0 [ 31 ] ) ;
                        _mm256_store_si256( (__m256i *) &dest [ 31 ] [ 1 * 32 ], parity1 [ 31 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}


int gf_2vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 2 ], taps [ 2 ] ;
        __m256i parity1 [ 2 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
        }
        return ( curPos ) ;
}

int gf_3vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 3 ], taps [ 3 ] ;
        __m256i parity1 [ 3 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
        }
        return ( curPos ) ;
}

int gf_4vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 4 ], taps [ 4 ] ;
        __m256i parity1 [ 4 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
        }
        return ( curPos ) ;
}

int gf_5vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 5 ], taps [ 5 ] ;
        __m256i parity1 [ 5 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
        }
        return ( curPos ) ;
}

int gf_6vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 6 ], taps [ 6 ] ;
        __m256i parity1 [ 6 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
        }
        return ( curPos ) ;
}

int gf_7vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 7 ], taps [ 7 ] ;
        __m256i parity1 [ 7 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
        }
        return ( curPos ) ;
}

int gf_8vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 8 ], taps [ 8 ] ;
        __m256i parity1 [ 8 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
        }
        return ( curPos ) ;
}

int gf_9vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 9 ], taps [ 9 ] ;
        __m256i parity1 [ 9 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
        }
        return ( curPos ) ;
}

int gf_10vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 10 ], taps [ 10 ] ;
        __m256i parity1 [ 10 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
        }
        return ( curPos ) ;
}

int gf_11vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 11 ], taps [ 11 ] ;
        __m256i parity1 [ 11 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
        }
        return ( curPos ) ;
}

int gf_12vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 12 ], taps [ 12 ] ;
        __m256i parity1 [ 12 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
        }
        return ( curPos ) ;
}

int gf_13vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 13 ], taps [ 13 ] ;
        __m256i parity1 [ 13 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
        }
        return ( curPos ) ;
}

int gf_14vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 14 ], taps [ 14 ] ;
        __m256i parity1 [ 14 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
        }
        return ( curPos ) ;
}

int gf_15vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 15 ], taps [ 15 ] ;
        __m256i parity1 [ 15 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
        }
        return ( curPos ) ;
}

int gf_16vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 16 ], taps [ 16 ] ;
        __m256i parity1 [ 16 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
        }
        return ( curPos ) ;
}

int gf_17vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 17 ], taps [ 17 ] ;
        __m256i parity1 [ 17 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
        }
        return ( curPos ) ;
}

int gf_18vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 18 ], taps [ 18 ] ;
        __m256i parity1 [ 18 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
        }
        return ( curPos ) ;
}

int gf_19vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 19 ], taps [ 19 ] ;
        __m256i parity1 [ 19 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
        }
        return ( curPos ) ;
}

int gf_20vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 20 ], taps [ 20 ] ;
        __m256i parity1 [ 20 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
        }
        return ( curPos ) ;
}

int gf_21vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 21 ], taps [ 21 ] ;
        __m256i parity1 [ 21 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
        }
        return ( curPos ) ;
}

int gf_22vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 22 ], taps [ 22 ] ;
        __m256i parity1 [ 22 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
        }
        return ( curPos ) ;
}

int gf_23vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 23 ], taps [ 23 ] ;
        __m256i parity1 [ 23 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
        }
        return ( curPos ) ;
}

int gf_24vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 24 ], taps [ 24 ] ;
        __m256i parity1 [ 24 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
        }
        return ( curPos ) ;
}

int gf_25vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 25 ], taps [ 25 ] ;
        __m256i parity1 [ 25 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
        }
        return ( curPos ) ;
}

int gf_26vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 26 ], taps [ 26 ] ;
        __m256i parity1 [ 26 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
        }
        return ( curPos ) ;
}

int gf_27vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 27 ], taps [ 27 ] ;
        __m256i parity1 [ 27 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 0 * 32 ], parity0 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 1 * 32 ], parity1 [ 26 ] ) ;
        }
        return ( curPos ) ;
}

int gf_28vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 28 ], taps [ 28 ] ;
        __m256i parity1 [ 28 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 0 * 32 ], parity0 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 1 * 32 ], parity1 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 0 * 32 ], parity0 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 1 * 32 ], parity1 [ 27 ] ) ;
        }
        return ( curPos ) ;
}

int gf_29vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 29 ], taps [ 29 ] ;
        __m256i parity1 [ 29 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 0 * 32 ], parity0 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 1 * 32 ], parity1 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 0 * 32 ], parity0 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 1 * 32 ], parity1 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 0 * 32 ], parity0 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 1 * 32 ], parity1 [ 28 ] ) ;
        }
        return ( curPos ) ;
}

int gf_30vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 30 ], taps [ 30 ] ;
        __m256i parity1 [ 30 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;
        taps [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity0 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity1 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 29 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity0 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 29 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity1 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 0 * 32 ], parity0 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 1 * 32 ], parity1 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 0 * 32 ], parity0 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 1 * 32 ], parity1 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 0 * 32 ], parity0 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 1 * 32 ], parity1 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 29 ] [ curPos + 0 * 32 ], parity0 [ 29 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 29 ] [ curPos + 1 * 32 ], parity1 [ 29 ] ) ;
        }
        return ( curPos ) ;
}

int gf_31vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 31 ], taps [ 31 ] ;
        __m256i parity1 [ 31 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;
        taps [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 8 ) ) ) ;
        taps [ 30 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 30 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity0 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                parity0 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity1 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                parity1 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 29 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity0 [ 29 ] = _mm256_xor_si256 ( parity0 [ 30 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0 )  ) ;
                        parity0 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 29 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity1 [ 29 ] = _mm256_xor_si256 ( parity1 [ 30 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0 )  ) ;
                        parity1 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 0 * 32 ], parity0 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 1 * 32 ], parity1 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 0 * 32 ], parity0 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 1 * 32 ], parity1 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 0 * 32 ], parity0 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 1 * 32 ], parity1 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 29 ] [ curPos + 0 * 32 ], parity0 [ 29 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 29 ] [ curPos + 1 * 32 ], parity1 [ 29 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 30 ] [ curPos + 0 * 32 ], parity0 [ 30 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 30 ] [ curPos + 1 * 32 ], parity1 [ 30 ] ) ;
        }
        return ( curPos ) ;
}

int gf_32vect_pls_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;
        __m256i parity0 [ 32 ], taps [ 32 ] ;
        __m256i parity1 [ 32 ] ;
        __m256i data_vec ;

        taps [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 8 ) ) ) ;
        taps [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 8 ) ) ) ;
        taps [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 8 ) ) ) ;
        taps [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 8 ) ) ) ;
        taps [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 8 ) ) ) ;
        taps [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 8 ) ) ) ;
        taps [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 8 ) ) ) ;
        taps [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 8 ) ) ) ;
        taps [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 8 ) ) ) ;
        taps [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 8 ) ) ) ;
        taps [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 8 ) ) ) ;
        taps [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 8 ) ) ) ;
        taps [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 8 ) ) ) ;
        taps [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 8 ) ) ) ;
        taps [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 8 ) ) ) ;
        taps [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 8 ) ) ) ;
        taps [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 8 ) ) ) ;
        taps [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 8 ) ) ) ;
        taps [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 8 ) ) ) ;
        taps [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 8 ) ) ) ;
        taps [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 8 ) ) ) ;
        taps [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 8 ) ) ) ;
        taps [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 8 ) ) ) ;
        taps [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 8 ) ) ) ;
        taps [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 8 ) ) ) ;
        taps [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 8 ) ) ) ;
        taps [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 8 ) ) ) ;
        taps [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 8 ) ) ) ;
        taps [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 8 ) ) ) ;
        taps [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 8 ) ) ) ;
        taps [ 30 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 30 * 8 ) ) ) ;
        taps [ 31 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 31 * 8 ) ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 0 * 32 ] ) ;
                parity0 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity0 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity0 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity0 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity0 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity0 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity0 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity0 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity0 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity0 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity0 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity0 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity0 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity0 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity0 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity0 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity0 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity0 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity0 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity0 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity0 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity0 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity0 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity0 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity0 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity0 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity0 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity0 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity0 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity0 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                parity0 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                parity0 [ 31 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 31 ], 0) ;
                data_vec = _mm256_load_si256( (__m256i *) &data [ 0 ] [ curPos + 1 * 32 ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                parity1 [ 0 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity1 [ 1 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity1 [ 2 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity1 [ 3 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                parity1 [ 4 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0) ;
                parity1 [ 5 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0) ;
                parity1 [ 6 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0) ;
                parity1 [ 7 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0) ;
                parity1 [ 8 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0) ;
                parity1 [ 9 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0) ;
                parity1 [ 10 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0) ;
                parity1 [ 11 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0) ;
                parity1 [ 12 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0) ;
                parity1 [ 13 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0) ;
                parity1 [ 14 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0) ;
                parity1 [ 15 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0) ;
                parity1 [ 16 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0) ;
                parity1 [ 17 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0) ;
                parity1 [ 18 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0) ;
                parity1 [ 19 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0) ;
                parity1 [ 20 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0) ;
                parity1 [ 21 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0) ;
                parity1 [ 22 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0) ;
                parity1 [ 23 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0) ;
                parity1 [ 24 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0) ;
                parity1 [ 25 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0) ;
                parity1 [ 26 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0) ;
                parity1 [ 27 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0) ;
                parity1 [ 28 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0) ;
                parity1 [ 29 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0) ;
                parity1 [ 30 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0) ;
                parity1 [ 31 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 31 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 0 * 32 ] ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity0 [ 0 ] ) ;
                        parity0 [ 0 ] = _mm256_xor_si256 ( parity0 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity0 [ 1 ] = _mm256_xor_si256 ( parity0 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity0 [ 2 ] = _mm256_xor_si256 ( parity0 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity0 [ 3 ] = _mm256_xor_si256 ( parity0 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity0 [ 4 ] = _mm256_xor_si256 ( parity0 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity0 [ 5 ] = _mm256_xor_si256 ( parity0 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity0 [ 6 ] = _mm256_xor_si256 ( parity0 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity0 [ 7 ] = _mm256_xor_si256 ( parity0 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity0 [ 8 ] = _mm256_xor_si256 ( parity0 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity0 [ 9 ] = _mm256_xor_si256 ( parity0 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity0 [ 10 ] = _mm256_xor_si256 ( parity0 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity0 [ 11 ] = _mm256_xor_si256 ( parity0 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity0 [ 12 ] = _mm256_xor_si256 ( parity0 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity0 [ 13 ] = _mm256_xor_si256 ( parity0 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity0 [ 14 ] = _mm256_xor_si256 ( parity0 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity0 [ 15 ] = _mm256_xor_si256 ( parity0 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity0 [ 16 ] = _mm256_xor_si256 ( parity0 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity0 [ 17 ] = _mm256_xor_si256 ( parity0 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity0 [ 18 ] = _mm256_xor_si256 ( parity0 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity0 [ 19 ] = _mm256_xor_si256 ( parity0 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity0 [ 20 ] = _mm256_xor_si256 ( parity0 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity0 [ 21 ] = _mm256_xor_si256 ( parity0 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity0 [ 22 ] = _mm256_xor_si256 ( parity0 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity0 [ 23 ] = _mm256_xor_si256 ( parity0 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity0 [ 24 ] = _mm256_xor_si256 ( parity0 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity0 [ 25 ] = _mm256_xor_si256 ( parity0 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity0 [ 26 ] = _mm256_xor_si256 ( parity0 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity0 [ 27 ] = _mm256_xor_si256 ( parity0 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity0 [ 28 ] = _mm256_xor_si256 ( parity0 [ 29 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity0 [ 29 ] = _mm256_xor_si256 ( parity0 [ 30 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0 )  ) ;
                        parity0 [ 30 ] = _mm256_xor_si256 ( parity0 [ 31 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0 )  ) ;
                        parity0 [ 31 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 31 ], 0) ;
                        data_vec = _mm256_load_si256( (__m256i *) &data [ curSym ] [ curPos + 1 * 32 ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm256_xor_si256( data_vec, parity1 [ 0 ] ) ;
                        parity1 [ 0 ] = _mm256_xor_si256 ( parity1 [ 1 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity1 [ 1 ] = _mm256_xor_si256 ( parity1 [ 2 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity1 [ 2 ] = _mm256_xor_si256 ( parity1 [ 3 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity1 [ 3 ] = _mm256_xor_si256 ( parity1 [ 4 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0 )  ) ;
                        parity1 [ 4 ] = _mm256_xor_si256 ( parity1 [ 5 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 4 ], 0 )  ) ;
                        parity1 [ 5 ] = _mm256_xor_si256 ( parity1 [ 6 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 5 ], 0 )  ) ;
                        parity1 [ 6 ] = _mm256_xor_si256 ( parity1 [ 7 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 6 ], 0 )  ) ;
                        parity1 [ 7 ] = _mm256_xor_si256 ( parity1 [ 8 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 7 ], 0 )  ) ;
                        parity1 [ 8 ] = _mm256_xor_si256 ( parity1 [ 9 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 8 ], 0 )  ) ;
                        parity1 [ 9 ] = _mm256_xor_si256 ( parity1 [ 10 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 9 ], 0 )  ) ;
                        parity1 [ 10 ] = _mm256_xor_si256 ( parity1 [ 11 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 10 ], 0 )  ) ;
                        parity1 [ 11 ] = _mm256_xor_si256 ( parity1 [ 12 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 11 ], 0 )  ) ;
                        parity1 [ 12 ] = _mm256_xor_si256 ( parity1 [ 13 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 12 ], 0 )  ) ;
                        parity1 [ 13 ] = _mm256_xor_si256 ( parity1 [ 14 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 13 ], 0 )  ) ;
                        parity1 [ 14 ] = _mm256_xor_si256 ( parity1 [ 15 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 14 ], 0 )  ) ;
                        parity1 [ 15 ] = _mm256_xor_si256 ( parity1 [ 16 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 15 ], 0 )  ) ;
                        parity1 [ 16 ] = _mm256_xor_si256 ( parity1 [ 17 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 16 ], 0 )  ) ;
                        parity1 [ 17 ] = _mm256_xor_si256 ( parity1 [ 18 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 17 ], 0 )  ) ;
                        parity1 [ 18 ] = _mm256_xor_si256 ( parity1 [ 19 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 18 ], 0 )  ) ;
                        parity1 [ 19 ] = _mm256_xor_si256 ( parity1 [ 20 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 19 ], 0 )  ) ;
                        parity1 [ 20 ] = _mm256_xor_si256 ( parity1 [ 21 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 20 ], 0 )  ) ;
                        parity1 [ 21 ] = _mm256_xor_si256 ( parity1 [ 22 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 21 ], 0 )  ) ;
                        parity1 [ 22 ] = _mm256_xor_si256 ( parity1 [ 23 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 22 ], 0 )  ) ;
                        parity1 [ 23 ] = _mm256_xor_si256 ( parity1 [ 24 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 23 ], 0 )  ) ;
                        parity1 [ 24 ] = _mm256_xor_si256 ( parity1 [ 25 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 24 ], 0 )  ) ;
                        parity1 [ 25 ] = _mm256_xor_si256 ( parity1 [ 26 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 25 ], 0 )  ) ;
                        parity1 [ 26 ] = _mm256_xor_si256 ( parity1 [ 27 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 26 ], 0 )  ) ;
                        parity1 [ 27 ] = _mm256_xor_si256 ( parity1 [ 28 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 27 ], 0 )  ) ;
                        parity1 [ 28 ] = _mm256_xor_si256 ( parity1 [ 29 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 28 ], 0 )  ) ;
                        parity1 [ 29 ] = _mm256_xor_si256 ( parity1 [ 30 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 29 ], 0 )  ) ;
                        parity1 [ 30 ] = _mm256_xor_si256 ( parity1 [ 31 ],
                                       _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 30 ], 0 )  ) ;
                        parity1 [ 31 ] = _mm256_gf2p8affine_epi64_epi8(data_vec, taps [ 31 ], 0) ;
                }

                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 0 * 32 ], parity0 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 0 ] [ curPos + 1 * 32 ], parity1 [ 0 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 0 * 32 ], parity0 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 1 ] [ curPos + 1 * 32 ], parity1 [ 1 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 0 * 32 ], parity0 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 2 ] [ curPos + 1 * 32 ], parity1 [ 2 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 0 * 32 ], parity0 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 3 ] [ curPos + 1 * 32 ], parity1 [ 3 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 0 * 32 ], parity0 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 4 ] [ curPos + 1 * 32 ], parity1 [ 4 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 0 * 32 ], parity0 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 5 ] [ curPos + 1 * 32 ], parity1 [ 5 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 0 * 32 ], parity0 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 6 ] [ curPos + 1 * 32 ], parity1 [ 6 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 0 * 32 ], parity0 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 7 ] [ curPos + 1 * 32 ], parity1 [ 7 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 0 * 32 ], parity0 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 8 ] [ curPos + 1 * 32 ], parity1 [ 8 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 0 * 32 ], parity0 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 9 ] [ curPos + 1 * 32 ], parity1 [ 9 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 0 * 32 ], parity0 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 10 ] [ curPos + 1 * 32 ], parity1 [ 10 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 0 * 32 ], parity0 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 11 ] [ curPos + 1 * 32 ], parity1 [ 11 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 0 * 32 ], parity0 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 12 ] [ curPos + 1 * 32 ], parity1 [ 12 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 0 * 32 ], parity0 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 13 ] [ curPos + 1 * 32 ], parity1 [ 13 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 0 * 32 ], parity0 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 14 ] [ curPos + 1 * 32 ], parity1 [ 14 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 0 * 32 ], parity0 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 15 ] [ curPos + 1 * 32 ], parity1 [ 15 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 0 * 32 ], parity0 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 16 ] [ curPos + 1 * 32 ], parity1 [ 16 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 0 * 32 ], parity0 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 17 ] [ curPos + 1 * 32 ], parity1 [ 17 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 0 * 32 ], parity0 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 18 ] [ curPos + 1 * 32 ], parity1 [ 18 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 0 * 32 ], parity0 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 19 ] [ curPos + 1 * 32 ], parity1 [ 19 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 0 * 32 ], parity0 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 20 ] [ curPos + 1 * 32 ], parity1 [ 20 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 0 * 32 ], parity0 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 21 ] [ curPos + 1 * 32 ], parity1 [ 21 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 0 * 32 ], parity0 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 22 ] [ curPos + 1 * 32 ], parity1 [ 22 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 0 * 32 ], parity0 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 23 ] [ curPos + 1 * 32 ], parity1 [ 23 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 0 * 32 ], parity0 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 24 ] [ curPos + 1 * 32 ], parity1 [ 24 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 0 * 32 ], parity0 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 25 ] [ curPos + 1 * 32 ], parity1 [ 25 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 0 * 32 ], parity0 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 26 ] [ curPos + 1 * 32 ], parity1 [ 26 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 0 * 32 ], parity0 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 27 ] [ curPos + 1 * 32 ], parity1 [ 27 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 0 * 32 ], parity0 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 28 ] [ curPos + 1 * 32 ], parity1 [ 28 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 29 ] [ curPos + 0 * 32 ], parity0 [ 29 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 29 ] [ curPos + 1 * 32 ], parity1 [ 29 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 30 ] [ curPos + 0 * 32 ], parity0 [ 30 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 30 ] [ curPos + 1 * 32 ], parity1 [ 30 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 31 ] [ curPos + 0 * 32 ], parity0 [ 31 ] ) ;
                _mm256_store_si256( (__m256i *) &dest [ 31 ] [ curPos + 1 * 32 ], parity1 [ 31 ] ) ;
        }
        return ( curPos ) ;
}

void pc_encode_data_avx2_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        switch (rows) {
        case 2: gf_2vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 3: gf_3vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 4: gf_4vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 5: gf_5vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 6: gf_6vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 7: gf_7vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 8: gf_8vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 9: gf_9vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 10: gf_10vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 11: gf_11vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 12: gf_12vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 13: gf_13vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 14: gf_14vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 15: gf_15vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 16: gf_16vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 17: gf_17vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 18: gf_18vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 19: gf_19vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 20: gf_20vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 21: gf_21vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 22: gf_22vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 23: gf_23vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 24: gf_24vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 25: gf_25vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 26: gf_26vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 27: gf_27vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 28: gf_28vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 29: gf_29vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 30: gf_30vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 31: gf_31vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        case 32: gf_32vect_pls_avx2_gfni(len, k, g_tbls, data, coding);
                 break ;
        }
}
int pc_decode_data_avx2_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < MAX_PC_RETRY ) )
        {

                switch (rows) {
                case 2: newPos = gf_2vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 3: newPos = gf_3vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 4: newPos = gf_4vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 5: newPos = gf_5vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 6: newPos = gf_6vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 7: newPos = gf_7vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 8: newPos = gf_8vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 9: newPos = gf_9vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 10: newPos = gf_10vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 11: newPos = gf_11vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 12: newPos = gf_12vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 13: newPos = gf_13vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 14: newPos = gf_14vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 15: newPos = gf_15vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 16: newPos = gf_16vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 17: newPos = gf_17vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 18: newPos = gf_18vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 19: newPos = gf_19vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 20: newPos = gf_20vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 21: newPos = gf_21vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 22: newPos = gf_22vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 23: newPos = gf_23vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 24: newPos = gf_24vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 25: newPos = gf_25vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 26: newPos = gf_26vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 27: newPos = gf_27vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 28: newPos = gf_28vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 29: newPos = gf_29vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 30: newPos = gf_30vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 31: newPos = gf_31vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 32: newPos = gf_32vect_pss_avx2_gfni(len, k, g_tbls, data, coding, newPos);
                         break ;
                }
                if ( newPos < len )
                {
                        if ( pc_correct ( newPos, k, rows, data, 32 ) )
                        {
                                return ( newPos ) ;
                        }

                }
        }
        return ( newPos ) ;
}
