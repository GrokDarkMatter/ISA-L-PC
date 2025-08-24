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

#define MAX_PC_RETRY 2
extern int pc_correct ( int newPos, int k, int rows, unsigned char ** data, int vLen ) ;

int gf_2vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 2 ], taps [ 1 ], tapsh [ 1 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_3vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 3 ], taps [ 2 ], tapsh [ 2 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_4vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 4 ], taps [ 3 ], tapsh [ 3 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_5vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 5 ], taps [ 4 ], tapsh [ 4 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_6vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 6 ], taps [ 5 ], tapsh [ 5 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_7vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 7 ], taps [ 6 ], tapsh [ 6 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;
                parity [ 4 ] = data_vec ;
                parity [ 5 ] = data_vec ;
                parity [ 6 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_8vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 8 ], taps [ 7 ], tapsh [ 7 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_9vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 9 ], taps [ 8 ], tapsh [ 8 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_10vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 10 ], taps [ 9 ], tapsh [ 9 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_11vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 11 ], taps [ 10 ], tapsh [ 10 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_12vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 12 ], taps [ 11 ], tapsh [ 11 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_13vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 13 ], taps [ 12 ], tapsh [ 12 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_14vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 14 ], taps [ 13 ], tapsh [ 13 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_15vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 15 ], taps [ 14 ], tapsh [ 14 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_16vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 16 ], taps [ 15 ], tapsh [ 15 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_17vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 17 ], taps [ 16 ], tapsh [ 16 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_18vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 18 ], taps [ 17 ], tapsh [ 17 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_19vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 19 ], taps [ 18 ], tapsh [ 18 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_20vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 20 ], taps [ 19 ], tapsh [ 19 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_21vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 21 ], taps [ 20 ], tapsh [ 20 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_22vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 22 ], taps [ 21 ], tapsh [ 21 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_23vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 23 ], taps [ 22 ], tapsh [ 22 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_24vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 24 ], taps [ 23 ], tapsh [ 23 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_25vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 25 ], taps [ 24 ], tapsh [ 24 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_26vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 26 ], taps [ 25 ], tapsh [ 25 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_27vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 27 ], taps [ 26 ], tapsh [ 26 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        vst1q_u8 ( &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_28vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 28 ], taps [ 27 ], tapsh [ 27 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        vst1q_u8 ( &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        vst1q_u8 ( &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_29vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 29 ], taps [ 28 ], tapsh [ 28 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        vst1q_u8 ( &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        vst1q_u8 ( &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        vst1q_u8 ( &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_30vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 30 ], taps [ 29 ], tapsh [ 29 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = vqeorq_u8(parity [ 28 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = vqeorq_u8(parity [ 28 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        vst1q_u8 ( &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        vst1q_u8 ( &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        vst1q_u8 ( &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        vst1q_u8 ( &dest [ 29 ] [ 0 ], parity [ 29 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_31vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 31 ], taps [ 30 ], tapsh [ 30 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 32 ) + 16 ) ) ;
        taps  [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 32 ) ) ) ;
        tapsh [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 30 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = vqeorq_u8(parity [ 28 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = vqeorq_u8(parity [ 28 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                        parity [ 29 ] = vqeorq_u8(parity [ 29 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 29 ] ) ;
                        parity [ 29 ] = vqeorq_u8(parity [ 29 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        vst1q_u8 ( &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        vst1q_u8 ( &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        vst1q_u8 ( &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        vst1q_u8 ( &dest [ 29 ] [ 0 ], parity [ 29 ] ) ;
                        vst1q_u8 ( &dest [ 30 ] [ 0 ], parity [ 30 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_32vect_pss_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 32 ], taps [ 31 ], tapsh [ 31 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 28 * 32 ) + 16 ) ) ;
        taps  [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 32 ) ) ) ;
        tapsh [ 29 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 29 * 32 ) + 16 ) ) ;
        taps  [ 30 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 30 * 32 ) ) ) ;
        tapsh [ 30 ] = _mm256_set1_epi64x( *( uint64_t * ) ( g_tbls + ( 30 * 32 ) + 16 ) ) ;

        for ( curPos = offSet ; curPos < len ; curPos += 16 )
        {
                data_vec = vldlq_u8( (__m256i *) &data [ 0 ] [ curPos ] ) ;
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
                        data_vec = vldlq_u8( &data [ curSym ] [ curPos ] ) ;
                        data_vech = vshrq_u8 ( data_vec, 4 ) ;
                        parity [ 31 ] = veorq_u8 ( parity [ 1 ], data_vec ) ;
                        data_vec = vandq_u8 ( data_vec, mask0f ) ;

                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = vqeorq_u8(parity [ 0 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = vqeorq_u8(parity [ 1 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = vqeorq_u8(parity [ 2 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = vqeorq_u8(parity [ 3 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = vqeorq_u8(parity [ 4 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = vqeorq_u8(parity [ 5 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = vqeorq_u8(parity [ 6 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = vqeorq_u8(parity [ 7 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = vqeorq_u8(parity [ 8 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = vqeorq_u8(parity [ 9 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = vqeorq_u8(parity [ 10 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = vqeorq_u8(parity [ 11 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = vqeorq_u8(parity [ 12 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = vqeorq_u8(parity [ 13 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = vqeorq_u8(parity [ 14 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = vqeorq_u8(parity [ 15 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = vqeorq_u8(parity [ 16 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = vqeorq_u8(parity [ 17 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = vqeorq_u8(parity [ 18 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = vqeorq_u8(parity [ 19 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = vqeorq_u8(parity [ 20 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = vqeorq_u8(parity [ 21 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = vqeorq_u8(parity [ 22 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = vqeorq_u8(parity [ 23 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = vqeorq_u8(parity [ 24 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = vqeorq_u8(parity [ 25 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = vqeorq_u8(parity [ 26 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = vqeorq_u8(parity [ 27 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = vqeorq_u8(parity [ 28 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = vqeorq_u8(parity [ 28 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                        parity [ 29 ] = vqeorq_u8(parity [ 29 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 29 ] ) ;
                        parity [ 29 ] = vqeorq_u8(parity [ 29 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 30 ] ) ;
                        parity [ 30 ] = vqeorq_u8(parity [ 30 ], temp ] ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 30 ] ) ;
                        parity [ 30 ] = vqeorq_u8(parity [ 30 ], temp ] ) ;
                }

                uint32x4_t tmp = vreinterpretq_u32_u8( parity [ 0 ] ) ;
                if ( vmaxvq_u32 ( tmp ) == 0 )
                {
                        vst1q_u8 ( &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        vst1q_u8 ( &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        vst1q_u8 ( &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        vst1q_u8 ( &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        vst1q_u8 ( &dest [ 4 ] [ 0 ], parity [ 4 ] ) ;
                        vst1q_u8 ( &dest [ 5 ] [ 0 ], parity [ 5 ] ) ;
                        vst1q_u8 ( &dest [ 6 ] [ 0 ], parity [ 6 ] ) ;
                        vst1q_u8 ( &dest [ 7 ] [ 0 ], parity [ 7 ] ) ;
                        vst1q_u8 ( &dest [ 8 ] [ 0 ], parity [ 8 ] ) ;
                        vst1q_u8 ( &dest [ 9 ] [ 0 ], parity [ 9 ] ) ;
                        vst1q_u8 ( &dest [ 10 ] [ 0 ], parity [ 10 ] ) ;
                        vst1q_u8 ( &dest [ 11 ] [ 0 ], parity [ 11 ] ) ;
                        vst1q_u8 ( &dest [ 12 ] [ 0 ], parity [ 12 ] ) ;
                        vst1q_u8 ( &dest [ 13 ] [ 0 ], parity [ 13 ] ) ;
                        vst1q_u8 ( &dest [ 14 ] [ 0 ], parity [ 14 ] ) ;
                        vst1q_u8 ( &dest [ 15 ] [ 0 ], parity [ 15 ] ) ;
                        vst1q_u8 ( &dest [ 16 ] [ 0 ], parity [ 16 ] ) ;
                        vst1q_u8 ( &dest [ 17 ] [ 0 ], parity [ 17 ] ) ;
                        vst1q_u8 ( &dest [ 18 ] [ 0 ], parity [ 18 ] ) ;
                        vst1q_u8 ( &dest [ 19 ] [ 0 ], parity [ 19 ] ) ;
                        vst1q_u8 ( &dest [ 20 ] [ 0 ], parity [ 20 ] ) ;
                        vst1q_u8 ( &dest [ 21 ] [ 0 ], parity [ 21 ] ) ;
                        vst1q_u8 ( &dest [ 22 ] [ 0 ], parity [ 22 ] ) ;
                        vst1q_u8 ( &dest [ 23 ] [ 0 ], parity [ 23 ] ) ;
                        vst1q_u8 ( &dest [ 24 ] [ 0 ], parity [ 24 ] ) ;
                        vst1q_u8 ( &dest [ 25 ] [ 0 ], parity [ 25 ] ) ;
                        vst1q_u8 ( &dest [ 26 ] [ 0 ], parity [ 26 ] ) ;
                        vst1q_u8 ( &dest [ 27 ] [ 0 ], parity [ 27 ] ) ;
                        vst1q_u8 ( &dest [ 28 ] [ 0 ], parity [ 28 ] ) ;
                        vst1q_u8 ( &dest [ 29 ] [ 0 ], parity [ 29 ] ) ;
                        vst1q_u8 ( &dest [ 30 ] [ 0 ], parity [ 30 ] ) ;
                        vst1q_u8 ( &dest [ 31 ] [ 0 ], parity [ 31 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}


int gf_2vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 2 ], taps [ 1 ], tapsh [ 1 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
        }
        return ( curPos ) ;
}

int gf_3vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 3 ], taps [ 2 ], tapsh [ 2 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
        }
        return ( curPos ) ;
}

int gf_4vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 4 ], taps [ 3 ], tapsh [ 3 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
        }
        return ( curPos ) ;
}

int gf_5vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 5 ], taps [ 4 ], tapsh [ 4 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
        }
        return ( curPos ) ;
}

int gf_6vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 6 ], taps [ 5 ], tapsh [ 5 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
        }
        return ( curPos ) ;
}

int gf_7vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 7 ], taps [ 6 ], tapsh [ 6 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
        }
        return ( curPos ) ;
}

int gf_8vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 8 ], taps [ 7 ], tapsh [ 7 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
        }
        return ( curPos ) ;
}

int gf_9vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 9 ], taps [ 8 ], tapsh [ 8 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
        }
        return ( curPos ) ;
}

int gf_10vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 10 ], taps [ 9 ], tapsh [ 9 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
        }
        return ( curPos ) ;
}

int gf_11vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 11 ], taps [ 10 ], tapsh [ 10 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
        }
        return ( curPos ) ;
}

int gf_12vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 12 ], taps [ 11 ], tapsh [ 11 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
        }
        return ( curPos ) ;
}

int gf_13vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 13 ], taps [ 12 ], tapsh [ 12 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
        }
        return ( curPos ) ;
}

int gf_14vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 14 ], taps [ 13 ], tapsh [ 13 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
        }
        return ( curPos ) ;
}

int gf_15vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 15 ], taps [ 14 ], tapsh [ 14 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
        }
        return ( curPos ) ;
}

int gf_16vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 16 ], taps [ 15 ], tapsh [ 15 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
        }
        return ( curPos ) ;
}

int gf_17vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 17 ], taps [ 16 ], tapsh [ 16 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
        }
        return ( curPos ) ;
}

int gf_18vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 18 ], taps [ 17 ], tapsh [ 17 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
        }
        return ( curPos ) ;
}

int gf_19vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 19 ], taps [ 18 ], tapsh [ 18 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
        }
        return ( curPos ) ;
}

int gf_20vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 20 ], taps [ 19 ], tapsh [ 19 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
        }
        return ( curPos ) ;
}

int gf_21vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 21 ], taps [ 20 ], tapsh [ 20 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
        }
        return ( curPos ) ;
}

int gf_22vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 22 ], taps [ 21 ], tapsh [ 21 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
        }
        return ( curPos ) ;
}

int gf_23vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 23 ], taps [ 22 ], tapsh [ 22 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
        }
        return ( curPos ) ;
}

int gf_24vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 24 ], taps [ 23 ], tapsh [ 23 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
        }
        return ( curPos ) ;
}

int gf_25vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 25 ], taps [ 24 ], tapsh [ 24 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
        }
        return ( curPos ) ;
}

int gf_26vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 26 ], taps [ 25 ], tapsh [ 25 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
        }
        return ( curPos ) ;
}

int gf_27vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 27 ], taps [ 26 ], tapsh [ 26 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;
                parity [ 26 ] = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 26 ] = veorq_u8 ( parity[ 26 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                vst1q_u8 ( &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
        }
        return ( curPos ) ;
}

int gf_28vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 28 ], taps [ 27 ], tapsh [ 27 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;
                parity [ 26 ] = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 26 ] = veorq_u8 ( parity[ 26 ], temp ) ;
                parity [ 27 ] = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 27 ] = veorq_u8 ( parity[ 27 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                vst1q_u8 ( &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                vst1q_u8 ( &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
        }
        return ( curPos ) ;
}

int gf_29vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 29 ], taps [ 28 ], tapsh [ 28 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;
                parity [ 26 ] = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 26 ] = veorq_u8 ( parity[ 26 ], temp ) ;
                parity [ 27 ] = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 27 ] = veorq_u8 ( parity[ 27 ], temp ) ;
                parity [ 28 ] = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 28 ] = veorq_u8 ( parity[ 28 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                vst1q_u8 ( &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                vst1q_u8 ( &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                vst1q_u8 ( &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
        }
        return ( curPos ) ;
}

int gf_30vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 30 ], taps [ 29 ], tapsh [ 29 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) + 16 ) ) ;
        taps  [ 29 ] = vld1q_u8( g_tbls + ( 29 * 32 ) ) ) ;
        tapsh [ 29 ] = vld1q_u8( g_tbls + ( 29 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;
                parity [ 26 ] = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 26 ] = veorq_u8 ( parity[ 26 ], temp ) ;
                parity [ 27 ] = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 27 ] = veorq_u8 ( parity[ 27 ], temp ) ;
                parity [ 28 ] = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 28 ] = veorq_u8 ( parity[ 28 ], temp ) ;
                parity [ 29 ] = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 29 ] = veorq_u8 ( parity[ 29 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 29 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 29 ] ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 29 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                vst1q_u8 ( &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                vst1q_u8 ( &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                vst1q_u8 ( &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
                vst1q_u8 ( &dest [ 29 ] [ curPos ], parity [ 29 ] ) ;
        }
        return ( curPos ) ;
}

int gf_31vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 31 ], taps [ 30 ], tapsh [ 30 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) + 16 ) ) ;
        taps  [ 29 ] = vld1q_u8( g_tbls + ( 29 * 32 ) ) ) ;
        tapsh [ 29 ] = vld1q_u8( g_tbls + ( 29 * 32 ) + 16 ) ) ;
        taps  [ 30 ] = vld1q_u8( g_tbls + ( 30 * 32 ) ) ) ;
        tapsh [ 30 ] = vld1q_u8( g_tbls + ( 30 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;
                parity [ 26 ] = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 26 ] = veorq_u8 ( parity[ 26 ], temp ) ;
                parity [ 27 ] = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 27 ] = veorq_u8 ( parity[ 27 ], temp ) ;
                parity [ 28 ] = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 28 ] = veorq_u8 ( parity[ 28 ], temp ) ;
                parity [ 29 ] = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 29 ] = veorq_u8 ( parity[ 29 ], temp ) ;
                parity [ 30 ] = vqtbl1q_u8 ( data_vec, taps [ 30 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 30 ] = veorq_u8 ( parity[ 30 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 29 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 29 ] ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 29 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 30 ] ) ;
                        parity [ 30 ] = veorq_u8 ( parity [ 30 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 30 ] ) ;
                        parity [ 30 ] = veorq_u8 ( parity [ 30 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                vst1q_u8 ( &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                vst1q_u8 ( &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                vst1q_u8 ( &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
                vst1q_u8 ( &dest [ 29 ] [ curPos ], parity [ 29 ] ) ;
                vst1q_u8 ( &dest [ 30 ] [ curPos ], parity [ 30 ] ) ;
        }
        return ( curPos ) ;
}

int gf_32vect_pls_neon(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos = 0 ;                      // Loop counters
        uint8x16_t parity [ 32 ], taps [ 31 ], tapsh [ 31 ] ;
        uint8x16_t data_vec, data_vech, temp ;
        uint8x16_t mask0f = vmovq_n_u8 (0x0f)

        taps  [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) ) ) ;
        tapsh [ 0 ] = vld1q_u8( g_tbls + ( 0 * 32 ) + 16 ) ) ;
        taps  [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) ) ) ;
        tapsh [ 1 ] = vld1q_u8( g_tbls + ( 1 * 32 ) + 16 ) ) ;
        taps  [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) ) ) ;
        tapsh [ 2 ] = vld1q_u8( g_tbls + ( 2 * 32 ) + 16 ) ) ;
        taps  [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) ) ) ;
        tapsh [ 3 ] = vld1q_u8( g_tbls + ( 3 * 32 ) + 16 ) ) ;
        taps  [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) ) ) ;
        tapsh [ 4 ] = vld1q_u8( g_tbls + ( 4 * 32 ) + 16 ) ) ;
        taps  [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) ) ) ;
        tapsh [ 5 ] = vld1q_u8( g_tbls + ( 5 * 32 ) + 16 ) ) ;
        taps  [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) ) ) ;
        tapsh [ 6 ] = vld1q_u8( g_tbls + ( 6 * 32 ) + 16 ) ) ;
        taps  [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) ) ) ;
        tapsh [ 7 ] = vld1q_u8( g_tbls + ( 7 * 32 ) + 16 ) ) ;
        taps  [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) ) ) ;
        tapsh [ 8 ] = vld1q_u8( g_tbls + ( 8 * 32 ) + 16 ) ) ;
        taps  [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) ) ) ;
        tapsh [ 9 ] = vld1q_u8( g_tbls + ( 9 * 32 ) + 16 ) ) ;
        taps  [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) ) ) ;
        tapsh [ 10 ] = vld1q_u8( g_tbls + ( 10 * 32 ) + 16 ) ) ;
        taps  [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) ) ) ;
        tapsh [ 11 ] = vld1q_u8( g_tbls + ( 11 * 32 ) + 16 ) ) ;
        taps  [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) ) ) ;
        tapsh [ 12 ] = vld1q_u8( g_tbls + ( 12 * 32 ) + 16 ) ) ;
        taps  [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) ) ) ;
        tapsh [ 13 ] = vld1q_u8( g_tbls + ( 13 * 32 ) + 16 ) ) ;
        taps  [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) ) ) ;
        tapsh [ 14 ] = vld1q_u8( g_tbls + ( 14 * 32 ) + 16 ) ) ;
        taps  [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) ) ) ;
        tapsh [ 15 ] = vld1q_u8( g_tbls + ( 15 * 32 ) + 16 ) ) ;
        taps  [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) ) ) ;
        tapsh [ 16 ] = vld1q_u8( g_tbls + ( 16 * 32 ) + 16 ) ) ;
        taps  [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) ) ) ;
        tapsh [ 17 ] = vld1q_u8( g_tbls + ( 17 * 32 ) + 16 ) ) ;
        taps  [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) ) ) ;
        tapsh [ 18 ] = vld1q_u8( g_tbls + ( 18 * 32 ) + 16 ) ) ;
        taps  [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) ) ) ;
        tapsh [ 19 ] = vld1q_u8( g_tbls + ( 19 * 32 ) + 16 ) ) ;
        taps  [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) ) ) ;
        tapsh [ 20 ] = vld1q_u8( g_tbls + ( 20 * 32 ) + 16 ) ) ;
        taps  [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) ) ) ;
        tapsh [ 21 ] = vld1q_u8( g_tbls + ( 21 * 32 ) + 16 ) ) ;
        taps  [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) ) ) ;
        tapsh [ 22 ] = vld1q_u8( g_tbls + ( 22 * 32 ) + 16 ) ) ;
        taps  [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) ) ) ;
        tapsh [ 23 ] = vld1q_u8( g_tbls + ( 23 * 32 ) + 16 ) ) ;
        taps  [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) ) ) ;
        tapsh [ 24 ] = vld1q_u8( g_tbls + ( 24 * 32 ) + 16 ) ) ;
        taps  [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) ) ) ;
        tapsh [ 25 ] = vld1q_u8( g_tbls + ( 25 * 32 ) + 16 ) ) ;
        taps  [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) ) ) ;
        tapsh [ 26 ] = vld1q_u8( g_tbls + ( 26 * 32 ) + 16 ) ) ;
        taps  [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) ) ) ;
        tapsh [ 27 ] = vld1q_u8( g_tbls + ( 27 * 32 ) + 16 ) ) ;
        taps  [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) ) ) ;
        tapsh [ 28 ] = vld1q_u8( g_tbls + ( 28 * 32 ) + 16 ) ) ;
        taps  [ 29 ] = vld1q_u8( g_tbls + ( 29 * 32 ) ) ) ;
        tapsh [ 29 ] = vld1q_u8( g_tbls + ( 29 * 32 ) + 16 ) ) ;
        taps  [ 30 ] = vld1q_u8( g_tbls + ( 30 * 32 ) ) ) ;
        tapsh [ 30 ] = vld1q_u8( g_tbls + ( 30 * 32 ) + 16 ) ) ;
        taps  [ 31 ] = vld1q_u8( g_tbls + ( 31 * 32 ) ) ) ;
        tapsh [ 31 ] = vld1q_u8( g_tbls + ( 31 * 32 ) + 16 ) ) ;

        for ( curPos = 0 ; curPos < len ; curPos += 16 )
        {
                data_vec =  vldlq_u8( &data [ 0 ] [ curPos ] ) ;
                data_vech = vlshrq_n_u8 ( data_vec, 4 ) ;
                data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                parity [ 0 ] = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 0 ] = veorq_u8 ( parity[ 0 ], temp ) ;
                parity [ 1 ] = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 1 ] = veorq_u8 ( parity[ 1 ], temp ) ;
                parity [ 2 ] = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 2 ] = veorq_u8 ( parity[ 2 ], temp ) ;
                parity [ 3 ] = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 3 ] = veorq_u8 ( parity[ 3 ], temp ) ;
                parity [ 4 ] = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 4 ] = veorq_u8 ( parity[ 4 ], temp ) ;
                parity [ 5 ] = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 5 ] = veorq_u8 ( parity[ 5 ], temp ) ;
                parity [ 6 ] = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 6 ] = veorq_u8 ( parity[ 6 ], temp ) ;
                parity [ 7 ] = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 7 ] = veorq_u8 ( parity[ 7 ], temp ) ;
                parity [ 8 ] = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 8 ] = veorq_u8 ( parity[ 8 ], temp ) ;
                parity [ 9 ] = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 9 ] = veorq_u8 ( parity[ 9 ], temp ) ;
                parity [ 10 ] = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 10 ] = veorq_u8 ( parity[ 10 ], temp ) ;
                parity [ 11 ] = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 11 ] = veorq_u8 ( parity[ 11 ], temp ) ;
                parity [ 12 ] = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 12 ] = veorq_u8 ( parity[ 12 ], temp ) ;
                parity [ 13 ] = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 13 ] = veorq_u8 ( parity[ 13 ], temp ) ;
                parity [ 14 ] = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 14 ] = veorq_u8 ( parity[ 14 ], temp ) ;
                parity [ 15 ] = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 15 ] = veorq_u8 ( parity[ 15 ], temp ) ;
                parity [ 16 ] = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 16 ] = veorq_u8 ( parity[ 16 ], temp ) ;
                parity [ 17 ] = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 17 ] = veorq_u8 ( parity[ 17 ], temp ) ;
                parity [ 18 ] = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 18 ] = veorq_u8 ( parity[ 18 ], temp ) ;
                parity [ 19 ] = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 19 ] = veorq_u8 ( parity[ 19 ], temp ) ;
                parity [ 20 ] = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 20 ] = veorq_u8 ( parity[ 20 ], temp ) ;
                parity [ 21 ] = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 21 ] = veorq_u8 ( parity[ 21 ], temp ) ;
                parity [ 22 ] = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 22 ] = veorq_u8 ( parity[ 22 ], temp ) ;
                parity [ 23 ] = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 23 ] = veorq_u8 ( parity[ 23 ], temp ) ;
                parity [ 24 ] = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 24 ] = veorq_u8 ( parity[ 24 ], temp ) ;
                parity [ 25 ] = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 25 ] = veorq_u8 ( parity[ 25 ], temp ) ;
                parity [ 26 ] = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 26 ] = veorq_u8 ( parity[ 26 ], temp ) ;
                parity [ 27 ] = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 27 ] = veorq_u8 ( parity[ 27 ], temp ) ;
                parity [ 28 ] = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 28 ] = veorq_u8 ( parity[ 28 ], temp ) ;
                parity [ 29 ] = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 29 ] = veorq_u8 ( parity[ 29 ], temp ) ;
                parity [ 30 ] = vqtbl1q_u8 ( data_vec, taps [ 30 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 30 ] = veorq_u8 ( parity[ 30 ], temp ) ;
                parity [ 31 ] = vqtbl1q_u8 ( data_vec, taps [ 31 ] ) ;
                temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                parity [ 31 ] = veorq_u8 ( parity[ 31 ], temp ) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec  = vld1q_u8(&data[curSym][curPos]);
                        data_vec  = veorq_ui ( data_vec, parity [ 0 ] ) ;
                        data_vech = vshrq_n_u8 ( data_vec, 4 ) ;
                        data_vec  = vandq_u8 ( data_vec, mask0f ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 0 ] ) ;
                        parity [ 0 ] = veorq_u8 ( parity [ 0 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 1 ] ) ;
                        parity [ 1 ] = veorq_u8 ( parity [ 1 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 2 ] ) ;
                        parity [ 2 ] = veorq_u8 ( parity [ 2 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 3 ] ) ;
                        parity [ 3 ] = veorq_u8 ( parity [ 3 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 4 ] ) ;
                        parity [ 4 ] = veorq_u8 ( parity [ 4 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 5 ] ) ;
                        parity [ 5 ] = veorq_u8 ( parity [ 5 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 6 ] ) ;
                        parity [ 6 ] = veorq_u8 ( parity [ 6 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 7 ] ) ;
                        parity [ 7 ] = veorq_u8 ( parity [ 7 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 8 ] ) ;
                        parity [ 8 ] = veorq_u8 ( parity [ 8 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 9 ] ) ;
                        parity [ 9 ] = veorq_u8 ( parity [ 9 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 10 ] ) ;
                        parity [ 10 ] = veorq_u8 ( parity [ 10 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 11 ] ) ;
                        parity [ 11 ] = veorq_u8 ( parity [ 11 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 12 ] ) ;
                        parity [ 12 ] = veorq_u8 ( parity [ 12 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 13 ] ) ;
                        parity [ 13 ] = veorq_u8 ( parity [ 13 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 14 ] ) ;
                        parity [ 14 ] = veorq_u8 ( parity [ 14 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 15 ] ) ;
                        parity [ 15 ] = veorq_u8 ( parity [ 15 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 16 ] ) ;
                        parity [ 16 ] = veorq_u8 ( parity [ 16 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 17 ] ) ;
                        parity [ 17 ] = veorq_u8 ( parity [ 17 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 18 ] ) ;
                        parity [ 18 ] = veorq_u8 ( parity [ 18 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 19 ] ) ;
                        parity [ 19 ] = veorq_u8 ( parity [ 19 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 20 ] ) ;
                        parity [ 20 ] = veorq_u8 ( parity [ 20 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 21 ] ) ;
                        parity [ 21 ] = veorq_u8 ( parity [ 21 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 22 ] ) ;
                        parity [ 22 ] = veorq_u8 ( parity [ 22 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 23 ] ) ;
                        parity [ 23 ] = veorq_u8 ( parity [ 23 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 24 ] ) ;
                        parity [ 24 ] = veorq_u8 ( parity [ 24 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 25 ] ) ;
                        parity [ 25 ] = veorq_u8 ( parity [ 25 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 26 ] ) ;
                        parity [ 26 ] = veorq_u8 ( parity [ 26 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 27 ] ) ;
                        parity [ 27 ] = veorq_u8 ( parity [ 27 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 28 ] ) ;
                        parity [ 28 ] = veorq_u8 ( parity [ 28 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 29 ] ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 29 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 29 ] ) ;
                        parity [ 29 ] = veorq_u8 ( parity [ 29 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 30 ] ) ;
                        parity [ 30 ] = veorq_u8 ( parity [ 30 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 30 ] ) ;
                        parity [ 30 ] = veorq_u8 ( parity [ 30 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vec, taps [ 31 ] ) ;
                        parity [ 31 ] = veorq_u8 ( parity [ 31 ], temp ) ;
                        temp = vqtbl1q_u8 ( data_vech, tapsh [ 31 ] ) ;
                        parity [ 31 ] = veorq_u8 ( parity [ 31 ], temp ) ;
                }

                vst1q_u8 ( &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                vst1q_u8 ( &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                vst1q_u8 ( &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                vst1q_u8 ( &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
                vst1q_u8 ( &dest [ 4 ] [ curPos ], parity [ 4 ] ) ;
                vst1q_u8 ( &dest [ 5 ] [ curPos ], parity [ 5 ] ) ;
                vst1q_u8 ( &dest [ 6 ] [ curPos ], parity [ 6 ] ) ;
                vst1q_u8 ( &dest [ 7 ] [ curPos ], parity [ 7 ] ) ;
                vst1q_u8 ( &dest [ 8 ] [ curPos ], parity [ 8 ] ) ;
                vst1q_u8 ( &dest [ 9 ] [ curPos ], parity [ 9 ] ) ;
                vst1q_u8 ( &dest [ 10 ] [ curPos ], parity [ 10 ] ) ;
                vst1q_u8 ( &dest [ 11 ] [ curPos ], parity [ 11 ] ) ;
                vst1q_u8 ( &dest [ 12 ] [ curPos ], parity [ 12 ] ) ;
                vst1q_u8 ( &dest [ 13 ] [ curPos ], parity [ 13 ] ) ;
                vst1q_u8 ( &dest [ 14 ] [ curPos ], parity [ 14 ] ) ;
                vst1q_u8 ( &dest [ 15 ] [ curPos ], parity [ 15 ] ) ;
                vst1q_u8 ( &dest [ 16 ] [ curPos ], parity [ 16 ] ) ;
                vst1q_u8 ( &dest [ 17 ] [ curPos ], parity [ 17 ] ) ;
                vst1q_u8 ( &dest [ 18 ] [ curPos ], parity [ 18 ] ) ;
                vst1q_u8 ( &dest [ 19 ] [ curPos ], parity [ 19 ] ) ;
                vst1q_u8 ( &dest [ 20 ] [ curPos ], parity [ 20 ] ) ;
                vst1q_u8 ( &dest [ 21 ] [ curPos ], parity [ 21 ] ) ;
                vst1q_u8 ( &dest [ 22 ] [ curPos ], parity [ 22 ] ) ;
                vst1q_u8 ( &dest [ 23 ] [ curPos ], parity [ 23 ] ) ;
                vst1q_u8 ( &dest [ 24 ] [ curPos ], parity [ 24 ] ) ;
                vst1q_u8 ( &dest [ 25 ] [ curPos ], parity [ 25 ] ) ;
                vst1q_u8 ( &dest [ 26 ] [ curPos ], parity [ 26 ] ) ;
                vst1q_u8 ( &dest [ 27 ] [ curPos ], parity [ 27 ] ) ;
                vst1q_u8 ( &dest [ 28 ] [ curPos ], parity [ 28 ] ) ;
                vst1q_u8 ( &dest [ 29 ] [ curPos ], parity [ 29 ] ) ;
                vst1q_u8 ( &dest [ 30 ] [ curPos ], parity [ 30 ] ) ;
                vst1q_u8 ( &dest [ 31 ] [ curPos ], parity [ 31 ] ) ;
        }
        return ( curPos ) ;
}

void pc_encode_data_neon(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        switch (rows) {
        case 2: gf_2vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 3: gf_3vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 4: gf_4vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 5: gf_5vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 6: gf_6vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 7: gf_7vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 8: gf_8vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 9: gf_9vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 10: gf_10vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 11: gf_11vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 12: gf_12vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 13: gf_13vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 14: gf_14vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 15: gf_15vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 16: gf_16vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 17: gf_17vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 18: gf_18vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 19: gf_19vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 20: gf_20vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 21: gf_21vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 22: gf_22vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 23: gf_23vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 24: gf_24vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 25: gf_25vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 26: gf_26vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 27: gf_27vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 28: gf_28vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 29: gf_29vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 30: gf_30vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 31: gf_31vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        case 32: gf_32vect_lfsr_avx2_neon(len, k, g_tbls, data, coding);
                 break ;
        }
}
int pc_decode_data_neon(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
        unsigned char **coding)
{
        int newPos = 0, retry = 0 ;
        while ( ( newPos < len ) && ( retry++ < MAX_PC_RETRY ) )
        {

                switch (rows) {
                case 2: newPos = gf_2vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 3: newPos = gf_3vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 4: newPos = gf_4vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 5: newPos = gf_5vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 6: newPos = gf_6vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 7: newPos = gf_7vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 8: newPos = gf_8vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 9: newPos = gf_9vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 10: newPos = gf_10vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 11: newPos = gf_11vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 12: newPos = gf_12vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 13: newPos = gf_13vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 14: newPos = gf_14vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 15: newPos = gf_15vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 16: newPos = gf_16vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 17: newPos = gf_17vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 18: newPos = gf_18vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 19: newPos = gf_19vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 20: newPos = gf_20vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 21: newPos = gf_21vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 22: newPos = gf_22vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 23: newPos = gf_23vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 24: newPos = gf_24vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 25: newPos = gf_25vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 26: newPos = gf_26vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 27: newPos = gf_27vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 28: newPos = gf_28vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 29: newPos = gf_29vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 30: newPos = gf_30vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 31: newPos = gf_31vect_pss_neon(len, k, g_tbls, data, coding, newPos);
                         break ;
                case 32: newPos = gf_32vect_pss_neon(len, k, g_tbls, data, coding, newPos);
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
