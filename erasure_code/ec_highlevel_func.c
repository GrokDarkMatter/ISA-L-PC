/**********************************************************************
  Copyright(c) 2011-2019 Intel Corporation All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************/
#include <limits.h>
#include "erasure_code.h"
#include <immintrin.h>
#include <x86intrin.h>
#include "ec_base.h" /* for GF tables */

#if __x86_64__ || __i386__ || _M_X64 || _M_IX86
void
ec_encode_data_sse(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                   unsigned char **coding)
{

        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows >= 6) {
                gf_6vect_dot_prod_sse(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_sse(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_sse(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_sse(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_sse(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_sse(len, k, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

void
ec_encode_data_avx(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                   unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows >= 6) {
                gf_6vect_dot_prod_avx(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_avx(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_avx(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_avx(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_avx(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_avx(len, k, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

void
ec_encode_data_avx2(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                    unsigned char **coding)
{

        if (len < 32) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows >= 6) {
                gf_6vect_dot_prod_avx2(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_avx2(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_avx2(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_avx2(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_avx2(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_avx2(len, k, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

extern int
gf_vect_dot_prod_avx512(int len, int k, unsigned char *g_tbls, unsigned char **data,
                        unsigned char *dest);
extern int
gf_2vect_dot_prod_avx512(int len, int k, unsigned char *g_tbls, unsigned char **data,
                         unsigned char **coding);
extern int
gf_3vect_dot_prod_avx512(int len, int k, unsigned char *g_tbls, unsigned char **data,
                         unsigned char **coding);
extern int
gf_4vect_dot_prod_avx512(int len, int k, unsigned char *g_tbls, unsigned char **data,
                         unsigned char **coding);
extern int
gf_5vect_dot_prod_avx512(int len, int k, unsigned char *g_tbls, unsigned char **data,
                         unsigned char **coding);
extern int
gf_6vect_dot_prod_avx512(int len, int k, unsigned char *g_tbls, unsigned char **data,
                         unsigned char **coding);
extern void
gf_vect_mad_avx512(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                   unsigned char *dest);
extern void
gf_2vect_mad_avx512(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                    unsigned char **dest);
extern void
gf_3vect_mad_avx512(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                    unsigned char **dest);
extern void
gf_4vect_mad_avx512(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                    unsigned char **dest);
extern void
gf_5vect_mad_avx512(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                    unsigned char **dest);
extern void
gf_6vect_mad_avx512(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                    unsigned char **dest);

void
ec_encode_data_avx512(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                      unsigned char **coding)
{

        if (len < 64) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows >= 6) {
                gf_6vect_dot_prod_avx512(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_avx512(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_avx512(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_avx512(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_avx512(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_avx512(len, k, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

void
ec_encode_data_update_avx512(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                             unsigned char *data, unsigned char **coding)
{
        if (len < 64) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }

        while (rows >= 6) {
                gf_6vect_mad_avx512(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_mad_avx512(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_avx512(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_avx512(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_avx512(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_avx512(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

extern void
gf_vect_dot_prod_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                             unsigned char *dest);
extern void
gf_2vect_dot_prod_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                              unsigned char **coding);
extern void
gf_3vect_dot_prod_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                              unsigned char **coding);
extern void
gf_4vect_dot_prod_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                              unsigned char **coding);
extern void
gf_5vect_dot_prod_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                              unsigned char **coding);
extern void
gf_6vect_dot_prod_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                              unsigned char **coding);
extern int
gf_vect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet);
extern int
gf_2vect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet);
extern int
gf_3vect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet);
extern int
gf_4vect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet);
extern int
gf_5vect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet);
extern int
gf_6vect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet);
extern int
gf_nvect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet, int synCount);

        extern void
gf_vect_mad_avx512_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                        unsigned char *dest);
extern void
gf_2vect_mad_avx512_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                         unsigned char **dest);
extern void
gf_3vect_mad_avx512_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                         unsigned char **dest);
extern void
gf_4vect_mad_avx512_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                         unsigned char **dest);
extern void
gf_5vect_mad_avx512_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                         unsigned char **dest);
extern void
gf_6vect_mad_avx512_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                         unsigned char **dest);

extern void
gf_vect_dot_prod_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                           unsigned char *dest);
extern void
gf_2vect_dot_prod_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                            unsigned char **coding);
extern void
gf_3vect_dot_prod_avx2_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
                            unsigned char **coding);
extern void
gf_vect_mad_avx2_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                      unsigned char *dest);
extern void
gf_2vect_mad_avx2_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                       unsigned char **dest);
extern void
gf_3vect_mad_avx2_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                       unsigned char **dest);
extern void
gf_4vect_mad_avx2_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                       unsigned char **dest);
extern void
gf_5vect_mad_avx2_gfni(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                       unsigned char **dest);

void
ec_init_tables_gfni(int k, int rows, unsigned char *a, unsigned char *g_tbls)
{
        int i, j;

        uint64_t *g64 = (uint64_t *) g_tbls;

        for (i = 0; i < rows; i++)
                for (j = 0; j < k; j++)
                        *(g64++) = gf_table_gfni[*a++];
}

extern int
gf_nvect_syndrome_avx512_gfni(int len, int k, unsigned char *g_tbls, unsigned char **data,
        int offSet, int synCount)
{
        int curSym, curRow, curPos = 0 ;              // Loop counters
        unsigned char * cur_g ;                       // Affine table pointer
        __m512i result, aff_vec, data_vec ;           // Working registers
        __m512i parity [ 32 ] ;                       // Parity registers
        __mmask8 mask ;                               // Mask used to test for zero

        // Loop through all the bytes, 64 at a time
        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                // Initialize affine table pointer
                cur_g = g_tbls ;

                // Initialize the parities
                result = _mm512_setzero_si512() ;
                for ( curSym = 0 ; curSym < synCount ; curSym ++ )
                {
                        parity [ curSym ] = result ;

                }

                // Loop for each symbol
                for ( curSym = 0 ; curSym < k ; curSym ++ )
                {
                        // Load data for current symbol
                        data_vec = _mm512_load_si512( (__m512i *) data [ curSym ] ) ;
                        for ( curRow = 0 ; curRow < synCount ; curRow ++ ) 
                        {
                                // Extend the 8x8 affine to 512 bytes
                                aff_vec = _mm512_broadcast_i32x2(*( __m128i * ) ( cur_g + ( curRow * 8 * k ) ) ); 
                                // Compute the result of the data multiplied by the affine
                                result = _mm512_gf2p8affine_epi64_epi8(data_vec, aff_vec, 0) ;
                                //Add in the current parity row
                                result = _mm512_xor_si512 ( result, parity [ curRow ] ) ;
                                // And now save it back to memory
                                parity [ curRow ] = result ;
                        }
                        // Move affine table forward by one entry
                        cur_g += 8 ;
                }


                // Now check for zero
                result = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                for ( curSym = 2 ; curSym < synCount ; curSym ++ )
                {
                        result = _mm512_or_si512 ( result, parity [ curSym ] ) ;
                }
                mask = _mm512_test_epi64_mask ( result, result ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) ) 
                {
                        for ( curSym = 0 ; curSym < synCount ; curSym ++ )
                        {
                                _mm512_store_si512( (__m512i *) data [ curSym + k ], parity [ curSym ] ) ;
                        }
                        return curPos ;
                }
        }
        return curPos ;
}

void pc_correct_AVX512_gfni ( int newPos, int k, int p, unsigned char ** data )
{
        __m512i ldata ;
        __mmask64 mask1 ;
        unsigned long long offSet ;
        unsigned char eVal, eLoc, synDromes [ 254 ] ;

        ldata = _mm512_load_si512( data[k] ) ;
        mask1 = _mm512_test_epi8_mask ( ldata, ldata ) ;
        offSet = _tzcnt_u64 ( mask1 ) ;
        //printf ( "k = %d Offset = %lld\n", k, offSet ) ;

        for ( eLoc = 0 ; eLoc < p ; eLoc ++ )
        {
                synDromes [ eLoc ] = data [ k - eLoc - 1 + p ] [ offSet ] ;
        }
        eVal = synDromes [ 0 ] ;
        eLoc = synDromes [ 1 ] ;
        eLoc = gf_mul ( eLoc, gf_inv ( eVal ) ) ;
        eLoc = gflog_base [ eLoc ] ;
        if ( eLoc == 255 )
        {
                eLoc = 0 ;
        }
        //printf ( "Error = %d Symbol location = %d Bufpos = %lld\n", eVal, eLoc , newPos + offSet ) ;

        // Correct the error
        if ( eLoc < k )
        {
                data [ k - eLoc - 1 ] [ newPos + offSet ] ^= eVal ;
        }

        return ;
}

#define MAX_PC_RETRY 2

int
ec_decode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data)
{
        int newPos = 0, retry = 0 ;

        if (rows > 6)
        {
                newPos = gf_nvect_syndrome_avx512_gfni ( len, k, g_tbls, data, newPos, rows ) ;
                while ( ( newPos < len ) && ( ++retry < MAX_PC_RETRY ) )
                {
                        pc_correct_AVX512_gfni ( newPos, k, rows, data ) ;
                        len -= newPos ;
                        newPos = gf_nvect_syndrome_avx512_gfni ( len, k, g_tbls, data, newPos, rows ) ;
                }
                return ( newPos ) ;
        }

        switch (rows)
        {
        case 6:
                newPos = gf_6vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                while ( ( newPos < len ) && ( ++retry < MAX_PC_RETRY ) )
                {
                        pc_correct_AVX512_gfni ( newPos, k, rows, data ) ;
                        len -= newPos ;
                        newPos = gf_6vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos) ;
                }
        case 5:
                newPos = gf_5vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                while ( ( newPos < len ) && ( ++retry < MAX_PC_RETRY ) )
                {
                        pc_correct_AVX512_gfni ( newPos, k, rows, data ) ;
                        len -= newPos ;
                        newPos = gf_5vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                }
                break;
        case 4:
                newPos = gf_4vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                while ( ( newPos < len ) && ( ++retry < MAX_PC_RETRY ) )
                {
                        pc_correct_AVX512_gfni ( newPos, k, rows, data ) ;
                        len -= newPos ;
                        newPos = gf_4vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos);
                }
                break;
        case 3:
                newPos = gf_3vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                while ( ( newPos < len ) && ( ++retry < MAX_PC_RETRY ) )
                {
                        pc_correct_AVX512_gfni ( newPos, k, rows, data ) ;
                        len -= newPos ;
                        newPos = gf_3vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                }
                break;
        case 2:
                newPos = gf_2vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ); 
                while ( ( newPos < len ) && ( ++retry < MAX_PC_RETRY ) )
                {
                        pc_correct_AVX512_gfni ( newPos, k, rows, data ) ;
                        len -= newPos ;
                        newPos = gf_2vect_syndrome_avx512_gfni(len, k, g_tbls, data, newPos ) ;
                }
                break;
        case 1:
                newPos = len ;
                break;
        case 0:
        default:
                break;
        }
        return ( newPos ) ;
}

void
ec_encode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                           unsigned char **coding)
{

        while (rows >= 6) {
                gf_6vect_dot_prod_avx512_gfni(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 8;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_avx512_gfni(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_avx512_gfni(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_avx512_gfni(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_avx512_gfni(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_avx512_gfni(len, k, g_tbls, data, *coding);
                break;
        case 0:
        default:
                break;
        }
}

void
ec_encode_data_avx2_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                         unsigned char **coding)
{
        while (rows >= 3) {
                gf_3vect_dot_prod_avx2_gfni(len, k, g_tbls, data, coding);
                g_tbls += 3 * k * 8;
                coding += 3;
                rows -= 3;
        }
        switch (rows) {
        case 2:
                gf_2vect_dot_prod_avx2_gfni(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_avx2_gfni(len, k, g_tbls, data, *coding);
                break;
        case 0:
        default:
                break;
        }
}

void
ec_encode_data_update_avx512_gfni(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                                  unsigned char *data, unsigned char **coding)
{
        while (rows >= 6) {
                gf_6vect_mad_avx512_gfni(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 8;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 5:
                gf_5vect_mad_avx512_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_avx512_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_avx512_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_avx512_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_avx512_gfni(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
        default:
                break;
        }
}

void
ec_encode_data_update_avx2_gfni(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                                unsigned char *data, unsigned char **coding)
{
        while (rows >= 5) {
                gf_5vect_mad_avx2_gfni(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 5 * k * 8;
                coding += 5;
                rows -= 5;
        }
        switch (rows) {
        case 4:
                gf_4vect_mad_avx2_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_avx2_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_avx2_gfni(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_avx2_gfni(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
        default:
                break;
        }
}

#if __WORDSIZE == 64 || _WIN64 || __x86_64__

void
ec_encode_data_update_sse(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                          unsigned char *data, unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }

        while (rows > 6) {
                gf_6vect_mad_sse(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_sse(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_sse(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_sse(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_sse(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_sse(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_sse(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

void
ec_encode_data_update_avx(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                          unsigned char *data, unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }
        while (rows > 6) {
                gf_6vect_mad_avx(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_avx(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_avx(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_avx(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_avx(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_avx(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_avx(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

void
ec_encode_data_update_avx2(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                           unsigned char *data, unsigned char **coding)
{
        if (len < 32) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }
        while (rows > 6) {
                gf_6vect_mad_avx2(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_avx2(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_avx2(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_avx2(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_avx2(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_avx2(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_avx2(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

#endif //__WORDSIZE == 64 || _WIN64 || __x86_64__
#endif //__x86_64__  || __i386__ || _M_X64 || _M_IX86
