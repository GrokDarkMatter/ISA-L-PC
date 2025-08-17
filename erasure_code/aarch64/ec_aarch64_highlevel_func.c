/**************************************************************
  Copyright (c) 2019 Huawei Technologies Co., Ltd.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Huawei Corporation nor the names of its
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
#include "erasure_code.h"
#include <arm_neon.h>
#include "ec_base.h" // For GF tables

/*external function*/
extern void
gf_vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char *dest);
extern void
gf_2vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_3vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_4vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_5vect_dot_prod_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern int
gf_vect_syndrome_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char *dest, int newPos);
extern int
gf_2vect_syndrome_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest, int newPos);
extern int
gf_3vect_syndrome_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest, int newPos);
extern int
gf_4vect_syndrome_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest, int newPos);
extern int
gf_5vect_syndrome_neon(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest, int newPos);
extern void
gf_vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char *dest);
extern void
gf_2vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_3vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_4vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_5vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_6vect_mad_neon(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);

void
ec_encode_data_neon(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                    unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows > 5) {
                gf_5vect_dot_prod_neon(len, k, g_tbls, data, coding);
                g_tbls += 5 * k * 32;
                coding += 5;
                rows -= 5;
        }
        switch (rows) {
        case 5:
                gf_5vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_neon(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_neon(len, k, g_tbls, data, *coding);
                break;
        case 0:
                break;
        default:
                break;
        }
}
#ifdef NDEF
int gf_nvect_syndrome_neon(int len, int k, unsigned char *g_tbls,
  unsigned char **data, int offSet, int synCount)
{
        int curSym, curRow, curPos = 0; // Loop counters
        unsigned char *cur_g; // Affine table pointer
        uint8x16_t result, data_vec; // Working registers
        uint8x16_t parity[32]; // Parity registers


        // Loop through all the bytes, 16 at a time (NEON processes 128 bits)
        for (curPos = 0; curPos < len; curPos += 16)
        {
                // Initialize affine table pointer
                cur_g = g_tbls;

                // Initialize the parities
                result = vdupq_n_u8(0);
                for (curSym = 0; curSym < synCount; curSym++)
                {
                        parity[curSym] = result;
                }

                // Loop for each symbol
                for (curSym = 0; curSym < k; curSym++)
                {
                        // Load data for current symbol (unaligned load)
                        data_vec = vld1q_u8(data[curSym] + curPos);

                        for (curRow = 0; curRow < synCount; curRow++)
                        {
                                // Load 8-byte affine table entry
                                //uint8x8_t aff_table = vld1_u8(cur_g + (curRow * 8 * k));
                                // Replicate affine table across 16 bytes
                                //aff_vec = vcombine_u8(aff_table, aff_table);
                                // Compute the result of the data multiplied by the affine
                                //result = gf2p8affine_neon(data_vec, aff_table);
                                // Add in the current parity row
                                result = veorq_u8(result, parity[curRow]);
                                // Save back to parity
                                parity[curRow] = result;
                        }
                        // Move affine table forward by one entry
                        cur_g += 8;
                }

                // Check for non-zero parity
                result = vorrq_u8(parity[0], parity[1]);
                for (curSym = 2; curSym < synCount; curSym++) 
                {
                        result = vorrq_u8(result, parity[curSym]);
                }

                // Test if result is non-zero
                 //uint8x16_t mask = vceqq_u8(result, vdupq_n_u8(0));
                uint32x4_t vec32 = vreinterpretq_u32_u8(result); // Reinterpret as uint32x4_t
                //return vmaxvq_u32(vec32) == 0; // Returns 1 if all zeros, 0 otherwise
                //mask = vmvnq_u8(mask); // Invert: non-zero bytes -> 0xFF
                //uint16x8_t mask16 = vmovn_high_u16(vmovn_u16(mask), mask);
                //bitmask = vgetq_lane_u16(mask16, 0);

                if (vmaxvq_u32(vec32) == 0)
                {
                        // Store non-zero parities to output
                        for (curSym = 0; curSym < synCount; curSym++)
                        {
                                vst1q_u8(data[curSym + k] + curPos, parity[curSym]);
                        }
                        return curPos;
                }
        }
        return curPos;
}
#endif
int pc_correct ( int newPos, int k, int p, unsigned char ** data, int vLen )
{
        int offSet = 0 ;
        unsigned char eVal, eLoc, synDromes [ 254 ] ;

        // Scan for first non-zero byte in vector
        while ( data [ k ] [ offSet ] == 0 ) 
        {
                offSet ++ ;
                if ( offSet == vLen )
                {
                        printf ( "Zeroes found\n" ) ;
                        return 1 ;
                }
        }

        // Gather up the syndromes
        for ( eLoc = 0 ; eLoc < p ; eLoc ++ )
        {
                synDromes [ eLoc ] = data [ k + p - eLoc - 1 ] [ offSet ] ;
        }

        // LSB has parity, for single error this equals error value
        eVal = synDromes [ 0 ] ;
        // Compute error location is log2(syndrome[1]/syndrome[0])
        eLoc = synDromes [ 1 ] ;
        eLoc = gf_mul ( eLoc, gf_inv ( eVal ) ) ;
        eLoc = gflog_base [ eLoc ] ;
        // First entry in log table
        if ( eLoc == 255 )
        {
                eLoc = 0 ;
        }
        printf ( "Error = %d Symbol location = %d Bufpos = %d\n", eVal, eLoc , newPos + offSet ) ;

        // Correct the error if it's within bounds
        if ( eLoc < k )
        {
                data [ k - eLoc - 1 ] [ newPos + offSet ] ^= eVal ;
                return 0 ;
        }

        return 1 ;
}

#define MAX_PC_RETRY 1

int
ec_decode_data_neon(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                    unsigned char **coding)
{
        int newPos = 0, retry = 0, p = rows, vSize ;
        unsigned char ** dest = coding, *g_orig = g_tbls ;
        
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, &data [ k ]);
                return 0;
        }

        while ( ( newPos < len ) && ( retry++ < MAX_PC_RETRY ) )
        {
                coding = dest ;
                rows = p ;
                g_tbls = g_orig ;
                while (rows >= 5) 
                {
                        vSize = 64 ;
                        newPos = gf_5vect_syndrome_neon(len, k, g_tbls, data, coding, newPos);
                        g_tbls += 5 * k * 32;
                        coding += 5;
                        rows -= 5;
                        if ( rows )
                        {
                                newPos = 0 ; // Start at top if more parity
                        }
                }
                switch (rows) {
                case 4:
                        vSize = 64 ;
                        newPos = gf_4vect_syndrome_neon(len, k, g_tbls, data, coding, newPos);
                        break;
                case 3:
                        vSize = 64 ;
                        newPos = gf_3vect_syndrome_neon(len, k, g_tbls, data, coding, newPos);
                        break;
                case 2:
                        vSize = 128 ;
                        newPos = gf_2vect_syndrome_neon(len, k, g_tbls, data, coding, newPos);
                        break;
                case 1:
                        vSize = 128 ;
                        newPos = gf_vect_syndrome_neon(len, k, g_tbls, data, *coding, newPos);
                        break;
                case 0:
                default:
                        break;
                }
                //printf ( "Newpos = %d Retry = %d\n", newPos, retry ) ;
                // If premature stop, correct data
                if ( newPos < len )
                {
                        if ( pc_correct ( newPos, k, p, data, vSize ) )
                        {
                                return ( newPos ) ;
                        }
                }
        }
        return ( newPos ) ;
}

void
ec_encode_data_update_neon(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                           unsigned char *data, unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }
        while (rows > 6) {
                gf_6vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_neon(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_neon(len, k, vec_i, g_tbls, data, *coding);
                break;
        case 0:
                break;
        }
}

/* SVE */
extern void
gf_vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                     unsigned char *dest);
extern void
gf_2vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_3vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_4vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_5vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_6vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_7vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_8vect_dot_prod_sve(int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char **dest);
extern void
gf_vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                unsigned char *dest);
extern void
gf_2vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_3vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_4vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_5vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);
extern void
gf_6vect_mad_sve(int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char **dest);

void
ec_encode_data_sve(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                   unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_base(len, k, rows, g_tbls, data, coding);
                return;
        }

        while (rows > 11) {
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }

        switch (rows) {
        case 11:
                /* 7 + 4 */
                gf_7vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 7 * k * 32;
                coding += 7;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 10:
                /* 6 + 4 */
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 9:
                /* 5 + 4 */
                gf_5vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 5 * k * 32;
                coding += 5;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 8:
                /* 4 + 4 */
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                g_tbls += 4 * k * 32;
                coding += 4;
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 7:
                gf_7vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 6:
                gf_6vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_dot_prod_sve(len, k, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_dot_prod_sve(len, k, g_tbls, data, *coding);
                break;
        default:
                break;
        }
}

void
ec_encode_data_update_sve(int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                          unsigned char *data, unsigned char **coding)
{
        if (len < 16) {
                ec_encode_data_update_base(len, k, rows, vec_i, g_tbls, data, coding);
                return;
        }
        while (rows > 6) {
                gf_6vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                g_tbls += 6 * k * 32;
                coding += 6;
                rows -= 6;
        }
        switch (rows) {
        case 6:
                gf_6vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 5:
                gf_5vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 4:
                gf_4vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 3:
                gf_3vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 2:
                gf_2vect_mad_sve(len, k, vec_i, g_tbls, data, coding);
                break;
        case 1:
                gf_vect_mad_sve(len, k, vec_i, g_tbls, data, *coding);
                break;
        default:
                break;
        }
}
