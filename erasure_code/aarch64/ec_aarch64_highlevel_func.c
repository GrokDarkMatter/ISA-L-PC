/**************************************************************

Copyright (c) 2019 Huawei Technologies Co., Ltd.

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

Neither the name of Huawei Corporation, nor Michael H. Anderson, nor the names
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
#include "erasure_code.h"
#include <arm_neon.h>
#include "ec_base.h" // For GF tables

#include <arm_neon.h>

/*external function*/
extern void
gf_vect_dot_prod_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char *dest);
extern void
gf_2vect_dot_prod_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest);
extern void
gf_3vect_dot_prod_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest);
extern void
gf_4vect_dot_prod_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest);
extern void
gf_5vect_dot_prod_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest);
extern int
gf_vect_syndrome_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char *dest, int newPos);
extern int
gf_2vect_syndrome_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest, int newPos);
extern int
gf_3vect_syndrome_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest, int newPos);
extern int
gf_4vect_syndrome_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest, int newPos);
extern int
gf_5vect_syndrome_neon (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                        unsigned char **dest, int newPos);
extern void
gf_vect_mad_neon (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char *dest);
extern void
gf_2vect_mad_neon (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                   unsigned char **dest);
extern void
gf_3vect_mad_neon (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                   unsigned char **dest);
extern void
gf_4vect_mad_neon (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                   unsigned char **dest);
extern void
gf_5vect_mad_neon (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                   unsigned char **dest);
extern void
gf_6vect_mad_neon (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                   unsigned char **dest);

void
ec_encode_data_neon (int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                     unsigned char **coding)
{
    if (len < 16)
    {
        ec_encode_data_base (len, k, rows, g_tbls, data, coding);
        return;
    }

    while (rows > 5)
    {
        gf_5vect_dot_prod_neon (len, k, g_tbls, data, coding);
        g_tbls += 5 * k * 32;
        coding += 5;
        rows -= 5;
    }
    switch (rows)
    {
    case 5:
        gf_5vect_dot_prod_neon (len, k, g_tbls, data, coding);
        break;
    case 4:
        gf_4vect_dot_prod_neon (len, k, g_tbls, data, coding);
        break;
    case 3:
        gf_3vect_dot_prod_neon (len, k, g_tbls, data, coding);
        break;
    case 2:
        gf_2vect_dot_prod_neon (len, k, g_tbls, data, coding);
        break;
    case 1:
        gf_vect_dot_prod_neon (len, k, g_tbls, data, *coding);
        break;
    case 0:
        break;
    default:
        break;
    }
}

extern int
pc_correct (int newPos, int k, int p, unsigned char **data, int vLen);

#define MAX_PC_RETRY 1

int
ec_decode_data_neon (int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                     unsigned char **coding)
{
    int newPos = 0, retry = 0, p = rows, vSize;
    unsigned char **dest = coding, *g_orig = g_tbls;

    if (len < 16)
    {
        ec_encode_data_base (len, k, rows, g_tbls, data, &data[ k ]);
        return 0;
    }

    while ((newPos < len) && (retry++ < MAX_PC_RETRY))
    {
        coding = dest;
        rows = p;
        g_tbls = g_orig;
        while (rows >= 5)
        {
            vSize = 64;
            newPos = gf_5vect_syndrome_neon (len, k, g_tbls, data, coding, newPos);
            g_tbls += 5 * k * 32;
            coding += 5;
            rows -= 5;
            if (rows)
            {
                newPos = 0; // Start at top if more parity
            }
        }
        switch (rows)
        {
        case 4:
            vSize = 64;
            newPos = gf_4vect_syndrome_neon (len, k, g_tbls, data, coding, newPos);
            break;
        case 3:
            vSize = 64;
            newPos = gf_3vect_syndrome_neon (len, k, g_tbls, data, coding, newPos);
            break;
        case 2:
            vSize = 128;
            newPos = gf_2vect_syndrome_neon (len, k, g_tbls, data, coding, newPos);
            break;
        case 1:
            vSize = 128;
            newPos = gf_vect_syndrome_neon (len, k, g_tbls, data, *coding, newPos);
            break;
        case 0:
        default:
            break;
        }
        // printf ( "Newpos = %d Retry = %d\n", newPos, retry ) ;
        //  If premature stop, correct data
        if (newPos < len)
        {
            if (pc_correct (newPos, k, p, data, vSize))
            {
                return (newPos);
            }
        }
    }
    return (newPos);
}

void
ec_encode_data_update_neon (int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                            unsigned char *data, unsigned char **coding)
{
    if (len < 16)
    {
        ec_encode_data_update_base (len, k, rows, vec_i, g_tbls, data, coding);
        return;
    }
    while (rows > 6)
    {
        gf_6vect_mad_neon (len, k, vec_i, g_tbls, data, coding);
        g_tbls += 6 * k * 32;
        coding += 6;
        rows -= 6;
    }
    switch (rows)
    {
    case 6:
        gf_6vect_mad_neon (len, k, vec_i, g_tbls, data, coding);
        break;
    case 5:
        gf_5vect_mad_neon (len, k, vec_i, g_tbls, data, coding);
        break;
    case 4:
        gf_4vect_mad_neon (len, k, vec_i, g_tbls, data, coding);
        break;
    case 3:
        gf_3vect_mad_neon (len, k, vec_i, g_tbls, data, coding);
        break;
    case 2:
        gf_2vect_mad_neon (len, k, vec_i, g_tbls, data, coding);
        break;
    case 1:
        gf_vect_mad_neon (len, k, vec_i, g_tbls, data, *coding);
        break;
    case 0:
        break;
    }
}

/* SVE */
extern void
gf_vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                      unsigned char *dest);
extern void
gf_2vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_3vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_4vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_5vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_6vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_7vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_8vect_dot_prod_sve (int len, int vlen, unsigned char *gftbls, unsigned char **src,
                       unsigned char **dest);
extern void
gf_vect_mad_sve (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                 unsigned char *dest);
extern void
gf_2vect_mad_sve (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_3vect_mad_sve (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_4vect_mad_sve (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_5vect_mad_sve (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);
extern void
gf_6vect_mad_sve (int len, int vec, int vec_i, unsigned char *gftbls, unsigned char *src,
                  unsigned char **dest);

void
ec_encode_data_sve (int len, int k, int rows, unsigned char *g_tbls, unsigned char **data,
                    unsigned char **coding)
{
    if (len < 16)
    {
        ec_encode_data_base (len, k, rows, g_tbls, data, coding);
        return;
    }

    while (rows > 11)
    {
        gf_6vect_dot_prod_sve (len, k, g_tbls, data, coding);
        g_tbls += 6 * k * 32;
        coding += 6;
        rows -= 6;
    }

    switch (rows)
    {
    case 11:
        /* 7 + 4 */
        gf_7vect_dot_prod_sve (len, k, g_tbls, data, coding);
        g_tbls += 7 * k * 32;
        coding += 7;
        gf_4vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 10:
        /* 6 + 4 */
        gf_6vect_dot_prod_sve (len, k, g_tbls, data, coding);
        g_tbls += 6 * k * 32;
        coding += 6;
        gf_4vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 9:
        /* 5 + 4 */
        gf_5vect_dot_prod_sve (len, k, g_tbls, data, coding);
        g_tbls += 5 * k * 32;
        coding += 5;
        gf_4vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 8:
        /* 4 + 4 */
        gf_4vect_dot_prod_sve (len, k, g_tbls, data, coding);
        g_tbls += 4 * k * 32;
        coding += 4;
        gf_4vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 7:
        gf_7vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 6:
        gf_6vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 5:
        gf_5vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 4:
        gf_4vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 3:
        gf_3vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 2:
        gf_2vect_dot_prod_sve (len, k, g_tbls, data, coding);
        break;
    case 1:
        gf_vect_dot_prod_sve (len, k, g_tbls, data, *coding);
        break;
    default:
        break;
    }
}

void
ec_encode_data_update_sve (int len, int k, int rows, int vec_i, unsigned char *g_tbls,
                           unsigned char *data, unsigned char **coding)
{
    if (len < 16)
    {
        ec_encode_data_update_base (len, k, rows, vec_i, g_tbls, data, coding);
        return;
    }
    while (rows > 6)
    {
        gf_6vect_mad_sve (len, k, vec_i, g_tbls, data, coding);
        g_tbls += 6 * k * 32;
        coding += 6;
        rows -= 6;
    }
    switch (rows)
    {
    case 6:
        gf_6vect_mad_sve (len, k, vec_i, g_tbls, data, coding);
        break;
    case 5:
        gf_5vect_mad_sve (len, k, vec_i, g_tbls, data, coding);
        break;
    case 4:
        gf_4vect_mad_sve (len, k, vec_i, g_tbls, data, coding);
        break;
    case 3:
        gf_3vect_mad_sve (len, k, vec_i, g_tbls, data, coding);
        break;
    case 2:
        gf_2vect_mad_sve (len, k, vec_i, g_tbls, data, coding);
        break;
    case 1:
        gf_vect_mad_sve (len, k, vec_i, g_tbls, data, *coding);
        break;
    default:
        break;
    }
}
