/**********************************************************************
Copyright (c) 2011-2024 Intel Corporation.
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

Neither the name of Intel Corporation, nor Michael H. Anderson, nor the names
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

SPDX-License-Identifier: BSD-3-Clause
**********************************************************************/

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "erasure_code.h"

#define MAX_CHECK 63 /* Size is limited by using uint64_t to represent subsets */
#define M_MAX     0x20
#define K_MAX     0x10
#define ROWS      M_MAX
#define COLS      K_MAX

static inline uint64_t
min(const uint64_t a, const uint64_t b)
{
        if (a <= b)
                return a;
        else
                return b;
}

void
gen_sub_matrix(unsigned char *out_matrix, const uint64_t dim, unsigned char *in_matrix,
               const uint64_t rows, const uint64_t cols, const uint64_t row_indicator,
               const uint64_t col_indicator)
{
        uint64_t i, j, r, s;

        for (i = 0, r = 0; i < rows; i++) {
                if (!(row_indicator & ((uint64_t) 1 << i)))
                        continue;

                for (j = 0, s = 0; j < cols; j++) {
                        if (!(col_indicator & ((uint64_t) 1 << j)))
                                continue;
                        out_matrix[dim * r + s] = in_matrix[cols * i + j];
                        s++;
                }
                r++;
        }
}

/* Gosper's Hack */
uint64_t
next_subset(uint64_t *subset, uint64_t element_count, uint64_t subsize)
{
        uint64_t tmp1 = *subset & -*subset;
        uint64_t tmp2 = *subset + tmp1;
        *subset = (((*subset ^ tmp2) >> 2) / tmp1) | tmp2;
        if (*subset & (((uint64_t) 1 << element_count))) {
                /* Overflow on last subset */
                *subset = ((uint64_t) 1 << subsize) - 1;
                return 1;
        }

        return 0;
}

int
are_submatrices_singular(unsigned char *vmatrix, const uint64_t rows, const uint64_t cols)
{
        unsigned char matrix[COLS * COLS];
        unsigned char invert_matrix[COLS * COLS];
        uint64_t subsize;

        /* Check all square subsize x subsize submatrices of the rows x cols
         * vmatrix for singularity*/
        for (subsize = 1; subsize <= min(rows, cols); subsize++) {
                const uint64_t subset_init = (1ULL << subsize) - 1ULL;
                uint64_t col_indicator = subset_init;
                do {
                        uint64_t row_indicator = subset_init;
                        do {
                                gen_sub_matrix(matrix, subsize, vmatrix, rows, cols, row_indicator,
                                               col_indicator);
                                if (gf_invert_matrix(matrix, invert_matrix, (int) subsize))
                                        return 1;

                        } while (next_subset(&row_indicator, rows, subsize) == 0);
                } while (next_subset(&col_indicator, cols, subsize) == 0);
        }

        return 0;
}

int
main(int argc, char **argv)
{
        unsigned char vmatrix[(ROWS + COLS) * COLS];
        uint64_t rows, cols;

        if (K_MAX > MAX_CHECK) {
                printf("K_MAX too large for this test\n");
                return 0;
        }
        if (M_MAX > MAX_CHECK) {
                printf("M_MAX too large for this test\n");
                return 0;
        }
        if (M_MAX < K_MAX) {
                printf("M_MAX must be smaller than K_MAX");
                return 0;
        }

        printf("Checking gen_poly_matrix for k <= %d and m <= %d.\n", K_MAX, M_MAX);
        printf("gen_poly_matrix creates erasure codes for:\n");

        for (cols = 1; cols <= K_MAX; cols++) {
                for (rows = 1; rows <= M_MAX - cols; rows++) {
                        // Generate a Polynomial Matrix
                        gf_gen_poly_matrix(vmatrix, rows + cols, cols);
                        // Verify the Polynomial Code portion of vmatrix contains no
                        // singular submatrix
                        if (are_submatrices_singular(&vmatrix[cols * cols], rows, cols))
                                break;
                }
                printf("   k = %2u, m <= %2u \n", (unsigned) cols, (unsigned) (rows + cols - 1));
        }
        return 0;
}
