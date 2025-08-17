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

#include "../include/aarch64_label.h"

.text

.global cdecl(gf_vect_dot_prod_neon)
.global cdecl(gf_vect_syndrome_neon)
#ifndef __APPLE__
.type gf_vect_dot_prod_neon, %function
.type gf_vect_syndrome_neon, %function
#endif

/* arguments */
x_len		.req	x0
x_vec		.req	x1
x_tbl		.req	x2
x_src		.req	x3
x_dest1		.req	x4

/* returns */
w_ret		.req	w0

/* local variables */
x_vec_i		.req	x5
x_ptr		.req	x6
x_pos		.req	x7
x_tmp		.req	x8
w_tmp		.req	w8
x_tbl1		.req	x9

/* vectors */
v_tmp1      .req    v0
s_tmp1		.req	s0
v_gft1_lo	.req	v0
v_gft1_hi	.req	v1
q_gft1_lo	.req	q0
q_gft1_hi	.req	q1
v_mask0f	.req	v2
q_mask0f	.req	q2

v_data_0	.req	v8
v_data_1	.req	v9
v_data_2	.req	v10
v_data_3	.req	v11
v_data_4	.req	v12
v_data_5	.req	v13
v_data_6	.req	v14
v_data_7	.req	v15
q_data_0	.req	q8
q_data_1	.req	q9
q_data_2	.req	q10
q_data_3	.req	q11
q_data_4	.req	q12
q_data_5	.req	q13
q_data_6	.req	q14
q_data_7	.req	q15

v_data_0_lo	.req	v16
v_data_1_lo	.req	v17
v_data_2_lo	.req	v18
v_data_3_lo	.req	v19
v_data_4_lo	.req	v20
v_data_5_lo	.req	v21
v_data_6_lo	.req	v22
v_data_7_lo	.req	v23
v_data_0_hi	.req	v_data_0
v_data_1_hi	.req	v_data_1
v_data_2_hi	.req	v_data_2
v_data_3_hi	.req	v_data_3
v_data_4_hi	.req	v_data_4
v_data_5_hi	.req	v_data_5
v_data_6_hi	.req	v_data_6
v_data_7_hi	.req	v_data_7

v_p0		.req	v24
v_p1		.req	v25
v_p2		.req	v26
v_p3		.req	v27
v_p4		.req	v28
v_p5		.req	v29
v_p6		.req	v30
v_p7		.req	v31
q_p0		.req	q24
q_p1		.req	q25
q_p2		.req	q26
q_p3		.req	q27
q_p4		.req	q28
q_p5		.req	q29
q_p6		.req	q30
q_p7		.req	q31

v_p		.req	v_p0
q_p		.req	q_p0
v_data		.req	v_p1
q_data		.req	q_p1
v_data_lo	.req	v_p2
v_data_hi	.req	v_p3


cdecl(gf_vect_dot_prod_neon):
	/* less than 16 bytes, return_fail */
	cmp	x_len, #16
	blt	.return_fail

	movi	v_mask0f.16b, #0x0f
	mov	x_pos, #0

	lsl	x_vec, x_vec, #3

.Lloop128_init:
	/* less than 128 bytes, goto Lloop16_init */
	cmp	x_len, #128
	blt	.Lloop16_init

	/* save d8 ~ d15 to stack */
	sub	sp, sp, #64
	stp	d8, d9, [sp]
	stp	d10, d11, [sp, #16]
	stp	d12, d13, [sp, #32]
	stp	d14, d15, [sp, #48]

	sub	x_len, x_len, #128

.Lloop128:
	movi	v_p0.16b, #0
	movi	v_p1.16b, #0
	movi	v_p2.16b, #0
	movi	v_p3.16b, #0
	movi	v_p4.16b, #0
	movi	v_p5.16b, #0
	movi	v_p6.16b, #0
	movi	v_p7.16b, #0

	mov	x_tbl1, x_tbl
	mov	x_vec_i, #0

.Lloop128_vects:
	ldr	x_ptr, [x_src, x_vec_i]
	add	x_vec_i, x_vec_i, #8
	add	x_ptr, x_ptr, x_pos

	ldp	q_gft1_lo, q_gft1_hi, [x_tbl1], #32

	ldp	q_data_0, q_data_1, [x_ptr], #32
	ldp	q_data_2, q_data_3, [x_ptr], #32
	ldp	q_data_4, q_data_5, [x_ptr], #32
	ldp	q_data_6, q_data_7, [x_ptr]

	prfm	pldl1keep, [x_tbl1]
	prfm	pldl1strm, [x_ptr]

	and	v_data_0_lo.16b, v_data_0.16b, v_mask0f.16b
	and	v_data_1_lo.16b, v_data_1.16b, v_mask0f.16b
	and	v_data_2_lo.16b, v_data_2.16b, v_mask0f.16b
	and	v_data_3_lo.16b, v_data_3.16b, v_mask0f.16b
	and	v_data_4_lo.16b, v_data_4.16b, v_mask0f.16b
	and	v_data_5_lo.16b, v_data_5.16b, v_mask0f.16b
	and	v_data_6_lo.16b, v_data_6.16b, v_mask0f.16b
	and	v_data_7_lo.16b, v_data_7.16b, v_mask0f.16b

	ushr	v_data_0_hi.16b, v_data_0.16b, #4
	ushr	v_data_1_hi.16b, v_data_1.16b, #4
	ushr	v_data_2_hi.16b, v_data_2.16b, #4
	ushr	v_data_3_hi.16b, v_data_3.16b, #4
	ushr	v_data_4_hi.16b, v_data_4.16b, #4
	ushr	v_data_5_hi.16b, v_data_5.16b, #4
	ushr	v_data_6_hi.16b, v_data_6.16b, #4
	ushr	v_data_7_hi.16b, v_data_7.16b, #4

	tbl	v_data_0_lo.16b, {v_gft1_lo.16b}, v_data_0_lo.16b
	tbl	v_data_1_lo.16b, {v_gft1_lo.16b}, v_data_1_lo.16b
	tbl	v_data_2_lo.16b, {v_gft1_lo.16b}, v_data_2_lo.16b
	tbl	v_data_3_lo.16b, {v_gft1_lo.16b}, v_data_3_lo.16b
	tbl	v_data_4_lo.16b, {v_gft1_lo.16b}, v_data_4_lo.16b
	tbl	v_data_5_lo.16b, {v_gft1_lo.16b}, v_data_5_lo.16b
	tbl	v_data_6_lo.16b, {v_gft1_lo.16b}, v_data_6_lo.16b
	tbl	v_data_7_lo.16b, {v_gft1_lo.16b}, v_data_7_lo.16b

	tbl	v_data_0_hi.16b, {v_gft1_hi.16b}, v_data_0_hi.16b
	tbl	v_data_1_hi.16b, {v_gft1_hi.16b}, v_data_1_hi.16b
	tbl	v_data_2_hi.16b, {v_gft1_hi.16b}, v_data_2_hi.16b
	tbl	v_data_3_hi.16b, {v_gft1_hi.16b}, v_data_3_hi.16b
	tbl	v_data_4_hi.16b, {v_gft1_hi.16b}, v_data_4_hi.16b
	tbl	v_data_5_hi.16b, {v_gft1_hi.16b}, v_data_5_hi.16b
	tbl	v_data_6_hi.16b, {v_gft1_hi.16b}, v_data_6_hi.16b
	tbl	v_data_7_hi.16b, {v_gft1_hi.16b}, v_data_7_hi.16b

	eor	v_p0.16b, v_data_0_lo.16b, v_p0.16b
	eor	v_p0.16b, v_p0.16b, v_data_0_hi.16b
	eor	v_p1.16b, v_data_1_lo.16b, v_p1.16b
	eor	v_p1.16b, v_p1.16b, v_data_1_hi.16b
	eor	v_p2.16b, v_data_2_lo.16b, v_p2.16b
	eor	v_p2.16b, v_p2.16b, v_data_2_hi.16b
	eor	v_p3.16b, v_data_3_lo.16b, v_p3.16b
	eor	v_p3.16b, v_p3.16b, v_data_3_hi.16b
	eor	v_p4.16b, v_data_4_lo.16b, v_p4.16b
	eor	v_p4.16b, v_p4.16b, v_data_4_hi.16b
	eor	v_p5.16b, v_data_5_lo.16b, v_p5.16b
	eor	v_p5.16b, v_p5.16b, v_data_5_hi.16b
	eor	v_p6.16b, v_data_6_lo.16b, v_p6.16b
	eor	v_p6.16b, v_p6.16b, v_data_6_hi.16b
	eor	v_p7.16b, v_data_7_lo.16b, v_p7.16b
	eor	v_p7.16b, v_p7.16b, v_data_7_hi.16b

	cmp	x_vec_i, x_vec
	blt	.Lloop128_vects

.Lloop128_vects_end:
	add	x_ptr, x_dest1, x_pos
	stp	q_p0, q_p1, [x_ptr], #32
	stp	q_p2, q_p3, [x_ptr], #32
	stp	q_p4, q_p5, [x_ptr], #32
	stp	q_p6, q_p7, [x_ptr]

	add	x_pos, x_pos, #128
	cmp	x_pos, x_len
	ble	.Lloop128

.Lloop128_end:
	/* restore d8 ~ d15 */
	ldp	d8,  d9,  [sp]
	ldp	d10, d11, [sp, #16]
	ldp	d12, d13, [sp, #32]
	ldp	d14, d15, [sp, #48]
	add	sp, sp, #64

	add	x_len, x_len, #128
	cmp	x_pos, x_len
	beq	.return_pass

.Lloop16_init:
	sub	x_len, x_len, #16
	cmp	x_pos, x_len
	bgt	.lessthan16_init

.Lloop16:
	movi	v_p.16b, #0
	mov	x_tbl1, x_tbl
	mov	x_vec_i, #0

.Lloop16_vects:
	ldr	x_ptr, [x_src, x_vec_i]
	ldr	q_data, [x_ptr, x_pos]
	add	x_vec_i, x_vec_i, #8

	ldp	q_gft1_lo, q_gft1_hi, [x_tbl1], #32

	and	v_data_lo.16b, v_data.16b, v_mask0f.16b
	ushr	v_data_hi.16b, v_data.16b, #4

	tbl	v_data_lo.16b, {v_gft1_lo.16b}, v_data_lo.16b
	tbl	v_data_hi.16b, {v_gft1_hi.16b}, v_data_hi.16b
	eor	v_p.16b, v_data_lo.16b, v_p.16b
	eor	v_p.16b, v_p.16b, v_data_hi.16b

	cmp	x_vec_i, x_vec
	blt	.Lloop16_vects

.Lloop16_vects_end:
	str	q_p, [x_dest1, x_pos]
	add	x_pos, x_pos, #16
	cmp	x_pos, x_len
	ble	.Lloop16

.Lloop16_end:
	sub	x_tmp, x_pos, x_len
	cmp	x_tmp, #16
	beq	.return_pass

.lessthan16_init:
	mov	x_pos, x_len
	b	.Lloop16

.return_pass:
	mov	w_ret, #0
	ret

.return_fail:
	mov	w_ret, #1
	ret

cdecl(gf_vect_syndrome_neon):
	/* less than 16 bytes, return_fail */
	cmp	x_len, #16
	blt	.return_fail

	movi	v_mask0f.16b, #0x0f
	mov	x_pos, x5

	lsl	x_vec, x_vec, #3

.Mloop128_init:
	/* less than 128 bytes, goto Lloop16_init */
	cmp	x_len, #128
	blt	.Mloop16_init

	/* save d8 ~ d15 to stack */
	sub	sp, sp, #64
	stp	d8, d9, [sp]
	stp	d10, d11, [sp, #16]
	stp	d12, d13, [sp, #32]
	stp	d14, d15, [sp, #48]

	sub	x_len, x_len, #128

.Mloop128:
	movi	v_p0.16b, #0
	movi	v_p1.16b, #0
	movi	v_p2.16b, #0
	movi	v_p3.16b, #0
	movi	v_p4.16b, #0
	movi	v_p5.16b, #0
	movi	v_p6.16b, #0
	movi	v_p7.16b, #0

	mov	x_tbl1, x_tbl
	mov	x_vec_i, #0

.Mloop128_vects:
	ldr	x_ptr, [x_src, x_vec_i]
	add	x_vec_i, x_vec_i, #8
	add	x_ptr, x_ptr, x_pos

	ldp	q_gft1_lo, q_gft1_hi, [x_tbl1], #32

	ldp	q_data_0, q_data_1, [x_ptr], #32
	ldp	q_data_2, q_data_3, [x_ptr], #32
	ldp	q_data_4, q_data_5, [x_ptr], #32
	ldp	q_data_6, q_data_7, [x_ptr]

	prfm	pldl1keep, [x_tbl1]
	prfm	pldl1strm, [x_ptr]

	and	v_data_0_lo.16b, v_data_0.16b, v_mask0f.16b
	and	v_data_1_lo.16b, v_data_1.16b, v_mask0f.16b
	and	v_data_2_lo.16b, v_data_2.16b, v_mask0f.16b
	and	v_data_3_lo.16b, v_data_3.16b, v_mask0f.16b
	and	v_data_4_lo.16b, v_data_4.16b, v_mask0f.16b
	and	v_data_5_lo.16b, v_data_5.16b, v_mask0f.16b
	and	v_data_6_lo.16b, v_data_6.16b, v_mask0f.16b
	and	v_data_7_lo.16b, v_data_7.16b, v_mask0f.16b

	ushr	v_data_0_hi.16b, v_data_0.16b, #4
	ushr	v_data_1_hi.16b, v_data_1.16b, #4
	ushr	v_data_2_hi.16b, v_data_2.16b, #4
	ushr	v_data_3_hi.16b, v_data_3.16b, #4
	ushr	v_data_4_hi.16b, v_data_4.16b, #4
	ushr	v_data_5_hi.16b, v_data_5.16b, #4
	ushr	v_data_6_hi.16b, v_data_6.16b, #4
	ushr	v_data_7_hi.16b, v_data_7.16b, #4

	tbl	v_data_0_lo.16b, {v_gft1_lo.16b}, v_data_0_lo.16b
	tbl	v_data_1_lo.16b, {v_gft1_lo.16b}, v_data_1_lo.16b
	tbl	v_data_2_lo.16b, {v_gft1_lo.16b}, v_data_2_lo.16b
	tbl	v_data_3_lo.16b, {v_gft1_lo.16b}, v_data_3_lo.16b
	tbl	v_data_4_lo.16b, {v_gft1_lo.16b}, v_data_4_lo.16b
	tbl	v_data_5_lo.16b, {v_gft1_lo.16b}, v_data_5_lo.16b
	tbl	v_data_6_lo.16b, {v_gft1_lo.16b}, v_data_6_lo.16b
	tbl	v_data_7_lo.16b, {v_gft1_lo.16b}, v_data_7_lo.16b

	tbl	v_data_0_hi.16b, {v_gft1_hi.16b}, v_data_0_hi.16b
	tbl	v_data_1_hi.16b, {v_gft1_hi.16b}, v_data_1_hi.16b
	tbl	v_data_2_hi.16b, {v_gft1_hi.16b}, v_data_2_hi.16b
	tbl	v_data_3_hi.16b, {v_gft1_hi.16b}, v_data_3_hi.16b
	tbl	v_data_4_hi.16b, {v_gft1_hi.16b}, v_data_4_hi.16b
	tbl	v_data_5_hi.16b, {v_gft1_hi.16b}, v_data_5_hi.16b
	tbl	v_data_6_hi.16b, {v_gft1_hi.16b}, v_data_6_hi.16b
	tbl	v_data_7_hi.16b, {v_gft1_hi.16b}, v_data_7_hi.16b

	eor	v_p0.16b, v_data_0_lo.16b, v_p0.16b
	eor	v_p0.16b, v_p0.16b, v_data_0_hi.16b
	eor	v_p1.16b, v_data_1_lo.16b, v_p1.16b
	eor	v_p1.16b, v_p1.16b, v_data_1_hi.16b
	eor	v_p2.16b, v_data_2_lo.16b, v_p2.16b
	eor	v_p2.16b, v_p2.16b, v_data_2_hi.16b
	eor	v_p3.16b, v_data_3_lo.16b, v_p3.16b
	eor	v_p3.16b, v_p3.16b, v_data_3_hi.16b
	eor	v_p4.16b, v_data_4_lo.16b, v_p4.16b
	eor	v_p4.16b, v_p4.16b, v_data_4_hi.16b
	eor	v_p5.16b, v_data_5_lo.16b, v_p5.16b
	eor	v_p5.16b, v_p5.16b, v_data_5_hi.16b
	eor	v_p6.16b, v_data_6_lo.16b, v_p6.16b
	eor	v_p6.16b, v_p6.16b, v_data_6_hi.16b
	eor	v_p7.16b, v_data_7_lo.16b, v_p7.16b
	eor	v_p7.16b, v_p7.16b, v_data_7_hi.16b

	cmp	x_vec_i, x_vec
	blt	.Mloop128_vects

.Mloop128_vects_end:
	// Or the resuls together
	orr   v_tmp1.16b, v_p0.16b, v_p1.16b
	orr   v_tmp1.16b, v_tmp1.16b, v_p2.16b
	orr   v_tmp1.16b, v_tmp1.16b, v_p3.16b
	orr   v_tmp1.16b, v_tmp1.16b, v_p4.16b
	orr   v_tmp1.16b, v_tmp1.16b, v_p5.16b
	orr   v_tmp1.16b, v_tmp1.16b, v_p6.16b
	orr   v_tmp1.16b, v_tmp1.16b, v_p7.16b
    umaxv s_tmp1, v_tmp1.4S               // Find max across V_tmp1 (V1) lanes
    fmov  w_tmp, s_tmp1                   // Move result to w_tmp
    cmp   w_tmp, #0                       // Compare W_tmp with 0 (Z=1 if all lanes are 0)
    beq .MSkip128                         // Branch to mSkip64 if W8 == 0 (all lanes 0)

	mov	x_ptr, x_dest1
	stp	q_p0, q_p1, [x_ptr], #32
	stp	q_p2, q_p3, [x_ptr], #32
	stp	q_p4, q_p5, [x_ptr], #32
	stp	q_p6, q_p7, [x_ptr]
	b .Mloop128_end

.MSkip128:

	add	x_pos, x_pos, #128
	cmp	x_pos, x_len
	ble	.Mloop128

.Mloop128_end:
	/* restore d8 ~ d15 */
	ldp	d8,  d9,  [sp]
	ldp	d10, d11, [sp, #16]
	ldp	d12, d13, [sp, #32]
	ldp	d14, d15, [sp, #48]
	add	sp, sp, #64

	add	x_len, x_len, #128
	cmp	x_pos, x_len
	beq	.Mreturn_pass

.Mloop16_init:
	sub	x_len, x_len, #16
	cmp	x_pos, x_len
	bgt	.lessthan16_init

.Mloop16:
	movi	v_p.16b, #0
	mov	x_tbl1, x_tbl
	mov	x_vec_i, #0

.Mloop16_vects:
	ldr	x_ptr, [x_src, x_vec_i]
	ldr	q_data, [x_ptr, x_pos]
	add	x_vec_i, x_vec_i, #8

	ldp	q_gft1_lo, q_gft1_hi, [x_tbl1], #32

	and	v_data_lo.16b, v_data.16b, v_mask0f.16b
	ushr	v_data_hi.16b, v_data.16b, #4

	tbl	v_data_lo.16b, {v_gft1_lo.16b}, v_data_lo.16b
	tbl	v_data_hi.16b, {v_gft1_hi.16b}, v_data_hi.16b
	eor	v_p.16b, v_data_lo.16b, v_p.16b
	eor	v_p.16b, v_p.16b, v_data_hi.16b

	cmp	x_vec_i, x_vec
	blt	.Mloop16_vects

.Mloop16_vects_end:
	// Check for zero
    umaxv s_tmp1, v_tmp1.4S               // Find max across V_tmp1 (V1) lanes
    fmov  w_tmp, s_tmp1                   // Move result to w_tmp
    cmp   w_tmp, #0                       // Compare W_tmp with 0 (Z=1 if all lanes are 0)
    beq .mSkip16                       // Branch to mSkip64 if W8 == 0 (all lanes 0.0)

	str	q_p, [x_dest1]
	b   .Mreturn_fail

.mSkip16:
	add	x_pos, x_pos, #16
	cmp	x_pos, x_len
	ble	.Mloop16

.Mloop16_end:
	sub	x_tmp, x_pos, x_len
	cmp	x_tmp, #16
	beq	.Mreturn_pass

.Mlessthan16_init:
	mov	x_pos, x_len
	b	.Mloop16

.Mreturn_pass:
	mov	x0, x_pos
	ret

.Mreturn_fail:
	mov	x0, x_pos
	ret
