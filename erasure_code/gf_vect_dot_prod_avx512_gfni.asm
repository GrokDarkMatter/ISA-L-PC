;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;Copyright (c) 2023 Intel Corporation.
;Copyright (c) 2025 Michael H. Anderson. All rights reserved.
;
;This software includes contributions protected by
;U.S. Patents 11,848,686 and 12,341,532.
;
;Redistribution and use in source and binary forms, with or without
;modification, are permitted for non-commercial evaluation purposes
;only, provided that the following conditions are met:
;
;Redistributions of source code must retain the above copyright notices,
;patent notices, this list of conditions, and the following disclaimer.
;
;Redistributions in binary form must reproduce the above copyright notices,
;patent notices, this list of conditions, and the following disclaimer in the
;documentation and/or other materials provided with the distribution.
;
;Neither the name of Intel Corporation, nor Michael H. Anderson, nor the names
;of their contributors may be used to endorse or promote products derived from
;this software without specific prior written permission.
;
;Commercial deployment or use of this software requires a separate license
;from the copyright holders and patent owners.
;
;In other words, this code is provided solely for the purposes of
;evaluation and is not licensed or intended to be licensed or used as part of
;or in connection with any commercial or non-commercial use other than evaluation
;of the potential for a license from Michael H. Anderson. Neither Michael H. Anderson
;nor any affiliated person grants any express or implied rights under any patents,
;copyrights, trademarks, or trade secret information. No content may be copied,
;stored, or utilized in any way without express written permission.
;
;THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
;"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
;THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
;ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
;FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
;(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES,
;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
;AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
;(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
;
;SPDX-License-Identifier: LicenseRef-Intel-Anderson-BSD-3-Clause-With-Restrictions
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;;; gf_vect_dot_prod_avx512_gfni(len, vec, *g_tbls, **buffs, *dest);
;;;

%include "reg_sizes.asm"
%include "gf_vect_gfni.inc"

%ifidn __OUTPUT_FORMAT__, elf64
 %define arg0  rdi
 %define arg1  rsi
 %define arg2  rdx
 %define arg3  rcx
 %define arg4  r8
 %define arg5  r9

 %define tmp   r11
 %define tmp2  r10

 %define func(x) x: endbranch
 %define FUNC_SAVE
 %define FUNC_RESTORE
%endif

%ifidn __OUTPUT_FORMAT__, win64
 %define arg0   rcx
 %define arg1   rdx
 %define arg2   r8
 %define arg3   r9

 %define arg4   r12 		; must be saved, loaded and restored
 %define arg5   r13 		; must be saved and restored
 %define tmp    r11
 %define tmp2   r10
 %define stack_size  0*16 + 3*8		; must be an odd multiple of 8
 %define arg(x)      [rsp + stack_size + 8 + 8*x]

 %define func(x) proc_frame x
 %macro FUNC_SAVE 0
	alloc_stack	stack_size
	mov	[rsp + 0*8], r12
	mov	[rsp + 1*8], r13
	end_prolog
	mov	arg4, arg(4)
	mov arg5, arg(5)
 %endmacro

 %macro FUNC_RESTORE 0
	mov	r12,  [rsp + 0*8]
	mov	r13,  [rsp + 1*8]
	add	rsp, stack_size
 %endmacro
%endif


%define len    arg0
%define vec    arg1
%define mul_array arg2
%define src    arg3
%define dest1  arg4
%define ptr    arg5
%define vec_i  tmp2
%define pos    rax


%ifndef EC_ALIGNED_ADDR
;;; Use Un-aligned load/store
 %define XLDR vmovdqu8
 %define XSTR vmovdqu8
%else
;;; Use Non-temporal load/stor
 %ifdef NO_NT_LDST
  %define XLDR vmovdqa64
  %define XSTR vmovdqa64
 %else
  %define XLDR vmovntdqa
  %define XSTR vmovntdq
 %endif
%endif

%define xgft1  zmm2

%define x0     zmm0
%define xp1    zmm1

default rel
[bits 64]
section .text

;;
;; Encodes 64 bytes of all "k" sources into 64 bytes (single parity disk)
;;
%macro ENCODE_64B 0-1
%define %%KMASK %1

	vpxorq	xp1, xp1, xp1
	mov	tmp, mul_array
	xor	vec_i, vec_i

%%next_vect:
	mov	ptr, [src + vec_i]
%if %0 == 1
	vmovdqu8 x0{%%KMASK}, [ptr + pos]	;Get next source vector (less than 64 bytes)
%else
	XLDR	x0, [ptr + pos]		;Get next source vector (64 bytes)
%endif
	add	vec_i, 8

        vbroadcastf32x2 xgft1, [tmp]
	add	tmp, 8

        GF_MUL_XOR EVEX, x0, xgft1, xgft1, xp1

	cmp	vec_i, vec
	jl	%%next_vect

%if %0 == 1
	vmovdqu8 [dest1 + pos]{%%KMASK}, xp1
%else
	XSTR	[dest1 + pos], xp1
%endif
%endmacro

;;
;; Decodes 64 bytes of all "k" sources into 64 bytes (single parity disk)
;;
%macro DECODE_64B 0-1
%define %%KMASK %1

	vpxorq	xp1, xp1, xp1
	mov	tmp, mul_array
	xor	vec_i, vec_i

%%next_vect:
	mov	ptr, [src + vec_i]
%if %0 == 1
	vmovdqu8 x0{%%KMASK}, [ptr + pos]	;Get next source vector (less than 64 bytes)
%else
	XLDR	x0, [ptr + pos]		;Get next source vector (64 bytes)
%endif
	add	vec_i, 8

        vbroadcastf32x2 xgft1, [tmp]
	add	tmp, 8

        GF_MUL_XOR EVEX, x0, xgft1, xgft1, xp1

	cmp	vec_i, vec
	jl	%%next_vect

	; Check to see if parity is zero
	vptestmq    k2, xp1, xp1
	ktestb		k2, k2
	jz			%%sndOK

	; Save non-zero parity and exit
%if %0 == 1
	vmovdqu8 [dest1]{%%KMASK}, xp1
%else
	XSTR	[dest1], xp1
%endif
	jmp		.exit
%%sndOK:
%endmacro

align 16
mk_global gf_vect_dot_prod_avx512_gfni, function
func(gf_vect_dot_prod_avx512_gfni)
	FUNC_SAVE
	xor	pos, pos
	shl	vec, 3		;vec *= 8. Make vec_i count by 8

	cmp	len, 64
        jl      .len_lt_64

.loop64:

        ENCODE_64B

	add	pos, 64			;Loop on 64 bytes at a time
        sub     len, 64
	cmp	len, 64
	jge	.loop64

.len_lt_64:
        cmp     len, 0
        jle     .exit

        xor     tmp, tmp
        bts     tmp, len
        dec     tmp
        kmovq   k1, tmp

        ENCODE_64B k1

.exit:
        vzeroupper

	FUNC_RESTORE
	ret

align 16
mk_global gf_vect_syndrome_avx512_gfni, function
func(gf_vect_syndrome_avx512_gfni)
	FUNC_SAVE
	;xor	pos, pos
	mov		pos, arg5
	shl	vec, 3		;vec *= 8. Make vec_i count by 8

	cmp	len, 64
        jl      .len_lt_64

.loop64:

        DECODE_64B

	add	pos, 64			;Loop on 64 bytes at a time
        sub     len, 64
	cmp	len, 64
	jge	.loop64

.len_lt_64:
        cmp     len, 0
        jle     .exit

        xor     tmp, tmp
        bts     tmp, len
        dec     tmp
        kmovq   k1, tmp

        DECODE_64B k1

.exit:
        vzeroupper

	FUNC_RESTORE
	ret

endproc_frame
