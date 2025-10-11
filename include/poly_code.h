#ifndef _POLY_CODE_H
#define _POLY_CODE_H

// Field size
#define PC_FIELD_SIZE 255

// Maximum parity symbols
#define PC_MAX_PAR 32

// Maximum lookup table size for parallel multiplier
#define PC_MAX_TAB 32

// Stride size, 64 bytes for avx512
#define PC_STRIDE 64

// Generator polynomial for 0x11d
#define PC_GEN_x11d 2

// Generator polynomial for 0x11b
#define PC_GEN_x11b 3

// Base value for number of test loops
#define PC_TEST_LOOPS 200

#endif
