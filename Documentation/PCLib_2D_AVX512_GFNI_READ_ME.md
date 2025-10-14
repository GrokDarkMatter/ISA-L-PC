# API Documentation for `PCLib_2D_AVX512_GFNI`

# PCLib_2D_AVX512_GFNI_for_doc.c Documentation

## Overview

This module implements a 2D error correction coding system using Reed-Solomon codes with optimizations for AVX512 hardware. It provides encoding and decoding functionalities leveraging advanced mathematical operations to handle errors in data transmission. This code specifically uses parallel operations to improve performance, particularly for large data sets.

### Licensing and Attribution

The code is copyrighted by Michael H. Anderson and may be subject to patents. It is intended for non-commercial evaluation only, and any commercial use requires a separate license.

## Constants and Global Variables

- **PC_TABLE_SIZE**:(256) Constant defining the size of various tables used (such as the power table).
- **PC_FIELD_SIZE**:(255) Size of the finite field used in the coding algorithm.
- **PC_L1PAR**:(4) Number of parity symbols for Level 1 coding.
- **PC_MAX_PAR**:(32) Maximum number of parity symbols supported.
  
- **Global Arrays**:
  - `unsigned char pc_ptab_2d[PC_TABLE_SIZE];`: Power table.
  - `unsigned char pc_ltab_2d[PC_TABLE_SIZE];`: Logarithm table.
  - `unsigned char pc_itab_2d[PC_TABLE_SIZE];`: Inverse table.
  - `__m512i EncMat[PC_FIELD_SIZE][PC_L1PAR];`: Encoding matrix for Level 1.
  - `__m512i Vand1b[PC_FIELD_SIZE][PC_L1PAR];`: Vandermonde matrix for decoding Level 1.
  - `unsigned char NumErrs;`: Number of detected errors on L1 reported to L2.
  - `unsigned char ErrLoc[PC_MAX_PAR];`: Locations of detected errors in L1.

## Macros

### L1Enc

```c
#define L1Enc(vec, p, pvec)
```
Defines the Level 1 encoding operation for a vector of bytes. It performs encoding using matrix-vector multiplication in GF(2^8).

### L1Dec

```c
#define L1Dec(vec, p, syn)
```
Defines the Level 1 decoding operation, which checks the syndromes based on the input vector.

## Functions

### PC_SingleEncoding

```c
int PC_SingleEncoding(unsigned char **data, int len, int symbols);
```
Encodes a block of data using single-level encoding (sequential symbols in memory). It processes symbols and writes the encoded output back to the original data.

**Parameters**:
- `data`: array containing the input symbols.
- `len`: Total number of codewords.
- `symbols`: Number of symbols to encode.

**Returns**: `0` on success.

---

### PC_SingleDecoding

```c
int PC_SingleDecoding(unsigned char **data, int len, int symbols, unsigned char *syn);
```
Decodes a block of data and checks for errors by evaluating syndromes.

**Parameters**:
- `data`: 2D array containing the encoded symbols.
- `len`: Total number of codewords.
- `symbols`: Number of symbols to decode.
- `syn`: Array to store the calculated syndromes.

**Returns**: Returns `0` if successful, `1` if errors are detected.

---

### Encoding/Decoding Functions (`PC_SingleEncoding_u`, `PC_SingleDecoding_u`)

These functions provide alternative implementations of encoding and decoding with different internal logic for handling parity and syndromes.

### Utility Functions

Several utility functions perform tasks such as:
- **pc_mul_2d**: Multiplies two elements in the Galois Field GF(2^8).
- **pc_bpow_2d**: Builds a power table for fast exponentiation.
- **pc_blog_2d**: Builds a logarithm table for fast logarithmic access.
- **pc_binv_2d**: Calculates multiplicative inverses.
- **Leading several functions to generate matrices** (`pc_gen_rsr_matrix_2d`, `pc_bmat_2d`, `pc_bvan_2d`): Used to form matrices needed for encoding and decoding.

---

### Error Correction and Syndrome Processing Functions

The module contains multiple functions that handle single and multiple error corrections. They include:

- **pc_verify_single_error_2d**: Checks for a single error and attempts to correct it.
- **pc_verify_multiple_errors_2d**: Identifies and corrects multiple errors using various strategies based on the syndromes generated during transmission.

### Parallel Processing Functions

Functions such as `gf_2vect_pss_avx512_gfni_2d` and `gf_3vect_pss_avx512_gfni_2d` illustrate parallel processing capabilities of the module using AVX512 instructions, optimizing the operations for multi-symbol processing.

---

## Conclusion

The `PCLib_2D_AVX512_GFNI_for_doc.c` code is designed for implementing and optimizing Reed-Solomon codes in a 2D setting for efficient data error correction, especially tailored for systems utilizing modern AVX512 architectures. It combines numerical methods with parallel processing to enhance ease of use and speed of operations in applications involving error detection and correction in data transmission settings. 

For more information regarding the licensing, refer to the copyright section included in the code. 

--- 
