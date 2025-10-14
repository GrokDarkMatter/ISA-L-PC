## API Documentation for PCLib_AVX512_GFNI Library

### Overview 
The `PCLib_AVX512_GFNI` library provides functions for error correction and encoding/decoding using finite field arithmetic, specifically designed to leverage Intel's AVX-512 and GFNI (Galois Field New Instructions) for performance optimization. The library is capable of handling operations required for various coding schemes, including Reed-Solomon codes and implementations of linear feedback shift registers (LFSR).

### License
This software is licensed under the Intel-Anderson-BSD-3-Clause-With-Restrictions. Redistribution and use are permitted for non-commercial evaluation purposes only with specific conditions outlined in the license header provided in the source file.

### Functions

#### Finite Field Arithmetic Functions 

1. **`gf_mul(unsigned char a, unsigned char b)`**
   - **Description**: Multiplies two elements in a Galois Field GF(256).
   - **Parameters**: 
     - `a`: First element.
     - `b`: Second element.
   - **Returns**: The product of `a` and `b` in GF(256).

2. **`gf_inv(unsigned char a)`**
   - **Description**: Computes the multiplicative inverse of an element in GF(256).
   - **Parameters**: 
     - `a`: Element whose inverse is to be calculated.
   - **Returns**: The multiplicative inverse of `a`.

3. **`gf_div_AVX512_GFNI(unsigned char a, unsigned char b)`**
   - **Description**: Divides two elements in GF(256) using multiplication and inverse.
   - **Parameters**: 
     - `a`: Dividend.
     - `b`: Divisor.
   - **Returns**: The result of `a / b`.

4. **`pc_pow_AVX512_GFNI(unsigned char base, unsigned char Power)`**
   - **Description**: Computes the power of a base element in GF(256).
   - **Parameters**: 
     - `base`: The base element.
     - `Power`: The exponent.
   - **Returns**: The result of `base^Power`.

#### Error Correction Functions

5. **`pc_verify_single_error_AVX512_GFNI(unsigned char *S, unsigned char **data, int k, int p, int newPos, int offSet)`**
   - **Description**: Attempts to correct a single error in the received syndromes.
   - **Parameters**: 
     - `S`: Array of syndromes.
     - `data`: Received data array.
     - `k`: Number of symbols.
     - `p`: Number of syndromes.
     - `newPos`: New position in the data to check.
     - `offSet`: Offset in the data array.
   - **Returns**: 1 if correction is successful, 0 otherwise.

6. **`pc_verify_multiple_errors_AVX512_GFNI(unsigned char *S, unsigned char **data, int mSize, int k, int p, int newPos, int offSet, unsigned char *keyEq)`**
   - **Description**: Attempts to correct multiple errors in the received data based on syndromes.
   - **Parameters**: 
     - `S`: Array of syndromes.
     - `data`: Received data array.
     - `mSize`: Size of the error locator polynomial.
     - `k`: Number of symbols.
     - `p`: Number of syndromes.
     - `newPos`: New position in the data to check.
     - `offSet`: Offset in the data array.
     - `keyEq`: Key equation array for error correction.
   - **Returns**: 1 if correction is successful, 0 otherwise.

7. **`pc_compute_error_values_AVX512_GFNI(int mSize, unsigned char *S, unsigned char *roots, unsigned char *errVal)`**
   - **Description**: Computes the error values using the Vandermonde matrix based on syndromes.
   - **Parameters**: 
     - `mSize`: Size of the matrix.
     - `S`: Array of syndromes.
     - `roots`: Roots of the error locator polynomial.
     - `errVal`: Array to hold computed error values.
   - **Returns**: 1 if successful, 0 if failure.

8. **`pc_correct_AVX512_GFNI(int newPos, int k, int p, unsigned char **data, unsigned char **coding, int vLen)`**
   - **Description**: Corrects errors in the data based on syndromes and returns the position of syndromes found.
   - **Parameters**: 
     - `newPos`: New position for checking errors.
     - `k`: Number of symbols.
     - `p`: Number of syndromes.
     - `data`: Reference to the data array.
     - `coding`: Coding data array.
     - `vLen`: Length of the vectors.
   - **Returns**: Position with corrected errors.

#### Matrix Operations

9. **`gf_invert_matrix_AVX512_GFNI(unsigned char *in_mat, unsigned char *out_mat, const int n)`**
   - **Description**: Computes the inverse of a matrix over GF(256).
   - **Parameters**: 
     - `in_mat`: Input matrix to be inverted.
     - `out_mat`: Output matrix to hold the result of the inversion.
     - `n`: Dimension of the matrix (n x n).
   - **Returns**: 0 if successful, -1 if the matrix is singular.

10. **`berlekamp_massey_AVX512_GFNI(unsigned char *syndromes, int length, unsigned char *lambda)`**
    - **Description**: Computes the error locator polynomial using the Berlekamp-Massey algorithm.
    - **Parameters**: 
      - `syndromes`: Array of syndromes.
      - `length`: Length of the syndromes.
      - `lambda`: Array to hold the coefficients of the error locator polynomial.
    - **Returns**: The degree of the error locator polynomial.

#### Encoding and Decoding Functions

11. **`pc_encode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data, unsigned char **coding)`**
    - **Description**: Encodes data using the specified generator tables.
    - **Parameters**:
      - `len`: Length of the data.
      - `k`: Number of symbols.
      - `rows`: Number of rows (codewords).
      - `g_tbls`: Generator tables.
      - `data`: Data to be encoded.
      - `coding`: Output encoded data.
    
12. **`pc_decode_data_avx512_gfni(int len, int k, int rows, unsigned char *g_tbls, unsigned char **data, unsigned char **coding, int retries)`**
    - **Description**: Decodes data and attempts to correct errors.
    - **Parameters**:
      - `len`: Length of the data.
      - `k`: Number of symbols.
      - `rows`: Number of codewords.
      - `g_tbls`: Generator tables.
      - `data`: Data to decode.
      - `coding`: Coding data.
      - `retries`: Number of retry attempts for correction.
    - **Returns**: The total number of decoded items.

### Dependencies
- This library relies on Intel AVX-512 and GFNI for optimized vectorized mathematical computations.
- Requires compatible hardware for executing AVX-512 instructions.

### Usage Notes
- Ensure that you have a proper development environment configured to utilize Intel’s AVX-512 instructions.
- The library's primary use is for low-level error correction tasks, especially in environments requiring high-performance data encoding and decoding.

### Conclusion
The `PCLib_AVX512_GFNI` library enhances operations on error-correcting codes and is well-suited for applications in coding theory and data integrity checks, providing both efficacy and optimized performance through advanced SIMD instruction sets.

---
*Documentation generated by* **[AutoCodeDocs.ai](https://autocodedocs.ai)**