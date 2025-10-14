# API Documentation for `poly_code_perf.c`

This documentation provides a detailed overview of the API found in the `poly_code_perf.c` source file. The source includes erasure code performance testing functions, PAPI performance monitoring, and test functions to evaluate different encoding and decoding methods used in polynomial-based error correction.

## Table of Contents
1. [File Overview](#file-overview)
2. [Utility Functions](#utility-functions)
3. [Performance Measurement with PAPI](#performance-measurement-with-papi)
4. [Main Functional API](#main-functional-api)
5. [Error Injection and Verification](#error-injection-and-verification)
6. [Utility Functions for Random Generation and Buffer Comparison](#utility-functions-for-random-generation-and-buffer-comparison)
7. [Main Function](#main-function)

### File Overview

The file is licensed under a proprietary license with significant contributions from Intel Corporation and Michael H. Anderson. It includes methods for encoding and decoding data using polynomial codes, designed for high-performance applications. The implementation uses vectorized instructions for both AVX512 and ARM NEON architectures.

Key features include:
- Polynomial encoding/decoding using AVX512-GFNI.
- PAPI integration for performance metrics.
- Utility functions for error injection and correction verification.

### Utility Functions

#### `void dump_u8xu8(unsigned char *s, int k, int m)`

This function prints a 2D representation of byte arrays in a formatted output.

**Parameters:**
- `s`: Pointer to the array to be printed.
- `k`: Number of rows.
- `m`: Number of columns.

### Performance Measurement with PAPI

The performance measurement of encoding and decoding processes is facilitated through the PAPI (Performance Application Programming Interface). The following key methods are used for this purpose:

#### `int InitPAPI(void)`

Initializes the PAPI library and creates a new event set for performance counters.

**Returns:**  
- An integer representing the created event set or a negative number on failure.

#### `void handle_error(int code)`

Handles errors from PAPI operations by printing a message and terminating the program.

**Parameters:**
- `code`: The error code returned by PAPI functions.

### Main Functional API

#### `ec_decode_perf(int m, int k, u8 *a, u8 *g_tbls, u8 **buffs, u8 *src_in_err, u8 *src_err_list, int nerrs, u8 **temp_buffs, struct perf *start)`

Performs the decoding process with error correction capabilities.

**Parameters:**
- `m`: Total number of data and parity buffers.
- `k`: Number of source buffers.
- `a`: Pointer to the encoding matrix.
- `g_tbls`: Pointer to the generated tables.
- `buffs`: An array of data buffers.
- `src_in_err`: Buffer that indicates which sources are in error.
- `src_err_list`: List of errors present in the source data.
- `nerrs`: Number of errors to be corrected.
- `temp_buffs`: Temporary buffers used during the decoding process.
- `start`: Pointer to structure for performance measurement.

**Returns:**  
- An integer status code; typically, 0 if successful or a negative error code.

### Error Injection and Verification

#### `void inject_errors_in_place_2d(unsigned char **data, unsigned char *offsets, int d1Len, int num_errors, unsigned char *error_positions, unsigned char *original_values)`

Injects simulated errors into the provided data buffers.

**Parameters:**
- `data`: An array of data buffers.
- `offsets`: Offset values for errors.
- `d1Len`: The length of the first dimension.
- `num_errors`: The number of errors to be injected.
- `error_positions`: Array indicating where errors are injected.
- `original_values`: Stores original values of data to verify corrections later.

#### `int verify_correction_in_place_2d(unsigned char **data, unsigned char *offSets, int d1Len, int num_errors, unsigned char *error_positions, uint8_t *original_values)`

Verifies if the data correction after simulation was successful.

**Parameters:**
- `data`: Data buffers with potential errors.
- `offSets`: Array with data offsets.
- `d1Len`: Length of the array in errors.
- `num_errors`: Number of errors corrected.
- `error_positions`: Array of positions where errors were injected.
- `original_values`: Original values for comparison.

**Returns:**  
- 1 if verification succeeds; 0 otherwise.

### Utility Functions for Random Generation and Buffer Comparison

These functions provide utilities for generating random values and comparing buffers, ensuring functionality during testing.

#### `void make_norepeat_rand(int listSize, int fieldSize, unsigned char *list)`

Generates a list of unique random values within the specified range.

**Parameters:**
- `listSize`: Size of the generated list.
- `fieldSize`: Maximum value for random numbers.
- `list`: Pointer to the array where generated values are stored.

#### `int Compare2Buffers(unsigned char *buf1, unsigned char *buf2, int len)`

Compares two buffers for equality.

**Parameters:**
- `buf1`: First buffer to compare.
- `buf2`: Second buffer to compare.
- `len`: Length of the buffers.

**Returns:**  
- 0 if equal; non-zero if not equal, with additional details logged.

### Main Function

#### `int main(int argc, char *argv[])`

The entry point for the application which processes command-line arguments for the number of data (`-k`) and parity buffers (`-p`), allocates necessary resources, initializes the encoding matrices, and executes the encoding and decoding benchmarks.

**Parameters:**
- `argc`: Argument count.
- `argv`: Array of argument strings.

**Returns:**  
- 0 on success, negative error codes for various failure states.

### Conclusion

The `poly_code_perf.c` file provides an extensive implementation for testing and measuring the performance of polynomial coding methods. With integrated PAPI for performance insights, it serves as a strong foundation for further development in erasure coding and performance optimization in error correction algorithms. This documentation aims to make the intricate functionalities more accessible for developers aiming to utilize or extend the codebase.

---
*Documentation generated by* **[AutoCodeDocs.ai](https://autocodedocs.ai)**