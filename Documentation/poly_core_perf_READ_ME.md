# API Documentation for `poly_core_perf.c`

## Overview
The `poly_core_perf.c` file contains an implementation of performance benchmarking for polynomial coding and erasure coding techniques. This file allows the user to evaluate the performance of encoding and decoding operations across multiple CPU cores, utilizing different testing modes and performance optimizations based on processor architecture.

This documentation outlines the key functionalities, data structures, macros, and usage of the functions provided in this file.

## Licensing Information
```plaintext
Copyright (c) 2011-2024 Intel Corporation.
Copyright (c) 2025 Michael H. Anderson. All rights reserved.
...
```
This software is licensed for non-commercial evaluation purposes only. Redistribution and use must comply with the specified conditions in the license.

## Include Files
- `ec_base.h`: Base definitions for error correction functionalities.
- `erasure_code.h`: Declarations for erasure coding functions.
- `poly_code.h`: Declarations related to polynomial coding.
- `test.h`: Test-related utilities.
- `<stdio.h>`: Standard Input/Output definitions.
- `<stdlib.h>`: Standard library definitions for memory allocation and process control.
- `<string.h>`: String manipulation definitions.

## Macros
### Time Management
- `ECCTIME`: Struct to manage time values, defined as `struct timeval`.
- `ECCGETTIME(X)`: Macro to retrieve the current time.
- `ECCELAPSED(X, Y, Z)`: Macro to compute elapsed time between two time points.

### Thread Management
- `ECCENDTHREAD`: Macro to end a thread.
- `ECCTHREAD`: Defines a thread handle type.
- `ECCTHREADSTART(T, F, A)`: Start a new thread.
- `ECCTHREADWAIT(T)`: Wait for a thread's completion.

## Data Structures
### `struct PCBenchStruct`
This structure is used to encapsulate the parameters and buffers required for benchmarking:
```c
struct PCBenchStruct
{
    unsigned char **Data;   // Pointer to data buffers
    unsigned char **Syn;    // Pointer to parity buffers (Syndromes)
    unsigned char k;        // Number of source buffers
    unsigned char p;        // Number of parity buffers
    unsigned char *g_tbls;  // Generator tables
    unsigned char *plyTab;   // Polynomial table
    unsigned char *pwrTab;   // Power table
    unsigned char *plyTab2d; // 2D Polynomial table
    unsigned char *pwrTab2d; // 2D Power table
    int testNum;            // Test number
    int testReps;          // Number of repetitions for testing
};
```

## Functions
### 1. `void dump_u8xu8(unsigned char *s, int k, int m);`
- **Description**: Utility function to print a matrix of unsigned characters in a formatted manner.
- **Parameters**:
  - `s`: Pointer to the data to be printed.
  - `k`: Number of rows.
  - `m`: Number of columns.

### 2. `void BenchWorker(void *t);`
- **Description**: Function that serves as a worker thread for executing benchmark tests based on the parameters specified in `PCBenchStruct`.
- **Parameters**:
  - `t`: Pointer to a `PCBenchStruct` instance which contains the test configuration.

### 3. `void usage(const char *app_name);`
- **Description**: Displays usage information for the command line application.
- **Parameters**:
  - `app_name`: The name of the application to print in the usage message.

### 4. `int InitClone(struct PCBenchStruct *ps, unsigned char k, unsigned char p, int testNum, int testReps);`
- **Description**: Initializes and allocates the necessary data buffers and tables for a benchmark test.
- **Parameters**:
  - `ps`: Pointer to a `PCBenchStruct` instance to initialize.
  - `k`: Number of source buffers.
  - `p`: Number of parity buffers.
  - `testNum`: Identifier for the type of test to execute.
  - `testReps`: Number of repetitions for the test.
- **Returns**: 1 on success, 0 on failure.

### 5. `void FreeClone(struct PCBenchStruct *ps, unsigned char k, unsigned char p);`
- **Description**: Frees the memory allocated for the data buffers and other structures in a `PCBenchStruct`.
- **Parameters**:
  - `ps`: Pointer to the `PCBenchStruct` containing the buffers to free.

### 6. `int main(int argc, char *argv[]);`
- **Description**: Entry point of the application. Parses command line arguments, initializes benchmarks, runs performance tests, and outputs results.
- **Parameters**:
  - `argc`: Argument count.
  - `argv`: Array of argument strings.
- **Returns**: Status code indicating success (0) or failure (non-zero).

## Testing Modes
- **Cached Test (Warm)**: Configurations suitable for executing tests with small datasets that fit well in CPU cache.
- **Uncached Test (Cold)**: Configurations designed for pulling data from larger memory space, measuring performance under non-cached conditions.
- **Custom Test**: A user-defined mode that may be configured through preprocessor directives.

## Error Handling
Several error checks are present throughout the functions, particularly in memory allocation (e.g., `posix_memalign`) which returns errors if memory cannot be allocated. The code utilizes `printf` to provide feedback regarding errors and the status of operations.

## Example Usage
```bash
./poly_core_perf -k 128 -p 32 -c 4
```
This command runs the performance benchmark with 128 source buffers, 32 parity buffers, and utilizes 4 CPU cores.

## Conclusion
This documentation provides a structured overview of the functionality found in `poly_core_perf.c`, focusing on benchmarking polynomial coding and erasure coding performance across various architectures. For any modifications, be mindful of the licensing implications and ensure compliance with stated permissions.

---
*Documentation generated by* **[AutoCodeDocs.ai](https://autocodedocs.ai)**