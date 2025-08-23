Intel(R) Intelligent Storage Acceleration Library
=================================================

![Continuous Integration](https://github.com/intel/isa-l/actions/workflows/ci.yml/badge.svg)
[![Package on conda-forge](https://img.shields.io/conda/v/conda-forge/isa-l.svg)](https://anaconda.org/conda-forge/isa-l)
[![Coverity Status](https://scan.coverity.com/projects/29480/badge.svg)](https://scan.coverity.com/projects/intel-isa-l)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/intel/isa-l/badge)](https://securityscorecards.dev/viewer/?uri=github.com/intel/isa-l)

ISA-L is a collection of optimized low-level functions targeting storage
applications.  ISA-L includes:
* Erasure codes - Fast block Reed-Solomon type erasure codes for any
  encode/decode matrix in GF(2^8).
* CRC - Fast implementations of cyclic redundancy check.  Six different
  polynomials supported.
  - iscsi32, ieee32, t10dif, ecma64, iso64, jones64, rocksoft64.
* Raid - calculate and operate on XOR and P+Q parity found in common RAID
  implementations.
* Compression - Fast deflate-compatible data compression.
* De-compression - Fast inflate-compatible data compression.
* igzip - A command line application like gzip, accelerated with ISA-L.

Also see:
* [ISA-L for updates](https://github.com/intel/isa-l).
* For crypto functions see [isa-l_crypto on github](https://github.com/intel/isa-l_crypto).
* The [github wiki](https://github.com/intel/isa-l/wiki) including a list of
  [distros/ports](https://github.com/intel/isa-l/wiki/Ports--Repos) offering binary packages
  as well as a list of [language bindings](https://github.com/intel/isa-l/wiki/Language-Bindings).
* [Contributing](CONTRIBUTING.md).
* [Security Policy](SECURITY.md).
* Docs on [units](doc/functions.md), [tests](doc/test.md), or [build details](doc/build.md).

Building ISA-L
--------------

### Prerequisites

* Make: GNU 'make' or 'nmake' (Windows).
* Optional: Building with autotools requires autoconf/automake/libtool packages.
* Optional: Manual generation requires help2man package.

x86_64:
* Assembler: nasm. 2.14.01 minimum version required [support](doc/build.md)).
* Compiler: gcc, clang, icc or VC compiler.

aarch64:
* Assembler: gas v2.24 or later.
* Compiler: gcc v4.7 or later.

other:
* Compiler: Portable base functions are available that build with most C compilers.

### Autotools
To build and install the library with autotools it is usually sufficient to run:

    ./autogen.sh
    ./configure
    make
    sudo make install

### Makefile
To use a standard makefile run:

    make -f Makefile.unx

### Windows
On Windows use nmake to build dll and static lib:

    nmake -f Makefile.nmake

or see [details on setting up environment here](doc/build.md).

### Other make targets
Other targets include:
* `make check` : create and run tests
* `make tests` : create additional unit tests
* `make perfs` : create included performance tests
* `make ex`    : build examples
* `make other` : build other utilities such as compression file tests
* `make doc`   : build API manual

DLL Injection Attack
--------------------

### Problem

The Windows OS has an insecure predefined search order and set of defaults when trying to locate a resource. If the resource location is not specified by the software, an attacker need only place a malicious version in one of the locations Windows will search, and it will be loaded instead. Although this weakness can occur with any resource, it is especially common with DLL files.

### Solutions

Applications using libisal DLL library may need to apply one of the solutions to prevent from DLL injection attack.

Two solutions are available:
- Using a Fully Qualified Path is the most secure way to load a DLL
- Signature verification of the DLL

### Resources and Solution Details

- Security remarks section of LoadLibraryEx documentation by Microsoft: <https://docs.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa#security-remarks>
- Microsoft Dynamic Link Library Security article: <https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-security>
- Hijack Execution Flow: DLL Search Order Hijacking: <https://attack.mitre.org/techniques/T1574/001>
- Hijack Execution Flow: DLL Side-Loading: <https://attack.mitre.org/techniques/T1574/002>

POLYNOMIAL ENCODING
-------------------

### Updates for Polynomial Encoding

Changes (c) Copyright 2025 by Michael H. Anderson. All rights reserved. See LICENSE file for details.

This version of ISA-L has been adapted to implement an Accelerated Polynomial System and Method, as described in US Patents 11,848,686 and 12,341,532. See https://patents.justia.com/patent/11848686.

### Background

Patterson, Gibson, and Katz introduced RAID in 1988, defining levels 1–5 with RAID5 using single parity for one failure. In 1997, James S. Plank published “A Tutorial on Reed-Solomon Coding for Fault-Tolerance in RAID-like Systems,” aiming to explain how Reed-Solomon codes could be applied to RAID for multiple failure tolerance. Plank suggested using a Vandermonde matrix to encode data into check symbols,  computing syndromes as weighted sums of data symbols alone (e.g., for  check symbols P and Q, using rows of the Vandermonde matrix. However, Plank misunderstood the  role of the Vandermonde matrix—it’s not for encoding but for defining the parity-check equations (H matrix). In a correct Reed-Solomon code, a  codeword c = (d₁, d₂, ..., dₖ, c₁, c₂, ..., cₘ) (data + check symbols) must satisfy Hc = 0, meaning syndromes evaluate to zero at the polynomial roots (α⁰, α¹, ..., αᵐ⁻¹). Plank’s method produced check symbols from data alone, so the syndromes of the full codeword didn’t evaluate to zero, making error location impossible.

In 2003, Plank and Ying Ding published a correction, “Note: Correction to the 1997 Tutorial on Reed-Solomon Coding,” addressing errors in the original paper (e.g., incorrect matrix operations in GF(2^w)). However, even with the correction, Plank’s approach still used an inverted and normalized Vandermonde matrix for encoding and computed check symbols from data alone, failing to produce codewords that evaluate to zero at the polynomial roots. The syndromes of the full codeword (data + check) were non-zero, so the code remained ineffective for unknown error location.

Peter Anvin’s “The Mathematics of RAID-6” (2004–2011) built on Plank’s work, citing Plank's 1997 paper as a reference. Anvin adopted the same flawed assumption that the Vandermonde matrix is used for encoding, defining P as the parity of data symbols only (P = d₁ ⊕ d₂ ⊕  ... ⊕ dₖ) and Q as a weighted sum of data (Q = 1·d₁ ⊕ 2·d₂ ⊕ ... ⊕ k·dₖ). Anvin also misunderstood syndromes, thinking they were computed over data alone, not the entire codeword. This ordering (P then Q) meant P didn’t include Q, so the codeword didn’t satisfy the parity-check equations (Hc ≠ 0). The syndromes didn’t evaluate to zero, making unknown error correction impossible.

Greg Tucker’s GitHub issues (#10 (https://github.com/intel/isa-l/issues/10) on ISA-L highlight the practical fallout of Plank and Anvin’s misunderstandings. ISA-L followed the same flawed approach (Vandermonde for encoding, syndromes over data only), leading to:

Issue  #10: “corrupted fragment on decode.” Tucker noted that the Vandermonde matrix fails for m - k errors (“unrecoverable”), a direct result of improper Reed Solomon encoding. He tried to explain this as “normal and usual,” ("If you increase m and k for this first case you can find an unrecoverable code where errors <= m - k but this is an expected result if you are using Vandermonde matrix. This is fundamental to the math and not particular to ISA-L EC"). But actually, it is a fundamental flaw in his approach to the problem, not an "expected result" or "fundamental to the math" if you use Vandermonde to decode properly encoded Reed Solomon codewords.

Issues  #13, #26, #40 and #46 all spring from the same flawed approach: Mixed data/check symbols fail to decode because syndromes don’t include check symbols.

### Polynomial Codes provide both Erasure Coding and Unknown Error Decoding

Polynomial Codes are similar to Erasure Codes (as implemented by ISA-L), but more powerful. This version of ISA-L employs Polynomial Codes to both detect and correct unknown errors, and includes a source code example to identify both unknown error location and unknown error value. The example targets a single error, but standard Reed Solomon error correction can be applied to the Syndromes produced by this code such that the error correction power = p/2, or the number of parity drives divided by two. For example, a RAID6 Polynomial Code (2 parities) can correct one unknown error, and a 4 parity Polynomial Code can correct two unknown errors simultaneously.

### Polynomial Codes support full size codewords

Unlike the code generated by gf_gen_rs_matrix, the Polynomial Codes generated by this change scale to any codeword size of 255 or less. After installation, you can verify this with "make other" followed by erasure_code/gen_rs_matrix_limits, which has been modifed to use Polynomial matrices. For example:

```test
erasure_code/gen_rs_matrix_limits
Checking gen_poly_matrix for k <= 16 and m <= 32.
gen_poly_matrix creates erasure codes for:
   k =  1, m <= 32
   k =  2, m <= 32
   k =  3, m <= 32
   k =  4, m <= 32
   k =  5, m <= 32
   k =  6, m <= 32
   k =  7, m <= 32
   k =  8, m <= 32
   k =  9, m <= 32
   k = 10, m <= 32
   k = 11, m <= 32
   k = 12, m <= 32
   k = 13, m <= 32
   k = 14, m <= 32
   k = 15, m <= 32
   k = 16, m <= 32

   ```

### Examining the Polynomial matrix with erasure_code_perf
You can see a simple example of a Polynomial matrix by running "make perfs", and then executing erasure_code/erasure_code_perf. By default, you will see a Polynomial matrix of dimension 10 by 4. By adding the argument -p 2, you can see the two parity version of a Polynomial matrix. It is not the same as RAID6 as described in The Mathematics of RAID6 by Peter Anvin.

### Unknown Error Detection and Correction example using Polynomial Codes

Code is supplied in erasure_code_perf.c to inject an unknown error and then perform the error value and error location decoding. The default error value is one (can be changed with -pe val), and the default error location is symbol 1 (can be changed with -pp val) For example:

```text
erasure_code/erasure_code_perf
Testing with 10 data buffers and 4 parity buffers (num errors = 4, in [ 4 6 2 9 ])
erasure_code_perf: 14x2396736 4
Poly Matrix
   1   0   0   0   0   0   0   0   0   0
   0   1   0   0   0   0   0   0   0   0
   0   0   1   0   0   0   0   0   0   0
   0   0   0   1   0   0   0   0   0   0
   0   0   0   0   1   0   0   0   0   0
   0   0   0   0   0   1   0   0   0   0
   0   0   0   0   0   0   1   0   0   0
   0   0   0   0   0   0   0   1   0   0
   0   0   0   0   0   0   0   0   1   0
   0   0   0   0   0   0   0   0   0   1
  f6  e2  fe  77  da  a7  54  5c  63   f
  52  21  81  39  d5  d9  8c   7  57  36
  87  b3  f2  cb  bc  db  47  bf  d2  78
  22  71  8c  84  b2  a4  9e  e5  e7  40

erasure_code_encode_cold: runtime =    3170406 usecs, bandwidth 30970 MB in 3.1704 sec = 9768.66 MB/s
erasure_code_decode_cold: runtime =    3055730 usecs, bandwidth 24360 MB in 3.0557 sec = 7972.05 MB/s

Error Syndromes (should not be zero)
  8  0  0  0  0  0  0  0
  4  0  0  0  0  0  0  0
  2  0  0  0  0  0  0  0
  1  0  0  0  0  0  0  0

Error value = 1 Error location = 1
Error value specified (1) and position (1) decoded correctly
done all: Pass
```

### Extended Erasure Code integrity testing for Polynomial Codes

For more complete Erasure Code integrity testing, erasure_code/erasure_code_test.c has been modfied to use a Polynomial matrix instead of a Cauchy matrix. You can verify the Polynomial matrix for Erasure Coding by first executing make checks, and then executing erasure_code/erasure_code_test.

```test
erasure_code/erasure_code_test
erasure_code_test: 127x8192 done EC tests: Pass

### Typical erasure_code_perf Test Results

Testing with 12 data buffers and 8 parity buffers (num errors = 8, in [ 6 0 7 11 10 1 3 4 ])
erasure_code_perf: 20x1677696 8
erasure_code_encode_cold: k=12 p=8 runtime = 3066588 usecs, bandwidth 30232 MB in 3.0666 sec = 9858.54 MB/s
polynomial_code_ls_cold: k=12 p=8 runtime = 3024831 usecs, bandwidth 42579 MB in 3.0248 sec = 14076.79 MB/s
dot_prod_decode_cold: k=20 p=8 runtime = 3449346 usecs, bandwidth 18555 MB in 3.4493 sec = 5379.37 MB/s
polynomial_code_pss_cold: k=20 p=8 runtime = 3059284 usecs, bandwidth 32849 MB in 3.0593 sec = 10737.57 MB/s
done all: Pass

The first test is encoding with a dot product, my rate was 9858.54 MB. The second test is encoding with an Parallel LFSR Sequencer, my rate was 14076.79. The third test is producing syndromes with a dot product, my rate was 5379.37. The last was producing (and testing) syndromes with a Parallel Syndrome Sequencer, my rate was 10737.57. Quite a substantial increase in performance on my system (my little Acer with AVX512GFNI), I would imagine you would see something similar (only higher).

This code basically replaces all the ISA-L ASM code with Intrinsics in C language, while simultaneously increasing performance. You can see the Intrinsic source code in erasure_code/PCLib_AVX512_GFNI.c

```

### Validated Platforms

Build and executables validated on x86 Ubuntu 24.04.2 LTS, ARM Raspberry Pi 5 Debian GNU/Linux 12 (bookworm) and Windows 11/Visual Studio 2022 x64 command line.