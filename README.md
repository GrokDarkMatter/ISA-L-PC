ISA-L-PC
--------

# Performance Analysis of Erasure and Polynomial Coding

The following graph shows the bandwidth performance (in MB/s) of ISA-L (bottom 2) vs. ISA-L-PC (top 2) as the number of CPU cores increases from 1 to 24. The data is sourced from `results.poly_core_perf.txt`.

![Performance Graph](https://raw.githubusercontent.com/GrokDarkMatter/ISA-L-PC/master/performance.png)

# LFSR Structure and Functionality

![LFSR Diagram](https://raw.githubusercontent.com/GrokDarkMatter/ISA-L-PC/master/LFSR.png)

# Optimized and Vectorized Reed Solomon Encoding and Error Decoding with AVX512 and GFNI

### Overview 
The `PCLib_AVX512_GFNI` library provides functions for single level error correction and encoding/decoding using finite field arithmetic, specifically designed to leverage AVX-512 and GFNI (Galois Field New Instructions) for performance optimization. The library is capable of handling operations required for Reed-Solomon codes and implementations of linear feedback shift registers (LFSR).

[API Details for PCLib_AVX512_GFNI](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/Documentation/PCLib_AVX512_GFNI_READ_ME.md)
# Multi-Level Reed Solomon Encoding/Decoding with AVX512 and GFNI
## Overview

The `PCLib_2D_AVX512_GFNI` library implements a 2D error correction coding system using Reed-Solomon codes with optimizations for AVX512 hardware. It provides encoding and decoding functionalities leveraging advanced mathematical operations to handle errors in data transmission. This code specifically uses parallel operations to improve performance, particularly for large data sets.

[API Details for PCLib_2D_AVX512_GFNI](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/Documentation/PCLib_2D_AVX512_GFNI_READ_ME.md)

# Performance Tests

[This Document](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/Documentation/erasure_code_perf_READ_ME.md) details the functionality and usage of the erasure_code_perf.c file. The software is developed for high-performance erasure coding, which is widely used for data recovery in distributed storage systems. Single threaded single level RS coding is measured and compared to ISA-L.

[This Document](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/Documentation/poly_code_perf_READ_ME.md) details the functionality and usage of the poly_code_perf.c file. The software is developed for high-performance erasure coding, which is widely used for data recovery in distributed storage systems. Single threaded multi-level RS coding is measured and compared to ISA-L

[This Document](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/Documentation/poly_core_perf_READ_ME.md) details the functionality and usage of the poly_core_perf.c file. The software is developed for high-performance erasure coding, which is widely used for data recovery in distributed storage systems. Multi-threaded multi-level RS encoding is measured and compared to ISA-L

### Performance Charts for x86 and ARM64

Here's a summary of single-core ISA-L vs ISA-L-PC performance: [View Performance Charts](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/ISA-LvISA-L-PC.pdf)

Here's a summary of single-core ISA-L vs. ISA-L-PC for ARM64: [View Performance Charts](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/ISA-LvISA-L-PC_AARCH64.pdf)

### Building ISA-L-PC

If you'd like to build ISA-L-PC on Windows, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/3

If you'd like to build ISA-L-PC on Linux, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/5

If you'd like to build ISA-L-PC for ARM64, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/4

If you'd like to build (and run) ISA-L-PC for your Android Cell Phone or Tablet, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/7

### Background papers

[Information Dispersal Matrices for RAID Error Correcting Codes](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/IDM.pdf)

[On the reliability of RAID systems: An Argument for More Check Drives](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/ECRAIDReliability.pdf)

[An Erasure coding Performance Metric for Windows 8](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/ECPWin8g.pdf)

[Our work with Baylor University - Guide to Reed Solomon Codes](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/BaylorWork-b188a16.pdf)

[Polynomial Encoding Table Values](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/PolyCodeValues.pdf)

### Battling Silent Data Corruption: Empowering Reliability with Modern Vector Processors

In today's data-intensive landscape, silent data corruption (SDC) poses a stealthy risk to computational integrity, where errors creep into systems undetected, potentially skewing AI models, financial calculations, or scientific simulations. CERN's groundbreaking 2007 study exposed SDC's reach, analyzing over 3,000 machines and 1.46 million device hours to uncover corruption in 1.8% of devices, stemming from hardware flaws, firmware glitches, and memory failures—resulting in 22 corrupted files across 8.7 TB of data. Hyperscalers like Meta, Google, and Microsoft now report SDC impacting one in a thousand machines in their vast fleets, fueling urgent calls for resilient solutions in cloud and AI environments.

Fortunately, modern vector processors—with SIMD (Single Instruction, Multiple Data) units in architectures like x86 (AVX), ARM (SVE), and RISC-V—offer a game-changing defense: application-level error detection and correction at full speed. By harnessing SIMD for parallel data operations, these processors enable in-software checks and fixes, outpacing traditional hardware ECC. Innovations like those in U.S. Patent 11,848,686 demonstrate this power, using SIMD-accelerated polynomial coding with linear feedback shift registers (LFSR) to efficiently compute check data and reconstruct corrupted elements. Unlike device-specific safeguards, CPU-level detection blankets errors from any source—disks, controllers, flash, or memory—delivering end-to-end protection.

This shift minimizes overhead via vector processing while empowering developers to embed resilience into applications, from AI workloads to databases. As detailed in U.S. Patent 11,848,686, parallel multipliers and syndrome sequencers in SIMD cores streamline Galois field operations, making high-speed ECC feasible for exascale systems. As SDC threats intensify, embracing vector processor solutions isn't optional—it's the key to unbreakable, efficient data ecosystems.

### New 2 Dimensional Reed Solomon Library Available

In the era of exabyte-scale data centers, silent data corruption (SDC) lurks as an invisible saboteur, silently corrupting "cold" archives—those vast troves of infrequently accessed but mission-critical data like backups, logs, and historical datasets. A 2023 Google study revealed SDC rates climbing to 1 in 500 drives annually in hyperscale environments, while Meta's reports highlight how bit flips from cosmic rays, hardware wear, or firmware bugs can cascade into catastrophic losses during rare reads. For data centers drowning in petabytes of cold storage, where verification is sporadic, these undetected errors threaten compliance, AI training integrity, and recovery from disasters like ransomware.

Enter a potent weapon: two-dimensional Reed-Solomon (2D-RS) encoding, an ECC powerhouse that improves upon legacy tape technologies for disk and cloud resilience. My new open-source library, ISA-L-PC, delivers exactly this—leveraging GFNI instructions for blazing-fast SIMD acceleration. It applies RS(64,60) encoding within blocks for intra-level correction, then layers a flexible RS code (up to 255 symbols) across blocks for inter-level fortification. With full source code, you can tweak both levels to any valid 8-bit configuration, tailoring protection to your workload.

[LTO-9 Technical Paper](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/LTO-UBER-Technical-Paper-August-2022.pdf)

Drawing from the LTO-9 technical paper's rigorous analysis, this 2D-RS approach yields 10X UBER improvements over single-level codes—pushing uncorrectable bit error rates to 10^-20, or 12+ NINES of durability. Even with degraded channels (think debris-clogged reads or flaky disks), it sustains 10^-19 UBER at input rates 99X higher than legacy systems, outshining HDDs by two orders of magnitude in cold data scenarios. No more latent sector panics; this blankets end-to-end protection against write errors, media defects, and aging.

Ideal for data centers stockpiling cold data, ISA-L-PC embeds resilience without hardware overhauls—perfect for erasure-coded archives or AI data lakes. The techniques are patented (e.g., U.S. Patent 11,848,686 and others), but licensed at reasonable rates for commercial use. Download, evaluate it free today, and fortify your ecosystem against SDC's stealth.

POLYNOMIAL ENCODING
-------------------

### Updates for Polynomial Encoding

Changes (c) Copyright 2025 by Michael H. Anderson. All rights reserved. See LICENSE file for details.

This version of ISA-L has been adapted to implement an Accelerated Polynomial System and Method, as described in US Patents 11,848,686 and 12,341,532. See https://patents.justia.com/patent/11848686.

### Validated Platforms

Build and executables validated on x86 Ubuntu 24.04.2 LTS, ARM Raspberry Pi 5 Debian GNU/Linux 12 (bookworm) and Windows 11/Visual Studio 2022 x64 command line.
