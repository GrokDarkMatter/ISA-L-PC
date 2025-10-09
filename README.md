ISA-L-PC
--------

### Performance Charts for x86 and ARM64

Here's a summary of ISA-L vs ISA-L-PC performance: [View Performance Charts](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/ISA-LvISA-L-PC.pdf)

Here's a summary of ISA-L vs. ISA-L-PC for ARM64: [View Performance Charts](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/ISA-LvISA-L-PC_AARCH64.pdf)

If you'd like to build ISA-L-PC on Windows, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/3

If you'd like to build ISA-L-PC on Linux, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/5

If you'd like to build ISA-L-PC for ARM64, here are some tips: https://github.com/GrokDarkMatter/ISA-L-PC/issues/4

In addition to performance, ISA-L offers both 1D and 2D Reed Solomon error encoding and decoding.

### Background papers

[An Erasure coding Performance Metric for Windows 8](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/ECPWin8g.pdf)

[On the reliability of RAID systems: An Argument for More Check Drives](https://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/ECRAIDReliability.pdf)


[Our work with Baylor University - Guide to Reed Solomon Codes](hhttps://github.com/GrokDarkMatter/ISA-L-PC/blob/master/MathPapers/BaylorWork-b188a16.pdf)
### Battling Silent Data Corruption: Empowering Reliability with Modern Vector Processors

In today's data-intensive landscape, silent data corruption (SDC) poses a stealthy risk to computational integrity, where errors creep into systems undetected, potentially skewing AI models, financial calculations, or scientific simulations. CERN's groundbreaking 2007 study exposed SDC's reach, analyzing over 3,000 machines and 1.46 million device hours to uncover corruption in 1.8% of devices, stemming from hardware flaws, firmware glitches, and memory failures—resulting in 22 corrupted files across 8.7 TB of data. Hyperscalers like Meta, Google, and Microsoft now report SDC impacting one in a thousand machines in their vast fleets, fueling urgent calls for resilient solutions in cloud and AI environments.

Fortunately, modern vector processors—with SIMD (Single Instruction, Multiple Data) units in architectures like x86 (AVX), ARM (SVE), and RISC-V—offer a game-changing defense: application-level error detection and correction at full speed. By harnessing SIMD for parallel data operations, these processors enable in-software checks and fixes, outpacing traditional hardware ECC. Innovations like those in U.S. Patent 11,848,686 demonstrate this power, using SIMD-accelerated polynomial coding with linear feedback shift registers (LFSR) to efficiently compute check data and reconstruct corrupted elements. Unlike device-specific safeguards, CPU-level detection blankets errors from any source—disks, controllers, flash, or memory—delivering end-to-end protection.

This shift minimizes overhead via vector processing while empowering developers to embed resilience into applications, from AI workloads to databases. As detailed in U.S. Patent 11,848,686, parallel multipliers and syndrome sequencers in SIMD cores streamline Galois field operations, making high-speed ECC feasible for exascale systems. As SDC threats intensify, embracing vector processor solutions isn't optional—it's the key to unbreakable, efficient data ecosystems.

### New 2 Dimensional Reed Solomon Library Available

In the era of exabyte-scale data centers, silent data corruption (SDC) lurks as an invisible saboteur, silently corrupting "cold" archives—those vast troves of infrequently accessed but mission-critical data like backups, logs, and historical datasets. A 2023 Google study revealed SDC rates climbing to 1 in 500 drives annually in hyperscale environments, while Meta's reports highlight how bit flips from cosmic rays, hardware wear, or firmware bugs can cascade into catastrophic losses during rare reads. For data centers drowning in petabytes of cold storage, where verification is sporadic, these undetected errors threaten compliance, AI training integrity, and recovery from disasters like ransomware.

Enter a potent weapon: two-dimensional Reed-Solomon (2D-RS) encoding, an ECC powerhouse that improves upon legacy tape technologies for disk and cloud resilience. My new open-source library, ISA-L-PC, delivers exactly this—leveraging Intel's GFNI instructions for blazing-fast SIMD acceleration. It applies RS(64,60) encoding within blocks for intra-level correction, then layers a flexible RS code (up to 255 symbols) across blocks for inter-level fortification. With full source code, you can tweak both levels to any valid 8-bit configuration, tailoring protection to your workload.

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
