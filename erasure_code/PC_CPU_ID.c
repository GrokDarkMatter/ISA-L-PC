#ifdef _WIN64
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <intrin.h>

void
get_cpu_brand (char *brand)
{
    int regs[ 4 ];
    char *p = brand;
    unsigned int max_level;

    __cpuid (regs, 0x80000000);
    max_level = regs[ 0 ];

    if (max_level < 0x80000004)
    {
        strcpy (brand, "Unknown");
        return;
    }

    for (unsigned int i = 0x80000002; i <= 0x80000004; ++i)
    {
        __cpuid (regs, i);
        memcpy (p, regs, sizeof (regs));
        p += sizeof (regs);
    }
    *p = '\0';

    // Trim leading/trailing spaces
    p = brand + strlen (brand) - 1;
    while (p > brand && *p == ' ')
        *p-- = '\0';
    while (*brand == ' ')
        ++brand;
}

int
check_gfni_support ()
{
    int regs[ 4 ];
    __cpuid (regs, 0x7);                // Leaf 0x7, subleaf 0
    return (regs[ 2 ] & (1 << 8)) != 0; // Check ECX bit 8 for GFNI
}

int
PC_CPU_ID ()
{
    char brand[ 64 ] = { 0 }; // Initialize to avoid garbage
    get_cpu_brand (brand);

    SYSTEM_INFO si;
    GetSystemInfo (&si);

    printf ("Processor Brand: %s\n", brand);
    printf ("Number of Logical Processors: %u\n", si.dwNumberOfProcessors);
    printf ("Processor Architecture: ");
    switch (si.wProcessorArchitecture)
    {
    case PROCESSOR_ARCHITECTURE_AMD64:
        printf ("x64\n");
        break;
    case PROCESSOR_ARCHITECTURE_INTEL:
        printf ("x86\n");
        break;
    case PROCESSOR_ARCHITECTURE_ARM:
        printf ("ARM\n");
        break;
    default:
        printf ("Unknown\n");
        break;
    }

    printf ("GFNI Support: %s\n", check_gfni_support () ? "Yes" : "No");

    HKEY hKey;
    DWORD mhz = 0;
    DWORD size = sizeof (DWORD);
    if (RegOpenKeyEx (HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                      KEY_READ, &hKey) == ERROR_SUCCESS)
    {
        RegQueryValueEx (hKey, "~MHz", NULL, NULL, (LPBYTE) &mhz, &size);
        RegCloseKey (hKey);
        printf ("Approximate Clock Speed: %u MHz\n", mhz);
    }
    else
    {
        printf ("Failed to retrieve clock speed from registry\n");
    }

    return si.dwNumberOfProcessors;
}
#else
#include <stdio.h>
#include <string.h>
#ifndef __aarch64__
#include <cpuid.h>

int
check_gfni_support (void)
{
    unsigned int eax, ebx, ecx, edx;
    __cpuid (0x7, eax, ebx, ecx, edx); // Leaf 0x7, subleaf 0
    return (ecx & (1 << 8)) != 0;      // Check ECX bit 8 for GFNI
}
#endif
int
PC_CPU_ID (void)
{
    FILE *fp = fopen ("/proc/cpuinfo", "r");
    if (fp == NULL)
    {
        printf ("Failed to open /proc/cpuinfo\n");
        return 1;
    }

    int s = 0, Cores = 0, m;
    char line[ 1024 ];
    printf ("Processor Details:\n");
    while (fgets (line, sizeof (line), fp))
    {
        // Print specific fields for name and details
        m = 0;
        if (strncmp (line, "vendor_id", 9) == 0)
            m = 1;
        if (strncmp (line, "processor", 9) == 0)
            m = 1;
        if (strncmp (line, "CPU architecture", 16) == 0)
        {
            if (strncmp (line, "CPU architecture: 8", 19) == 0)
            {
                strncpy (line, "CPU architecture: ARMv8\n", 25);
            }
            m = 1;
        }
        if (strncmp (line, "cpu family", 10) == 0)
            m = 1;
        if (strncmp (line, "CPU implementer", 15) == 0)
        {
            if (strncmp (line, "CPU implementer	: 0x41", 22) == 0)
            {
                strncpy (line, "CPU Implementer : ARM Holdings\n", 32);
                ;
            }
            m = 1;
        }
        if (strncmp (line, "CPU variant", 11) == 0)
            m = 1;
        if (strncmp (line, "CPU part", 8) == 0)
        {
            if (strncmp (line, "CPU part	: 0xd0b", 16) == 0)
            {
                strncpy (line, "CPU part	: Cortex-A76 core\n", 28);
                ;
            }
            m = 1;
        }
        if (strncmp (line, "CPU revision", 12) == 0)
            m = 1;
        if (strncmp (line, "BogoMIPS", 8) == 0)
            m = 1;
        if (strncmp (line, "cpu family", 10) == 0)
            m = 1;
        if (strncmp (line, "model\t", 6) == 0)
            m = 1;
        if (strncmp (line, "model name", 10) == 0)
            m = 1;
        if (strncmp (line, "stepping", 8) == 0)
            m = 1;
        if (strncmp (line, "microcode", 9) == 0)
            m = 1;
        if (strncmp (line, "cpu MHz", 7) == 0)
            m = 1;
        if (m)
        {
            if (s == 0)
                printf ("%s", line);
            if (strncmp (line, "cpu family", 10) == 0)
                Cores++;
            if (strncmp (line, "processor", 9) == 0)
                Cores++;
            if (strncmp (line, "cpu MHz", 7) == 0)
                s = 1;
            if (strncmp (line, "CPU revision", 12) == 0)
                s = 1;
        }
    }
    printf ("cpu cores       : %d\n", Cores);
#ifndef __aarch64__
    printf ("GFNI Support    : %s\n", check_gfni_support () ? "Yes" : "No");
#endif
    fclose (fp);
    return Cores;
}
#endif
