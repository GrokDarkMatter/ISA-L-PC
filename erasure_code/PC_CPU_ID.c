#ifdef _WIN64
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <intrin.h>

void get_cpu_brand(char* brand) {
    int regs[4];
    char* p = brand;
    unsigned int max_level;

    __cpuid(regs, 0x80000000);
    max_level = regs[0];

    if (max_level < 0x80000004) {
        //strcpy(brand, "Unknown");
        return;
    }

    for (unsigned int i = 0x80000002; i <= 0x80000004; ++i) {
        __cpuid(regs, i);
        memcpy(p, regs, sizeof(regs));
        p += sizeof(regs);
    }
    *p = '\0';

    // Trim leading/trailing spaces
    p = brand + strlen(brand) - 1;
    while (p > brand && *p == ' ') *p-- = '\0';
    while (*brand == ' ') ++brand;
}

int PC_CPU_ID() {
    char brand[64];
    get_cpu_brand(brand);

    SYSTEM_INFO si;
    GetSystemInfo(&si);

    printf("Processor Brand: %s\n", brand);
    printf("Number of Logical Processors: %u\n", si.dwNumberOfProcessors);
    printf("Processor Architecture: ");
    switch (si.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64: printf("x64\n"); break;
    case PROCESSOR_ARCHITECTURE_INTEL: printf("x86\n"); break;
    case PROCESSOR_ARCHITECTURE_ARM: printf("ARM\n"); break;
    default: printf("Unknown\n");
    }

    HKEY hKey;
    DWORD mhz = 0;
    DWORD size = sizeof(DWORD);
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegQueryValueEx(hKey, L"~MHz", NULL, NULL, (LPBYTE)&mhz, &size); // Use LPBYTE
        RegCloseKey(hKey);
        printf("Approximate Clock Speed: %u MHz\n", mhz);
    }
    else {
        //printf("Failed to retrieve clock speed from registry\n");
    }

    return 0;
}
#else
#endif