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
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegQueryValueEx(hKey, "~MHz", NULL, NULL, (LPBYTE)&mhz, &size); // Use LPBYTE
        RegCloseKey(hKey);
        printf("Approximate Clock Speed: %u MHz\n", mhz);
    }
    else {
        //printf("Failed to retrieve clock speed from registry\n");
    }

    return 0;
}
#else
#include <stdio.h>
#include <string.h>

int PC_CPU_ID( void ) {
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp == NULL) {
        printf("Failed to open /proc/cpuinfo\n");
        return 1;
    }

    int s=0, Cores = 0;
    char line[1024];
    printf("Processor Details:\n");
    while (fgets(line, sizeof(line), fp)) {
        // Print specific fields for name and details
        if (strncmp(line, "vendor_id", 9) == 0 ||
            strncmp(line, "processor", 9) == 0 ||
            strncmp(line, "CPU architecture", 16) == 0 ||
            strncmp(line, "cpu family", 10) == 0 ||
            strncmp(line, "CPU implementer", 15) == 0 ||
            strncmp(line, "CPU variant", 11) == 0 ||
            strncmp(line, "CPU part", 8) == 0 ||
            strncmp(line, "CPU revision", 12) == 0 ||
            strncmp(line, "BogoMIPS", 8) == 0 ||
            strncmp(line, "cpu family", 10) == 0 ||
            strncmp(line, "model\t", 6) == 0 ||
            strncmp(line, "model name", 10) == 0 ||
            strncmp(line, "stepping", 8) == 0 ||
            strncmp(line, "microcode", 9) == 0 ||
            strncmp(line, "cpu MHz", 7) == 0) {
 //           strncmp(line, "flags", 5) == 0) {
            if ( s == 0 ) printf("%s", line);
            if ( strncmp(line, "cpu family", 10) == 0 ) Cores ++ ;
            if ( strncmp(line, "processor", 9) == 0 ) Cores ++ ;
            if ( strncmp(line, "cpu MHz", 7 ) == 0 ) s=1 ;
            if ( strncmp(line, "CPU revision", 12 ) == 0 ) s=1 ;
        }
    }
    printf ( "cpu cores	: %d\n", Cores ) ;

    fclose(fp);
    return 0;
}
#endif
