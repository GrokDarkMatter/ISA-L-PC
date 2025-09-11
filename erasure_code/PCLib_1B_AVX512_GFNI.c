static unsigned char pc_ptab [ 256 ], pc_ltab [ 256 ], pc_itab [ 256 ] ;
static __m512i Vand1b [ 255 ] [ 4 ] ;
static __m512i EncMat [ 255 ] [ 4 ] ;

// Multiply two bytes using the hardware GF multiply
unsigned char pc_mul_1b ( unsigned char a, unsigned char b) 
{
    __m128i va, vb ;

    unsigned char * veca = ( unsigned char * ) &va ;
    unsigned char * vecb = ( unsigned char * ) &vb ;
    *veca = a ;
    *vecb = b ;

    va = _mm_gf2p8mul_epi8 ( va, vb ) ;
    return *veca ;
}

// pc_bpow - Build a table of power values
void pc_bpow ( unsigned char Gen )
{
    int i ;


    // A positive integer raised to the power 0 is one
    pc_ptab [ 0 ] = 1 ;

    // Two is a good generator for 0x1d, three is a good generator for 0x1b
    for ( i = 1 ; i < 256 ; i ++ )
    {
        pc_ptab[ i ] = pc_mul_1b ( pc_ptab [ i - 1 ], Gen ) ;
    }
}

// pc_blog - Use the power table to build the log table
void pc_blog ( void ) 
{
    int i ;


    // Use the power table to index into the log table and store log value
    for ( i = 0 ; i < 256 ; i ++ )
    {
        pc_ltab [ pc_ptab [ i ] ] = i ;
    }
}

// pc_linv - Calculate the inverse of a number, that is, 1/Number
void  pc_binv ( void ) 
{
    int i ;
    for ( i = 0 ; i < 256 ; i ++ )
    {
        pc_itab [ i ] = pc_ptab [ 255 - pc_ltab [ i ] ] ;
    }
}

void pc_gen_rsr_matrix_1b(unsigned char *a, int k)
{
        int i, j;
        unsigned char p, gen = 1;

        // Loop through rows and cols backward
        for (i = k-1; i>=0; i--)
        {
                p = 1;
                for ( j = 0; j <= 255 ; j++ )
                {
                        a[255 * i + (255 - j - 1)] = p;
                        p = pc_mul_1b(p, gen);
                }
                gen = gf_mul(gen, 3);
        }
}

// Initialize encoding matrix for encoding
void pc_bmat ( unsigned char * vals, int p )
{
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        printf ( "Loop %d\n", curP ) ;
        memcpy ( ( unsigned char * ) &EncMat [ curP ], &vals [ curP * ( 255 - p ) ], 255 - p ) ;
        unsigned char * extra = ( unsigned char * ) &EncMat [ curP ] ;
        memset ( extra + 255 - p, 0, p ) ;
    }
}

// Initialize vandermonde matrix for decoding
void pc_bvan ( unsigned char * vals, int p )
{
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        printf ( "Loop %d\n", curP ) ;
        memcpy (  ( unsigned char * ) &Vand1b [ curP ], &vals [ curP * ( 255 ) ], 255 ) ;
        unsigned char * extra = ( unsigned char * ) &Vand1b [ curP ] ;
        extra [ 255 ] = 0 ;
    }
}

// Encode using the Vandermonde matrix
void pc_encoder1b ( unsigned char * codeWord, unsigned char * par, int p ) 
{
    __m512i codeWordvec [ 4 ], encMatvec [ 4 ], vreg [ 4 ] ;
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        printf ( "curP = %d\n", curP ) ;
        codeWordvec [ 0 ] = _mm512_loadu_si512 ( codeWord + 0 * 64 ) ;
        codeWordvec [ 1 ] = _mm512_loadu_si512 ( codeWord + 1 * 64 ) ;
        codeWordvec [ 2 ] = _mm512_loadu_si512 ( codeWord + 2 * 64 ) ;
        codeWordvec [ 3 ] = _mm512_loadu_si512 ( codeWord + 3 * 64 ) ;
        printf ( "Codeword LSB\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &codeWordvec [ 0 ], 1, 255 ) ;
        printf ( "Encmat\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &EncMat [ curP ] [ 0 ], 1, 255 ) ;
        encMatvec [ 0 ] = _mm512_loadu_si512 ( EncMat [ curP ] ) ;
        encMatvec [ 1 ] = _mm512_loadu_si512 ( &EncMat [ curP ] [ 1 ] ) ;
        encMatvec [ 2 ] = _mm512_loadu_si512 ( &EncMat [ curP ] [ 2 ] ) ;
        encMatvec [ 3 ] = _mm512_loadu_si512 ( &EncMat [ curP ] [ 3 ] ) ;
        printf ( "EncMatvec\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &encMatvec [ 0 ], 1, 256 ) ;


        printf ( "Load done\n" ) ;
        vreg [ 0 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 0 ], encMatvec [ 0 ] ) ;
        vreg [ 1 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 1 ], encMatvec [ 1 ] ) ;
        vreg [ 2 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 2 ], encMatvec [ 2 ] ) ;
        vreg [ 3 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 3 ], encMatvec [ 3 ] ) ;

        printf ( "vreg3\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &vreg [ 3 ], 1, 64 ) ;
        // Add up the 4 64 byte registers
        vreg [ 0 ] = _mm512_or_si512 ( vreg [ 0 ], vreg [ 1 ] ) ;
        vreg [ 0 ] = _mm512_or_si512 ( vreg [ 0 ], vreg [ 2 ] ) ;
        vreg [ 0 ] = _mm512_or_si512 ( vreg [ 0 ], vreg [ 3 ] ) ;

        // Shuffle and XOR 512-bit to 256-bit
        __m256i low = _mm512_castsi512_si256(vreg [ 0 ] );
        __m256i high = _mm512_extracti64x4_epi64(vreg [ 0 ], 1);
        __m256i xored = _mm256_xor_si256(low, high);

        // Shuffle and XOR 256-bit to 128-bit
        __m128i low128 = _mm256_castsi256_si128(xored);
        __m128i high128 = _mm256_extracti128_si256(xored, 1);
        __m128i xored128 = _mm_xor_si128(low128, high128);

        // Shuffle 128-bit to 64-bit using permute
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i xored64 = _mm_xor_si128(xored128, perm);

        // Reduce 64-bit to 32-bit
        uint64_t result_64 = _mm_cvtsi128_si64(xored64);
        result_64 ^= result_64 >> 32 ;
        result_64 ^= result_64 >> 16 ;
        result_64 ^= result_64 >> 8 ;
        printf ( "Before store\n" ) ;
        par [ curP ] = ( unsigned char ) result_64 ;
        printf ( "parity=%d\n", (unsigned char) result_64 ) ;
    }
}

// Encode using the Vandermonde matrix
void pc_decoder1b ( unsigned char * codeWord, unsigned char * syn, int p ) 
{
    __m512i codeWordvec [ 4 ], vanMatvec [ 4 ], vreg [ 4 ] ;
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        printf ( "curP = %d\n", curP ) ;
        codeWordvec [ 0 ] = _mm512_loadu_si512 ( codeWord + 0 * 64 ) ;
        codeWordvec [ 1 ] = _mm512_loadu_si512 ( codeWord + 1 * 64 ) ;
        codeWordvec [ 2 ] = _mm512_loadu_si512 ( codeWord + 2 * 64 ) ;
        codeWordvec [ 3 ] = _mm512_loadu_si512 ( codeWord + 3 * 64 ) ;
        printf ( "Codeword LSB\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &codeWordvec [ 0 ], 1, 255 ) ;
        printf ( "Vandermonde\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &Vand1b [ curP ] [ 0 ], 1, 255 ) ;

        vanMatvec [ 0 ] = _mm512_loadu_si512 ( &Vand1b [ curP ] [ 0 ]) ;
        vanMatvec [ 1 ] = _mm512_loadu_si512 ( &Vand1b [ curP ] [ 1 ] ) ;
        vanMatvec [ 2 ] = _mm512_loadu_si512 ( &Vand1b [ curP ] [ 2 ] ) ;
        vanMatvec [ 3 ] = _mm512_loadu_si512 ( &Vand1b [ curP ] [ 3 ] ) ;
        printf ( "VandVec\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &vanMatvec [ 0 ], 1, 255 ) ;


        printf ( "Load done\n" ) ;
        vreg [ 0 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 0 ], vanMatvec [ 0 ] ) ;
        vreg [ 1 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 1 ], vanMatvec [ 1 ] ) ;
        vreg [ 2 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 2 ], vanMatvec [ 2 ] ) ;
        vreg [ 3 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 3 ], vanMatvec [ 3 ] ) ;

        printf ( "vreg3\n" ) ;
        dump_u8xu8 ( ( unsigned char * ) &vreg [ 3 ], 1, 64 ) ;
        // Add up the 4 64 byte registers
        vreg [ 0 ] = _mm512_or_si512 ( vreg [ 0 ], vreg [ 1 ] ) ;
        vreg [ 0 ] = _mm512_or_si512 ( vreg [ 0 ], vreg [ 2 ] ) ;
        vreg [ 0 ] = _mm512_or_si512 ( vreg [ 0 ], vreg [ 3 ] ) ;

        // Shuffle and XOR 512-bit to 256-bit
        __m256i low = _mm512_castsi512_si256(vreg [ 0 ] );
        __m256i high = _mm512_extracti64x4_epi64(vreg [ 0 ], 1);
        __m256i xored = _mm256_xor_si256(low, high);

        // Shuffle and XOR 256-bit to 128-bit
        __m128i low128 = _mm256_castsi256_si128(xored);
        __m128i high128 = _mm256_extracti128_si256(xored, 1);
        __m128i xored128 = _mm_xor_si128(low128, high128);

        // Shuffle 128-bit to 64-bit using permute
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i xored64 = _mm_xor_si128(xored128, perm);

        // Reduce 64-bit to 32-bit
        uint64_t result_64 = _mm_cvtsi128_si64(xored64);
        result_64 ^= result_64 >> 32 ;
        result_64 ^= result_64 >> 16 ;
        result_64 ^= result_64 >> 8 ;
        printf ( "Before store\n" ) ;
        syn [ curP ] = ( unsigned char ) result_64 ;
        printf ( "syndrome=%d\n", (unsigned char) result_64 ) ;
    }
}

// Utility routine to sum across a ZMM register
unsigned char sum_zmm_AVX512_GFNI(__m512i * zmm) 
{
    __m512i vreg ;
    vreg = _mm512_loadu_si512( zmm ) ;

    // Shuffle and XOR 512-bit to 256-bit
    __m256i low = _mm512_castsi512_si256(vreg);
    __m256i high = _mm512_extracti64x4_epi64(vreg, 1);
    __m256i xored = _mm256_xor_si256(low, high);

    // Shuffle and XOR 256-bit to 128-bit
    __m128i low128 = _mm256_castsi256_si128(xored);
    __m128i high128 = _mm256_extracti128_si256(xored, 1);
    __m128i xored128 = _mm_xor_si128(low128, high128);

    // Shuffle 128-bit to 64-bit using permute
    __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2));
    __m128i xored64 = _mm_xor_si128(xored128, perm);

    // Reduce 64-bit to 32-bit
    uint64_t result_64 = _mm_cvtsi128_si64(xored64);
    result_64 ^= result_64 >> 32 ;
    result_64 ^= result_64 >> 16 ;
    result_64 ^= result_64 >> 8 ;
    return ( unsigned char ) result_64 ;
}

void pc_gen_poly_1b( unsigned char *p, int rank)
{
        int c, alpha, cr ; // Loop variables

        p [ 0 ] = 1 ; // Start with (x+1)
        alpha = 3 ;
        for ( cr = 1 ; cr < rank ; cr ++ ) // Loop rank-1 times
        {
                // Compute the last term of the polynomial by multiplying
                p [ cr ] = pc_mul_1b ( p [ cr - 1 ], alpha ) ;

                // Pass the middle terms to produce multiply result
                for ( c = cr - 1 ; c > 0 ; c -- )
                {
                        p [ c ] ^= pc_mul_1b ( p [ c - 1 ], alpha ) ;
                }

                // Compute the first term by adding in alphaI
                p [ 0 ] ^= alpha ;

                // Compute next alpha (power of 2)
                alpha = pc_mul_1b ( alpha, 3 ) ;
        }
}

void pc_gen_poly_matrix_1b(unsigned char *a, int m, int k)
{
        int i, j, par, over, lpos ;
        unsigned char *p, taps [ 254 ], lfsr [ 254 ] ;

        // First compute the generator polynomial and initialize the taps
        par = m - k ;

        pc_gen_poly_1b ( taps, par ) ;
        printf ( "Poly parites=%d\n", par ) ;
        dump_u8xu8 ( taps, 1, par ) ;
        memcpy ( lfsr, taps, par ) ; // Initial value of LFSR is the taps

        // Now use an LFSR to build the values
        p = a ;
        for ( i = k - 1 ; i >= 0 ; i-- ) // Outer loop for each col
        {
                for (j = 0; j < par ; j++) // Each row
                {
                        // Copy in the current LFSR values
                        p [ ( j * k ) + i ] = lfsr [ j ] ;
                }
                // Now update values with LFSR - first compute overflow
                over = lfsr [ 0 ] ;

                // Loop through the MSB LFSR terms (not the LSB)
                for ( lpos = 0 ; lpos < par - 1 ; lpos ++ )
                {
                        lfsr [ lpos ] = pc_mul_1b ( over, taps [ lpos ] ) ^ lfsr [ lpos + 1 ] ;
                }
                // Now do the LSB of the LFSR to finish
                lfsr [ par - 1 ] = pc_mul_1b ( over, taps [ par - 1 ] ) ;
        }
}
