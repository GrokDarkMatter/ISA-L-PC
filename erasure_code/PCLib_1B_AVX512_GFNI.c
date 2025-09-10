static unsigned char pc_ptab [ 256 ], pc_ltab [ 256 ], pc_itab [ 256 ] ;

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
void
pc_gen_poly_1b( unsigned char *p, int rank)
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

void
pc_gen_poly_matrix_1b(unsigned char *a, int m, int k)
{
        int i, j, par, over, lpos ;
        unsigned char *p, taps [ 254 ], lfsr [ 254 ] ;

        // First compute the generator polynomial and initialize the taps
        par = m - k ;

        pc_gen_poly_1b ( taps, par ) ;
        printf ( "Poly\n" ) ;
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

