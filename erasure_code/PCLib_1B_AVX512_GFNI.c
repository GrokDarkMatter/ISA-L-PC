static unsigned char pc_ptab_1b [ 256 ], pc_ltab_1b [ 256 ], pc_itab_1b [ 256 ] ;
static __m512i EncMat [ 255 ] [ 4 ] ;
static __m512i Vand1b [  32 ] [ 4 ] ;
#ifdef NDEF
static __m512i Vand2 [ 2 ] [ 4 ] ;
#endif

#define L1Enc(vec,p,pvec) \
pvec = _mm512_xor_si512 ( pvec, pvec ) ; \
for ( int curP = 0 ; curP < p ; curP ++ ) \
{ \
        matVec = _mm512_load_si512 ( &EncMat [ curP ] [ 0 ]) ; \
        vreg = _mm512_gf2p8mul_epi8 ( vec, matVec ) ; \
        __m256i low = _mm512_castsi512_si256(vreg); \
        __m256i high = _mm512_extracti64x4_epi64(vreg, 1); \
        __m256i xored = _mm256_xor_si256(low, high); \
        __m128i low128 = _mm256_castsi256_si128(xored); \
        __m128i high128 = _mm256_extracti128_si256(xored, 1); \
        __m128i xored128 = _mm_xor_si128(low128, high128); \
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2)); \
        __m128i xored64 = _mm_xor_si128(xored128, perm); \
        uint64_t result_64 = _mm_cvtsi128_si64(xored64); \
        result_64 ^= result_64 >> 32 ; \
        result_64 ^= result_64 >> 16 ; \
        result_64 ^= result_64 >> 8 ; \
        pp [ curP ] = ( unsigned char ) result_64 ; \
} \
vec = _mm512_xor_si512 ( vec, pvec ) ;

#define L1Dec(vec,p,err,syn) \
for ( int curP = 0 ; curP < p ; curP ++ ) \
{ \
        matVec = _mm512_load_si512 ( &Vand1b [ curP ] [ 0 ]) ; \
        vreg = _mm512_gf2p8mul_epi8 ( vec, matVec ) ; \
        __m256i low = _mm512_castsi512_si256(vreg); \
        __m256i high = _mm512_extracti64x4_epi64(vreg, 1); \
        __m256i xored = _mm256_xor_si256(low, high); \
        __m128i low128 = _mm256_castsi256_si128(xored); \
        __m128i high128 = _mm256_extracti128_si256(xored, 1); \
        __m128i xored128 = _mm_xor_si128(low128, high128); \
        __m128i perm = _mm_shuffle_epi32(xored128, _MM_SHUFFLE(3, 2, 3, 2)); \
        __m128i xored64 = _mm_xor_si128(xored128, perm); \
        uint64_t result_64 = _mm_cvtsi128_si64(xored64); \
        result_64 ^= result_64 >> 32 ; \
        result_64 ^= result_64 >> 16 ; \
        result_64 ^= result_64 >> 8 ; \
        syn [ curP ] = ( unsigned char ) result_64 ; \
        if ( syn [ curP ] != 0 ) err = 1 ; \
} 


int gf_4vect_pss_avx512_gfni_2d(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest, int offSet)
{
        int curSym, curPos ;                          // Loop counters
        __mmask8 mask ;                               // Mask used to test for zero
        __m512i parity [ 4 ], taps [ 3 ] ;            // Parity registers
        __m512i data_vec ;
        unsigned char err = 0 ;
        unsigned char syn [ 32 ] ;
      __m512i matVec, vreg ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );

        for ( curPos = offSet ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Dec(data_vec, 4, err, syn ) ;
                if ( err != 0 )
                {
                    return curPos ;
                }
                parity [ 0 ] = data_vec ;
                parity [ 1 ] = data_vec ;
                parity [ 2 ] = data_vec ;
                parity [ 3 ] = data_vec ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        L1Dec( data_vec, 4, err, syn ) ;
                        if ( err != 0 )
                        {
                            return curPos ;
                        }

                        parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 0 ], taps [ 0 ], 0) ;
                        parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 1 ], taps [ 1 ], 0) ;
                        parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(parity [ 2 ], taps [ 2 ], 0) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 0 ], data_vec ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 1 ], data_vec ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 2 ], data_vec ) ;
                        parity [ 3 ] = _mm512_xor_si512 ( parity [ 3 ], data_vec ) ;
                }

                data_vec = _mm512_or_si512 ( parity [ 0 ], parity [ 1 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 2 ] ) ;
                data_vec = _mm512_or_si512 ( data_vec, parity [ 3 ] ) ;
                mask = _mm512_test_epi64_mask ( data_vec, data_vec ) ;
                if ( !_ktestz_mask8_u8 ( mask, mask ) )
                {
                        _mm512_store_si512( (__m512i *) &dest [ 0 ] [ 0 ], parity [ 0 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 1 ] [ 0 ], parity [ 1 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 2 ] [ 0 ], parity [ 2 ] ) ;
                        _mm512_store_si512( (__m512i *) &dest [ 3 ] [ 0 ], parity [ 3 ] ) ;
                        return ( curPos ) ;
                }
        }
        return ( curPos ) ;
}

int gf_4vect_pls_avx512_gfni_2d(int len, int k, unsigned char *g_tbls, unsigned char **data,
        unsigned char ** dest)
{
        int curSym, curPos ;                          // Loop counters
        __m512i parity [ 4 ], taps [ 4 ] ;          // Parity registers
        __m512i data_vec, par_vec ;
        unsigned char * pp = ( unsigned char * ) &par_vec + 64 - 4 ;
      __m512i matVec, vreg ;

        taps [ 0 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 0 * 8 ) ) );
        taps [ 1 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 1 * 8 ) ) );
        taps [ 2 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 2 * 8 ) ) );
        taps [ 3 ] = _mm512_broadcast_i32x2(*( __m128i * ) ( g_tbls + ( 3 * 8 ) ) );

        for ( curPos = 0 ; curPos < len ; curPos += 64 )
        {
                data_vec = _mm512_load_si512( (__m512i *) &data [ 0 ] [ curPos ] ) ;
              __builtin_prefetch ( &data [ 0 ] [ curPos + 64 ], 0, 3 ) ;
                L1Enc(data_vec, 4, par_vec ) ;
                parity [ 0 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0) ;
                parity [ 1 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0) ;
                parity [ 2 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0) ;
                parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;

                for ( curSym = 1 ; curSym < k ; curSym ++ )
                {
                        data_vec = _mm512_load_si512( (__m512i *) &data [ curSym ] [ curPos ] ) ;
                      __builtin_prefetch ( &data [ curSym ] [ curPos + 64 ], 0, 3 ) ;
                        data_vec = _mm512_xor_si512( data_vec, parity [ 0 ] ) ;
                        L1Enc(data_vec, 4, par_vec ) ;
                        parity [ 0 ] = _mm512_xor_si512 ( parity [ 1 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 0 ], 0 )  ) ;
                        parity [ 1 ] = _mm512_xor_si512 ( parity [ 2 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 1 ], 0 )  ) ;
                        parity [ 2 ] = _mm512_xor_si512 ( parity [ 3 ],
                                       _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 2 ], 0 )  ) ;
                        parity [ 3 ] = _mm512_gf2p8affine_epi64_epi8(data_vec, taps [ 3 ], 0) ;
                }

                _mm512_store_si512( (__m512i *) &dest [ 0 ] [ curPos ], parity [ 0 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 1 ] [ curPos ], parity [ 1 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 2 ] [ curPos ], parity [ 2 ] ) ;
                _mm512_store_si512( (__m512i *) &dest [ 3 ] [ curPos ], parity [ 3 ] ) ;
        }
        return ( curPos ) ;
}

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
void pc_bpow_1b ( unsigned char Gen )
{
    int i ;


    // A positive integer raised to the power 0 is one
    pc_ptab_1b [ 0 ] = 1 ;

    // Two is a good generator for 0x1d, three is a good generator for 0x1b
    for ( i = 1 ; i < 256 ; i ++ )
    {
        pc_ptab_1b[ i ] = pc_mul_1b ( pc_ptab_1b [ i - 1 ], Gen ) ;
    }
}

// pc_blog - Use the power table to build the log table
void pc_blog_1b ( void ) 
{
    int i ;


    // Use the power table to index into the log table and store log value
    for ( i = 0 ; i < 256 ; i ++ )
    {
        pc_ltab_1b [ pc_ptab_1b [ i ] ] = i ;
    }
}

// pc_linv - Calculate the inverse of a number, that is, 1/Number
void  pc_binv_1b ( void ) 
{
    int i ;
    for ( i = 0 ; i < 256 ; i ++ )
    {
        pc_itab_1b [ i ] = pc_ptab_1b [ 255 - pc_ltab_1b [ i ] ] ;
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
                gen = pc_mul_1b(gen, 3);
        }
}

// Initialize encoding matrix for encoding
void pc_bmat_1b ( unsigned char * vals, int p )
{
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        memcpy ( ( unsigned char * ) &EncMat [ curP ], &vals [ curP * ( 255 - p ) ], 255 - p ) ;
        unsigned char * extra = ( unsigned char * ) &EncMat [ curP ] ;
        memset ( extra + 255 - p, 0, p ) ;
    }
}

// Initialize vandermonde matrix for decoding
void pc_bvan_1b ( unsigned char * vals, int p )
{
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        memcpy (  ( unsigned char * ) &Vand1b [ curP ], &vals [ curP * ( 255 ) ], 255 ) ;
        unsigned char * extra = ( unsigned char * ) &Vand1b [ curP ] ;
        extra [ 255 ] = 0 ;
    }
}

// Encode using the Vandermonde matrix
void pc_encoder1b ( unsigned char * codeWord, unsigned char * par, int p ) 
{
    __m512i codeWordvec [ 4 ], encMatvec [ 4 ], vreg [ 4 ] ;

    // Load the entire codeword into 4 vector registers
    codeWordvec [ 0 ] = _mm512_loadu_si512 ( codeWord + 0 * 64 ) ;
    codeWordvec [ 1 ] = _mm512_loadu_si512 ( codeWord + 1 * 64 ) ;
    codeWordvec [ 2 ] = _mm512_loadu_si512 ( codeWord + 2 * 64 ) ;
    codeWordvec [ 3 ] = _mm512_loadu_si512 ( codeWord + 3 * 64 ) ;
    //printf ( "Codeword\n" ) ;
    //dump_u8xu8 ( ( unsigned char * ) &codeWordvec [ 0 ], 1, 255 ) ;

    // Now loop and compute each parity using the encoding matrix
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        //printf ( "Encmat\n" ) ;
        //dump_u8xu8 ( ( unsigned char * ) &EncMat [ curP ] [ 0 ], 1, 255 ) ;

        // Load one row of the encoding matrix into vector registers
        encMatvec [ 0 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 0 ] ) ;
        encMatvec [ 1 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 1 ] ) ;
        encMatvec [ 2 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 2 ] ) ;
        encMatvec [ 3 ] = _mm512_load_si512 ( &EncMat [ curP ] [ 3 ] ) ;

        // Multiply the codeword by the encoding matrix
        vreg [ 0 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 0 ], encMatvec [ 0 ] ) ;
        vreg [ 1 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 1 ], encMatvec [ 1 ] ) ;
        vreg [ 2 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 2 ], encMatvec [ 2 ] ) ;
        vreg [ 3 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 3 ], encMatvec [ 3 ] ) ;

        // Now collapse the 255 symbols down to 1
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 1 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 2 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 3 ] ) ;

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
        par [ curP ] = ( unsigned char ) result_64 ;
        //printf ( "Par [ %d ] = %d\n", curP, par [ curP ] ) ;
    }
}

// Encode using the Vandermonde matrix
void pc_decoder1b ( unsigned char * codeWord, unsigned char * syn, int p ) 
{
    __m512i codeWordvec [ 4 ], vanMatvec [ 4 ], vreg [ 4 ] ;

    // Load the whole codeword into vector registers
    codeWordvec [ 0 ] = _mm512_loadu_si512 ( codeWord + 0 * 64 ) ;
    codeWordvec [ 1 ] = _mm512_loadu_si512 ( codeWord + 1 * 64 ) ;
    codeWordvec [ 2 ] = _mm512_loadu_si512 ( codeWord + 2 * 64 ) ;
    codeWordvec [ 3 ] = _mm512_loadu_si512 ( codeWord + 3 * 64 ) ;
    //printf ( "Codeword LSB\n" ) ;
    //dump_u8xu8 ( ( unsigned char * ) &codeWordvec [ 0 ], 1, 255 ) ;

    // Loop through each decoding vector of Vandermonde
    for ( int curP = 0 ; curP < p ; curP ++ )
    {
        //printf ( "curP = %d\n", curP ) ;
        //printf ( "Vandermonde\n" ) ;
        //dump_u8xu8 ( ( unsigned char * ) &Vand1b [ curP ] [ 0 ], 1, 255 ) ;

        // Load an entire row from the Vandermonde matrix
        vanMatvec [ 0 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 0 ]) ;
        vanMatvec [ 1 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 1 ] ) ;
        vanMatvec [ 2 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 2 ] ) ;
        vanMatvec [ 3 ] = _mm512_load_si512 ( &Vand1b [ curP ] [ 3 ] ) ;

        // Multiply the codeword by the entire row
        vreg [ 0 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 0 ], vanMatvec [ 0 ] ) ;
        vreg [ 1 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 1 ], vanMatvec [ 1 ] ) ;
        vreg [ 2 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 2 ], vanMatvec [ 2 ] ) ;
        vreg [ 3 ] = _mm512_gf2p8mul_epi8 ( codeWordvec [ 3 ], vanMatvec [ 3 ] ) ;
        //printf ( "VReg\n" ) ;
        //dump_u8xu8 ( (unsigned char *) vreg, 1, 255 ) ;

        // Now collapse the 255 symbols down to 1
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 1 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 2 ] ) ;
        vreg [ 0 ] = _mm512_xor_si512 ( vreg [ 0 ], vreg [ 3 ] ) ;

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
        syn [ curP ] = ( unsigned char ) result_64 ;
    }
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

unsigned char gf_div_1b_AVX512_GFNI ( unsigned char a, unsigned char b )
{
        return pc_mul_1b ( a, pc_itab_1b [ b ] ) ;
}

// Compute base ^ Power
int pc_pow_1b_AVX512_GFNI ( unsigned char base, unsigned char Power )
{
        // The first power is always 1
        if ( Power == 0 )
        {
                return 1 ;
        }

        // Otherwise compute the power of two for Power
        unsigned char computedPow = base ;
        for ( int i = 1 ; i < Power ; i ++ )
        {
                computedPow = pc_mul_1b ( computedPow, base ) ;
        }
        return computedPow ;
}

// Assume there is a single error and try to correct, see if syndromes match
int pc_verify_single_error_1b_AVX512_GFNI ( unsigned char * S, unsigned char ** data, int k, int p,
        int newPos, int offSet )
{
        // LSB has parity, for single error this equals error value
        unsigned char eVal = S [ 0 ] ;

        // Compute error location is log2(syndrome[1]/syndrome[0])
        unsigned char eLoc = S [ 1 ] ;
        unsigned char pVal = gf_mul ( eLoc, pc_itab_1b [ eVal ] ) ;
        eLoc = pc_ltab_1b [ pVal ] ;

        // Verify error location is reasonable
        if ( eLoc >= k )
        {
                return 0 ;
        }

        // If more than 2 syndromes, verify we can produce them all
        if ( p > 2 )
        {
                // Now verify that the error can be used to produce the remaining syndromes
                for ( int i = 2 ; i < p ; i ++ )
                {
                        if ( pc_mul_1b ( S [ i - 1 ], pVal ) != S [ i ] )
                        {
                                return 0 ;
                        }
                }
        }
        // Good correction
        data [ k - eLoc - 1 ] [ newPos + offSet ] ^= eVal ;
        return 1 ;
}

int gf_invert_matrix_1b_AVX512_GFNI(unsigned char *in_mat, unsigned char *out_mat, const int n)
{
        __m512i affineVal512 ;
        __m128i affineVal128 ;

        if (n > 32) return -1; // Assumption: n <= 32

        int i, j;
        __m512i aug_rows[32] ;                                 // Ensure 64-byte alignment
        unsigned char *matrix_mem = (unsigned char *)aug_rows; // Point to aug_rows memory

        // Initialize augmented matrix: [in_mat row | out_mat row | padding zeros]
        for (i = 0; i < n; i++)
        {
                memcpy ( &matrix_mem [ i * 64 ], &in_mat [ i * n ], n ) ;
                memset ( &matrix_mem [ i * 64 + n ], 0, n ) ;
                matrix_mem [ i * 64 + n + i ] = 1 ;
                //dump_u8xu8 ( &matrix_mem [ i * 64 + n ], 1, n ) ;
        }

        // Inverse using Gaussian elimination
        for ( i = 0; i < n; i++ )
        {
                // Check for 0 in pivot element using matrix_mem
                unsigned char pivot = matrix_mem [ i * 64 + i ] ;
                //printf ( "Pivot = %d\n", pivot ) ;
                if ( pivot == 0 )
                {
                        // Find a row with non-zero in current column and swap
                        for ( j = i + 1; j < n; j++ )
                        {
                                if ( matrix_mem [ j * 64 + i ] != 0 )
                                {
                                        break ;
                                }
                        }
                        if ( j == n )
                        {
                                // Couldn't find means it's singular
                                return -1;
                        }
                        // Swap rows i and j in ZMM registers
                        __m512i temp_vec = aug_rows [ i ] ;
                        aug_rows[i] = aug_rows [ j ] ;
                        aug_rows[j] = temp_vec ;
                }

                // Get pivot and compute 1/pivot
                pivot = matrix_mem [ i * 64 + i ]  ;
                //printf ( "Pivot2 = %d\n", pivot ) ;
                unsigned char temp_scalar = pc_itab_1b [ pivot ] ;
                //printf ( "Scalar = %d\n", temp_scalar ) ;

                // Scale row i by 1/pivot using GFNI affine
                affineVal128 = _mm_set1_epi64x ( gf_table_gfni [ temp_scalar ] ) ;
                affineVal512 = _mm512_broadcast_i32x2 ( affineVal128 ) ;
                aug_rows [ i ]  = _mm512_gf2p8affine_epi64_epi8 ( aug_rows[ i ], affineVal512, 0 ) ;

                // Eliminate in other rows
                for ( j = 0; j < n; j++ )
                {
                        if ( j == i)  continue;
                        unsigned char factor = matrix_mem[j * 64 + i];
                        // Compute scaled pivot row: pivot_row * factor
                        affineVal128 = _mm_set1_epi64x ( gf_table_gfni [ factor ] ) ;
                        affineVal512 = _mm512_broadcast_i32x2 ( affineVal128 ) ;
                        __m512i scaled = _mm512_gf2p8affine_epi64_epi8 ( aug_rows [ i ], affineVal512, 0 ) ;
                        // row_j ^= scaled
                        aug_rows [ j ] = _mm512_xor_si512( aug_rows [ j ], scaled ) ;
                }
        }
        // Copy back to out_mat
        for ( i = 0; i < n; i++ )
        {
                //dump_u8xu8 ( &matrix_mem [ i * 64 + n ], 1, n ) ;
                memcpy ( &out_mat [ i * n ] , &matrix_mem [ i * 64 + n ], n )  ;
        }
        return 0 ;
}

int find_roots_1b_AVX512_GFNI(unsigned char *keyEq, unsigned char *roots, int mSize)
{
        static __m512i Vandermonde [ 16 ] [ 4 ] ;
        __m512i sum [ 4 ], temp, affineVal512 ;
        __m128i affineVal128 ;
        int i, j ;

        unsigned char * vVal = ( unsigned char *) Vandermonde ;
        // Check to see if Vandermonde has been initialized yet
        if ( vVal [ 0 ] == 0 )
        {
                unsigned char base = 2, cVal = 1 ;
                for (  i = 0 ; i < 16 ; i ++ )
                {
                        vVal = ( unsigned char * ) &Vandermonde [ i ] ;
                        for ( j = 0 ; j < 255 ; j ++ )
                        {
                                vVal [ j ] = cVal ;
                                cVal = gf_mul ( cVal, base ) ;
                        }
                        base = pc_mul_1b ( base, 3 ) ;
                }
        }
        // Initialize our sum to the constant term, no need for multiply
        sum [ 0 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;
        sum [ 1 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;
        sum [ 2 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;
        sum [ 3 ] = _mm512_set1_epi8 ( keyEq [ 0 ] ) ;

        // Loop through each keyEq value, multiply it by Vandermonde and add it to sum
        for ( i = 1 ; i < mSize ; i ++ )
        {
                affineVal128 = _mm_set1_epi64x ( gf_table_gfni [ keyEq [ i ] ] ) ;
                affineVal512 = _mm512_broadcast_i32x2 ( affineVal128 ) ;
                // Remember that we did not build the first row of Vandermonde, so use i-1
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 0 ], affineVal512, 0 ) ;
                sum [ 0 ] = _mm512_xor_si512 ( sum [ 0 ], temp ) ;
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 1 ], affineVal512, 0 ) ;
                sum [ 1 ] = _mm512_xor_si512 ( sum [ 1 ], temp ) ;
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 2 ], affineVal512, 0 ) ;
                sum [ 2 ] = _mm512_xor_si512 ( sum [ 2 ], temp ) ;
                temp = _mm512_gf2p8affine_epi64_epi8 ( Vandermonde [ i-1 ] [ 3 ], affineVal512, 0 ) ;
                sum [ 3 ] = _mm512_xor_si512 ( sum [ 3 ], temp ) ;
        }
        // Add in the leading Vandermonde row, just assume it's a one so no multiply
        sum [ 0 ] = _mm512_xor_si512 ( sum [ 0 ], Vandermonde [ mSize - 1 ] [ 0 ] ) ;
        sum [ 1 ] = _mm512_xor_si512 ( sum [ 1 ], Vandermonde [ mSize - 1 ] [ 1 ] ) ;
        sum [ 2 ] = _mm512_xor_si512 ( sum [ 2 ], Vandermonde [ mSize - 1 ] [ 2 ] ) ;
        sum [ 3 ] = _mm512_xor_si512 ( sum [ 3 ], Vandermonde [ mSize - 1 ] [ 3 ] ) ;

        int rootCount = 0, idx = 0 ;
        // Create the list of roots
        for ( i = 0 ; i < 4 ; i ++ )
        {
                // Compare each byte to zero, generating a 64-bit mask
                __mmask64 mask = _mm512_cmpeq_epi8_mask( sum [ i ], _mm512_setzero_si512());

                // Count number of zeros (popcount of mask)
                rootCount += _mm_popcnt_u64(mask);

                // Extract indices of set bits (zero bytes)
                while ( mask )
                {
                        // Find the next set bit (index of zero byte)
                        uint64_t pos = _tzcnt_u64(mask);
                        roots[idx++] = (uint8_t) pos + ( i * 64 ) ;
                        // Clear the lowest set bit
                        mask = _blsr_u64 ( mask ) ; // mask &= (mask - 1)
                }
        }
        return rootCount ;
}

// Compute error values using Vandermonde
int pc_compute_error_values_1b_AVX512_GFNI ( int mSize, unsigned char * S, unsigned char * roots,
        unsigned char * errVal )
{
        int i, j ;
        unsigned char Mat [ PC_MAX_ERRS * PC_MAX_ERRS ] ;
        unsigned char Mat_inv [ PC_MAX_ERRS * PC_MAX_ERRS ] ;

        // Find error values by building and inverting Vandemonde
        for ( i = 0 ; i < mSize ; i ++ )
        {
                Mat [ i ] = 1 ;
        }
        unsigned char base = 2 ;
        for ( i = 1 ; i < mSize ; i ++ )
        {
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        Mat [ i * mSize + j ] = pc_pow_1b_AVX512_GFNI ( base, roots [ j ] ) ;
                }
                base = gf_mul ( base, 2 ) ;
        }
        // Invert matrix and verify inversion
        if ( gf_invert_matrix_1b_AVX512_GFNI ( Mat, Mat_inv, mSize ) != 0 )
        {
                return 0 ;
        }

        // Compute error values by summing Syndrome terms across inverted Vandermonde
        for ( i = 0 ; i < mSize ; i ++ )
        {
                errVal [ i ] = 0 ;
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        errVal [ i ] ^= pc_mul_1b ( S [ j ], Mat_inv [ i * mSize + j ] ) ;
                }
        }
        return 1 ;
}

// Verify proposed data values and locations can generate syndromes
int pc_verify_syndromes_1b_AVX512_GFNI ( unsigned char * S, int p, int mSize, unsigned char * roots,
        unsigned char * errVal )
{
        int i,j ;
        unsigned char sum = 0 ;

        // Verify syndromes across each power row
        unsigned char base = 1 ;
        for ( i = 0 ; i < p ; i++ )
        {
                sum = 0 ;
                for ( j = 0 ; j < mSize ; j ++ )
                {
                        // Scale up the data value based on location
                        unsigned char termVal = pc_mul_1b ( errVal [ j ], pc_pow_1b_AVX512_GFNI ( base, roots [ j ] ) ) ;
                        sum ^= termVal ;
                }

                // Verify we reproduced the syndrome
                if ( sum != S [ i ] )
                {
                        return 0 ;
                }
                // Move to next syndrome
                base = pc_mul_1b ( base, 3 ) ;
        }
        return 1 ;
}

// Affine table from ec_base.h: 256 * 8-byte matrices for GF(256) multiplication
static const uint64_t gf_table_gfni[256];  // Assume defined in ec_base.h

// syndromes: array of length 'length' (typically 2t), syndromes[0] = S1, [1] = S2, etc.
// lambda: caller-allocated array of size at least (length + 1 + 31), filled with locator poly coeffs. Padded for SIMD.
// Returns: degree L of the error locator polynomial.
// Note: Assumes length <= 32 for AVX-512 (32-byte vectors); extend loops for larger lengths.
int berlekamp_massey_1b_AVX512_GFNI(unsigned char *syndromes, int length, unsigned char *lambda)
{
    unsigned char b[PC_MAX_ERRS*2 + 1];  // Padded for AVX-512 (32-byte alignment)
    unsigned char temp[PC_MAX_ERRS*2 + 1];
    int L = 0;
    int m = 1;
    unsigned char old_d = 1;  // Initial previous discrepancy

    memset(lambda, 0, length + 1 + 31);
    lambda[0] = 1;
    memset(b, 0, length + 1 + 31);
    b[0] = 1;

    for (int r = 0; r < length; r++)
    {
        unsigned char d = syndromes[r];
        for (int j = 1; j <= L; j++)
        {
            if (r - j >= 0) {
                d ^= pc_mul_1b(lambda[j], syndromes[r - j]);
            }
        }

        if (d == 0)
        {
            m++;
        }
        else
        {
            unsigned char q = gf_div_AVX512_GFNI(d, old_d);
            memcpy(temp, lambda, length + 1 + 31);

            // SIMD update: lambda[j + m] ^= gf_mul(q, b[j]) using AVX-512 GF2P8AFFINE
            // Load and broadcast 8-byte affine matrix for q
            __m128i matrix_128 = _mm_set1_epi64x(gf_table_gfni[q]);  // Load uint64_t from gf_table_gfni[q]
            __m256i matrix = _mm256_broadcast_i32x2(matrix_128);  // Broadcast to all 4 lanes
            __m256i b_vec = _mm256_loadu_si256((const __m256i *)b);
            // Perform GF(256) multiplication: result = affine(b_vec, matrix) + 0
            __m256i mul_res = _mm256_gf2p8affine_epi64_epi8(b_vec, matrix, 0);
            __m256i vec_lam = _mm256_loadu_si256((const __m256i *)&lambda[m]);
            vec_lam = _mm256_xor_si256(vec_lam, mul_res);
            _mm256_storeu_si256((__m256i *)&lambda[m], vec_lam);

            if (2 * L <= r)
            {
                L = r + 1 - L;
                memcpy(b, temp, length + 1 + 31);
                old_d = d;
                m = 1;
            }
            else
            {
                m++;
            }
        }
    }

    return L;
}

// Attempt to detect multiple error locations and values
int pc_verify_multiple_errors_1b_AVX512_GFNI ( unsigned char * S, unsigned char ** data, int mSize, int k,
        int p, int newPos, int offSet, unsigned char * keyEq )
{
        unsigned char roots [ PC_MAX_ERRS ] = {0} ;
        unsigned char errVal [ PC_MAX_ERRS ] ;

        // Find roots, exit if mismatch with expected roots
        int nroots = find_roots_AVX512_GFNI ( keyEq, roots, mSize );
        if ( nroots != mSize )
        {
                printf ( "Bad roots expected %d got %d\n", mSize, nroots ) ;
                return 0 ;
        }

        // Compute the error values
        if ( pc_compute_error_values_AVX512_GFNI ( mSize, S, roots, errVal ) == 0 )
        {
                return 0 ;
        }

        // Verify all syndromes are correct
        if ( pc_verify_syndromes_AVX512_GFNI ( S, p, mSize, roots, errVal ) == 0 )
        {
                return 0 ;
        }

        // Syndromes are OK, correct the user data
        for ( int i = 0 ; i < mSize ; i ++ )
        {
                int sym = k - roots [ i ] - 1 ;
                data [ sym ] [ newPos + offSet ] ^= errVal [ i ] ;
        }
        // Good correction
        return 1 ;
}

// PGZ decoding step 1, see if we can invert the matrix, if so, compute key equation
int PGZ_1b_AVX512_GFNI ( unsigned char * S, int p, unsigned char * keyEq )
{
       unsigned char SMat [ PC_MAX_ERRS * PC_MAX_ERRS ], SMat_inv [ PC_MAX_ERRS * PC_MAX_ERRS ] ;
        int i,j ;

       // For each potential size, create and find Hankel matrix that will invert
        for ( int mSize = ( p / 2 ) ; mSize >= 2 ; mSize -- )
        {
                for ( i = 0 ; i < mSize ; i ++ )
                {
                        for ( j = 0 ; j < mSize ; j ++ )
                        {
                                SMat [ i * mSize + j ] = S [ i + j ] ;
                        }
                }
                // If good inversion then we know error count and can compute key equation
                if ( gf_invert_matrix_AVX512_GFNI ( SMat, SMat_inv, mSize ) == 0 )
                {
                        // Compute the key equation terms
                        for ( i = 0 ; i < mSize ; i ++ )
                        {
                                for ( j = 0 ; j < mSize ; j ++ )
                                {
                                        keyEq [ i ] ^= gf_mul ( S [ mSize + j ], SMat_inv [ i * mSize + j ] ) ;
                                }
                        }
                        return mSize ;
                }
        }
        return 0 ;

}

// Syndromes are non-zero, try to calculate error location and data values
int pc_correct_1b_AVX512_GFNI ( int newPos, int k, int p,
    unsigned char ** data, unsigned char ** coding, int vLen )
{
        int i, mSize  ;
        unsigned char S [ PC_MAX_ERRS ], keyEq [ PC_MAX_ERRS + 1 ] = { 0 } ;

        __m512i vec, vec2 ;

        // Get a "or" of all the syndrome vectors
        vec = _mm512_load_si512( ( __m512i * )  coding [ 0 ] ) ;
        for ( i = 0 ; i < p ; i ++ )
        {
                vec2 = _mm512_load_si512 ( ( __m512i * ) coding [ i ] ) ;
                vec = _mm512_or_si512 ( vec,  vec2 ) ;
        }

        // Now find the first non-zero byte
        __mmask64 mask = _mm512_test_epi8_mask( vec, vec ) ;
        uint64_t offSet = _tzcnt_u64(mask);

        // Verify we found a non-zero syndrome
        if ( offSet >= vLen )
        {
                return 0 ;
        }

        // Gather up the syndromes
        for ( i = 0 ; i < p ; i ++ )
        {
                S [ i ] = coding [ p - i - 1 ] [ offSet ] ;
        }

        // Check to see if a single error can be verified
        if ( pc_verify_single_error_AVX512_GFNI ( S, data, k, p, newPos, offSet ) )
        {
                return 1 ;
        }

        mSize = PGZ_AVX512_GFNI ( S, p, keyEq ) ;

        if ( mSize > 1 )
        {
                return pc_verify_multiple_errors_AVX512_GFNI ( S, data, mSize, k, p, newPos, offSet, keyEq ) ;
        }
        return 0 ;
}

