void prepackA_6x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    int x_len = kmax - k0;
    uint32_t zerobuff[x_len];
    memset(zerobuff, 0, sizeof(uint32_t) * x_len);

    uint32_t *dout = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    uint32_t* outptr = dout;

    //! data A is not transposed, transpose A to k * 6
    for (int y = m0; y < mmax; y += 6) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;

        int x = x_len;
        //! cope with row index exceed real size, set to zero buffer
        if ((y + 5) >= mmax) {
            switch ((y + 5) - mmax) {
                case 4:
                    inptr1 = zerobuff;
                case 3:
                    inptr2 = zerobuff;
                case 2:
                    inptr3 = zerobuff;
                case 1:
                    inptr4 = zerobuff;
                case 0:
                    inptr5 = zerobuff;
                default:
                    break;
            }
        }

        for (; x > 7; x -= 8) {

            //! zip load 8 elements (2 neon Q registers) from each of 6 rows
            asm volatile (
            "vld4.32  {d0-d3}, [%[inptr0]]!   @ zip load r0, q0,q1=r00,r04,r01,r05,r02,r06,r03,r07\n"
                    "vld4.32  {d4-d7}, [%[inptr1]]!   @ zip load r1, q2,q3=r10,r14,r11,r15,r12,r16,r13,r17\n"
                    "vld4.32  {d8-d11}, [%[inptr2]]!  @ zip load r2, q4,q5=r20,r24,r21,r25,r22,r26,r23,r27\n"
                    "vld4.32  {d12-d15}, [%[inptr3]]! @ zip load r3, q6,q7=r30,r34,r31,r35,r32,r36,r33,r37\n"
                    "vld4.32  {d16-d19}, [%[inptr4]]! @ zip load r4, q8,q9=r40,r44,r41,r45,r42,r46,r43,r47\n"
                    "vld4.32  {d20-d23}, [%[inptr5]]! @ zip load r5, q10,q11=r50,r54,r51,r55,r52,r56,r53,r57\n"

                    "vtrn.32  q0, q2                  @ trans data: q0=r00,r10,r01,r11; q2=r04,r14,r05,r15\n"
                    "vtrn.32  q4, q6                  @ trans data: q4=r20,r30,r21,r31; q6=r24,r34,r25,r35\n"
                    "vtrn.32  q8, q10                 @ trans data: q8=r40,r50,r41,r51; q10=r44,r54,r45,r55\n"

                    "vswp     d1, d8                  @ swap d1, d8, q0=r00,r10,r20,r30; q4=r01,r11,r21,r31\n"
                    "vst1.32  {d0-d1},  [%[outptr]]!  @ write q0:r00,r10,r20,r30\n"
                    "vst1.32  {d16},    [%[outptr]]!  @ write d16(q8,low),r40,r50\n"
                    "vst1.32  {d8-d9},  [%[outptr]]!  @ write q4:r01,r11,r21,r31\n"
                    "vst1.32  {d17},    [%[outptr]]!  @ write d16(q8,high),r41,r51\n"

                    "vtrn.32  q1, q3                  @ trans data: q1=r02,r12,r03,r13; q3=r06,r16,r07,r17\n"
                    "vtrn.32  q5, q7                  @ trans data: q5=r22,r32,r23,r33; q7=r26,r36,r27,r37\n"
                    "vtrn.32  q9, q11                 @ trans data: q9=r42,r52,r43,r53; q11=r46,r56,r47,r57\n"

                    "vswp     d3, d10                 @ swap d3, d10, q1=r02,r12,r22,r32; q5=r03,r13,r23,r33\n"
                    "vst1.32  {d2-d3},  [%[outptr]]!  @ write q1:r02,r12,r22,r32\n"
                    "vst1.32  {d18},    [%[outptr]]!  @ write d18(q9,low),r42,r52\n"
                    "vst1.32  {d10-d11},[%[outptr]]!  @ write q5:r03,r13,r23,r33\n"
                    "vst1.32  {d19},    [%[outptr]]!  @ write d19(q9,high),r43,r53\n"

                    "vswp     d5, d12                 @ swap d5, d12,q2=r04,r14,r24,r34; q6=r05,r15,r25,r35\n"
                    "vst1.32  {d4-d5},  [%[outptr]]!  @ write q2:r04,r14,r24,r34\n"
                    "vst1.32  {d20},    [%[outptr]]!  @ write d20(q10,low),r44,r54\n"
                    "vst1.32  {d12-d13},[%[outptr]]!  @ write q6:r05,r15,r25,r35\n"
                    "vst1.32  {d21},    [%[outptr]]!  @ write d21(q10,high),r45,r55\n"

                    "vswp     d7, d14                 @ swap d7, d14, q3=r06,r16,r26,r36; q7=r07,r17,r27,r37\n"
                    "vst1.32  {d6-d7},  [%[outptr]]!  @ write q3:r06,r16,r26,r36\n"
                    "vst1.32  {d22},    [%[outptr]]!  @ write d22(q11,low),r46,r56\n"
                    "vst1.32  {d14-d15},[%[outptr]]!  @ write q7:r07,r17,r27,r37\n"
                    "vst1.32  {d23},    [%[outptr]]!  @ write d23(q11,high),r47,r57\n"
            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), \
                [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), \
                [outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "cc", "memory"
            );
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
        }
    }
}

void prepackA_trans_6x8(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax) {

    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + m0;

    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int x_len = mmax - m0;
    int y_len = kmax - k0;
    int right_remain = x_len - 6 * (x_len / 6);
    int right_pad = 6 - right_remain;
    if (right_remain == 0) {
        right_pad = 0;
    }

    uint32_t *outptr_row = outptr;
    int stride_out = 6 * y_len;

    uint32x4_t vzero = vdupq_n_u32(0);
    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

#pragma omp parallel for
    for (int y = 0; y < y_len - 3; y += 4) {

        const uint32_t* ptr0 = inptr + y * ldin;
        const uint32_t* ptr1 = ptr0 + ldin;
        const uint32_t* ptr2 = ptr1 + ldin;
        const uint32_t* ptr3 = ptr2 + ldin;

        uint32_t *outptr_row_col = outptr_row + y * 6;
        int i = 0;
        for (; i < x_len - 5; i += 6) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr1]]!        @ load r1, 6 elements\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d0-d2}, [%[ptr2]]!        @ load r2, 6 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr3]]!        @ load r3, 6 elements\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr1]]!        @ load r1, 6 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d6, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"

                    "vld1.32 {d0-d2}, [%[ptr2]]!        @ load r2, 8 elements\n"
                    "vld1.32 {d4-d6}, [%[ptr3]]!        @ load r3, 8 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vbif   q2, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d6, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
                    "vst1.32 {d4-d6}, [%[outptr]]!      @ write to output ptr\n"
            : [outptr] "+r" (ptr_out), [ptr0] "+r" (ptr0), [ptr1] "+r" (ptr1),
            [ptr2] "+r" (ptr2), [ptr3] "+r" (ptr3)
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "q2", "q3", "cc", "memory"
            );
        }
    }

#pragma omp parallel for
    for (int y = 4 * (y_len / 4); y < y_len; ++y) {

        const uint32_t* ptr0 = inptr + y * ldin;
        uint32_t *outptr_row_col = outptr_row + y * 6;
        int i = 0;
        for (; i < x_len - 5; i += 6) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            :
            : "q0", "q1", "cc", "memory"
            );
            outptr_row_col += stride_out;
        }
        if (right_pad > 0) {
            uint32_t *ptr_out = outptr_row_col;
            asm volatile(
            "vld1.32 {d0-d2}, [%[ptr0]]!        @ load r0, 6 elements\n"
                    "vbif   q0, %q[vzero], %q[vmask1]   @ bit select, pad zero\n"
                    "vbif   d2, %e[vzero], %e[vmask2]   @ bit select, pad zero\n"
                    "vst1.32 {d0-d2}, [%[outptr]]!      @ write to output ptr\n"
            : [ptr0] "+r" (ptr0), [outptr] "+r" (ptr_out)
            : [vmask1] "w" (vmask1), [vmask2] "w" (vmask2), \
              [vzero] "w" (vzero)
            : "q0", "q1", "cc", "memory"
            );
        }
    }
}
