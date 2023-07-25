#include <stdio.h>
#include <string.h>

#include <immintrin.h>
#include<math.h>

#include "params.h"
#include "matrixmod.h"

/*reduction modulo operation(after multiplication/ addition might exceed number of bits it can hold)
find methods for SIMD might be more or less efficient
Gaussian or Multipilcation, which take the most time? Maximise the one which took the most time
turn off frequency scaling?*/

void pmod_mat_print(pmod_mat_t *M, int M_r, int M_c) //prime modulo matrix
{
  pmod_mat_fprint(stdout, M, M_r, M_c);
}

void pmod_mat_fprint(FILE *stream, pmod_mat_t *M, int M_r, int M_c)
{
  for (int r = 0; r < M_r; r++)
  {
    fprintf(stream, "[");
    for (int c = 0; c < M_c-1; c++)
      fprintf(stream, "%4i ", pmod_mat_entry(M, M_r, M_c, r, c));
    fprintf(stream, "%4i", pmod_mat_entry(M, M_r, M_c, r, M_c-1));
    fprintf(stream, "]\n");
  }
}

void pmod_mat_mul(pmod_mat_t *C, int C_r, int C_c, pmod_mat_t *A, int A_r, int A_c, pmod_mat_t *B, int B_r, int B_c)
{
  // handle not multiple of 8
  int num_of_instructions = ceil(C_r / 8); //multiple of 8
  int remainder = C_r % 8;
  int mask_arr[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

  if (remainder != 0)
  {
    for (int i = 0; i < 8 - remainder; i++)
    {
      mask_arr[remainder + i] = 1;
    }
  }

  __m256i mask = _mm256_loadu_si256((const __m256i *)mask_arr);

  for (int c = 0; c < C_c; c++)
  {
    for (int r = 0; r < C_r; r++)
    {
      uint64_t val = 0;
      __m256i vec_c = _mm256_setzero_si256();

      for (int i = 0; i < num_of_instructions; i++)
      {
        __m256i vec_a = _mm256_setr_epi32(
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 0+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 1+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 2+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 3+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 4+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 5+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 6+i*8),
          (uint64_t) pmod_mat_entry(A, A_r, A_c, r, 7+i*8)
        );
        __m256i vec_b = _mm256_setr_epi32(
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 0+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 1+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 2+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 3+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 4+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 5+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 6+i*8, c),
          (uint64_t) pmod_mat_entry(B, B_r, B_c, 7+i*8, c)
        );

        if (remainder != 0 && i == num_of_instructions - 1)
        {
          uint64_t arr_vec_a[8] = {
          pmod_mat_entry(A, A_r, A_c, r, 0+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 1+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 2+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 3+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 4+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 5+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 6+i*8),
          pmod_mat_entry(A, A_r, A_c, r, 7+i*8)
          };
          uint64_t arr_vec_b[8] = {
          pmod_mat_entry(B, B_r, B_c, 0+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 1+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 2+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 3+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 4+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 5+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 6+i*8, c),
          pmod_mat_entry(B, B_r, B_c, 7+i*8, c)
          };

          __m256i vec_a = _mm256_maskload_epi32((const int *)arr_vec_a, mask);
          __m256i vec_b = _mm256_maskload_epi32((const int *)arr_vec_b, mask);

          __m256i vec_tmp = _mm256_mullo_epi32(vec_a, vec_b);

          vec_c = _mm256_add_epi32(vec_c, vec_tmp);
          int *ptr = (int *)&vec_c;

          for (int i = 0; i < 8; i++)
          {
            val += ptr[i];
          }

          val = val % MEDS_p;

          pmod_mat_set_entry(C, C_r, C_c, r, c, val);

          continue;
        }
        __m256i vec_tmp = _mm256_mullo_epi32(vec_a, vec_b);

        vec_c = _mm256_add_epi32(vec_c, vec_tmp);
      }
      
      int *ptr = (int *)&vec_c;

      for (int i = 0; i < 8; i++)
      {
        val += ptr[i];
      }

      val = val % MEDS_p;

      pmod_mat_set_entry(C, C_r, C_c, r, c, val);
    }
  }
}
//   GFq_t tmp[C_r*C_c]; // uint16_t tmp[C_r*C_c]

//   for (int c = 0; c < C_c; c++)
//     for (int r = 0; r < C_r; r++)
//     {
//       uint64_t val = 0;

//       for (int i = 0; i < A_r; i++)
//         val = (val + (uint64_t)pmod_mat_entry(A, A_r, A_c, r, i) * (uint64_t)pmod_mat_entry(B, B_r, B_c, i, c));

//       tmp[r*C_c + c] = val % MEDS_p; // modulo operation after multiplication
//     }

//   for (int c = 0; c < C_c; c++)
//     for (int r = 0; r < C_r; r++)
//       pmod_mat_set_entry(C, C_r, C_c, r, c, tmp[r*C_c + c]);
// }

int pmod_mat_syst_ct(pmod_mat_t *M, int M_r, int M_c) //systematc form
{
  if (pmod_mat_row_echelon_ct(M, M_r, M_c) < 0)
    return -1;

  return pmod_mat_back_substitution_ct(M, M_r, M_c);
}

int pmod_mat_row_echelon_ct(pmod_mat_t *M, int M_r, int M_c)
{
  for (int r = 0; r < M_r; r++)
  {
    // swap
    for (int r2 = r+1; r2 < M_r; r2++)
    {
      uint64_t Mrr = pmod_mat_entry(M, M_r, M_c, r, r);

      for (int c = r; c < M_c; c++)
      {
        uint64_t val = pmod_mat_entry(M, M_r, M_c, r2, c);

        uint64_t Mrc = pmod_mat_entry(M, M_r, M_c, r, c);

        pmod_mat_set_entry(M, M_r, M_c, r, c, (Mrc + val * (Mrr == 0)) % MEDS_p);
      }
    }

    uint64_t val = pmod_mat_entry(M, M_r, M_c, r, r);

    if (val == 0)
      return -1;

    val = GF_inv(val);

    // normalize
    for (int c = r; c < M_c; c++)
    {
      uint64_t tmp = ((uint64_t)pmod_mat_entry(M, M_r, M_c, r, c) * val) % MEDS_p;
      pmod_mat_set_entry(M, M_r, M_c, r, c, tmp);
    }

    // eliminate
    for (int r2 = r+1; r2 < M_r; r2++)
    {
      uint64_t factor = pmod_mat_entry(M, M_r, M_c, r2, r);

      for (int c = r; c < M_c; c++)
      {
        uint64_t tmp0 = pmod_mat_entry(M, M_r, M_c, r, c);
        uint64_t tmp1 = pmod_mat_entry(M, M_r, M_c, r2, c);

        int64_t val = (tmp0 * factor) % MEDS_p;

        val = tmp1 - val;

        val += MEDS_p * (val < 0);

        pmod_mat_set_entry(M, M_r, M_c,  r2, c, val);
      }
    }
  }

  return 0;
}

int pmod_mat_back_substitution_ct(pmod_mat_t *M, int M_r, int M_c)
{
  // back substitution
  for (int r = M_r - 1; r >= 0; r--)
    for (int r2 = 0; r2 < r; r2++)
    {
      uint64_t factor = pmod_mat_entry(M, M_r, M_c, r2, r);

      uint64_t tmp0 = pmod_mat_entry(M, M_r, M_c, r, r);
      uint64_t tmp1 = pmod_mat_entry(M, M_r, M_c, r2, r);

      int64_t val = (tmp0 * factor) % MEDS_p;

      val = tmp1 - val;

      val += MEDS_p * (val < 0);

      pmod_mat_set_entry(M, M_r, M_c,  r2, r, val);

      for (int c = M_r; c < M_c; c++)
      {
        uint64_t tmp0 = pmod_mat_entry(M, M_r, M_c, r, c);
        uint64_t tmp1 = pmod_mat_entry(M, M_r, M_c, r2, c);

        int val = (tmp0 * factor) % MEDS_p;

        val = tmp1 - val;

        val += MEDS_p * (val < 0);

        pmod_mat_set_entry(M, M_r, M_c,  r2, c, val);
      }
    }

  return 0;
}

GFq_t GF_inv(GFq_t val)
{
  if (MEDS_p == 8191)
  {
    // Use optimal addition chain...
    uint64_t tmp_0  = val;
    uint64_t tmp_1  = (tmp_0 * tmp_0) % MEDS_p;
    uint64_t tmp_2  = (tmp_1 * tmp_0) % MEDS_p;
    uint64_t tmp_3  = (tmp_2 * tmp_1) % MEDS_p;
    uint64_t tmp_4  = (tmp_3 * tmp_3) % MEDS_p;
    uint64_t tmp_5  = (tmp_4 * tmp_3) % MEDS_p;
    uint64_t tmp_6  = (tmp_5 * tmp_5) % MEDS_p;
    uint64_t tmp_7  = (tmp_6 * tmp_6) % MEDS_p;
    uint64_t tmp_8  = (tmp_7 * tmp_7) % MEDS_p;
    uint64_t tmp_9  = (tmp_8 * tmp_8) % MEDS_p;
    uint64_t tmp_10 = (tmp_9 * tmp_5) % MEDS_p;
    uint64_t tmp_11 = (tmp_10 * tmp_10) % MEDS_p;
    uint64_t tmp_12 = (tmp_11 * tmp_11) % MEDS_p;
    uint64_t tmp_13 = (tmp_12 * tmp_2) % MEDS_p;
    uint64_t tmp_14 = (tmp_13 * tmp_13) % MEDS_p;
    uint64_t tmp_15 = (tmp_14 * tmp_14) % MEDS_p;
    uint64_t tmp_16 = (tmp_15 * tmp_15) % MEDS_p;
    uint64_t tmp_17 = (tmp_16 * tmp_3) % MEDS_p;

    return tmp_17;
  }
  else
  {
    uint64_t exponent = MEDS_p - 2;
    uint64_t t = 1;

    while (exponent > 0)
    {
      if ((exponent & 1) != 0)
        t = (t*(uint64_t)val) % MEDS_p;

      val = ((uint64_t)val*(uint64_t)val) % MEDS_p;

      exponent >>= 1;
    }

    return t;
  }
}

int pmod_mat_inv(pmod_mat_t *B, pmod_mat_t *A, int A_r, int A_c)
{
  pmod_mat_t M[A_r * A_c*2];

  for (int r = 0; r < A_r; r++)
  {
    memcpy(&M[r * A_c*2], &A[r * A_c], A_c * sizeof(GFq_t));

    for (int c = 0; c < A_c; c++)
      pmod_mat_set_entry(M, A_r, A_c*2, r, A_c + c, r==c ? 1 : 0);
  }

  int ret = pmod_mat_syst_ct(M, A_r, A_c*2);

  if ((ret == 0) && B)
    for (int r = 0; r < A_r; r++)
      memcpy(&B[r * A_c], &M[r * A_c*2 + A_c], A_c * sizeof(GFq_t));

  return ret;
}

