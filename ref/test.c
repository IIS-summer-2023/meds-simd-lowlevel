#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "randombytes.h"

#include "params.h"
#include "api.h"
#include "meds.h"

double osfreq(void);

long long cpucycles(void)
{
  unsigned long long result;
  asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
      : "=a" (result) ::  "%rdx");
  return result;
}

int main(int argc, char *argv[])
{
  printf("paramter set: %s\n\n", MEDS_name);

  long long time = 0;
  long long keygen_time_simd = 0xfffffffffffffff;
  long long sign_time_simd = 0xfffffffffffffff;
  long long verify_time_simd = 0xfffffffffffffff;

  double freq = osfreq();

  int rounds = 1;

  if (argc > 1)
    rounds = atoi(argv[1]);

  unsigned char entropy_input[48] = {0};

  randombytes_init(entropy_input, NULL, 256);

  char msg[4] = "Test";

  printf("pk:  %i bytes\n", CRYPTO_PUBLICKEYBYTES);
  printf("sk:  %i bytes\n", CRYPTO_SECRETKEYBYTES);
  printf("sig: %i bytes\n", CRYPTO_BYTES);
  printf("\n");

  printf("-----Optimized version-----\n");
  //simd
  for (int round = 0; round < rounds; round++)
  {
    uint8_t sk[CRYPTO_SECRETKEYBYTES] = {0};
    uint8_t pk[CRYPTO_PUBLICKEYBYTES] = {0};

    time = -cpucycles();
    crypto_sign_keypair_simd(pk, sk);
    time += cpucycles();

    if (time < keygen_time_simd) keygen_time_simd = time;

    uint8_t sig[CRYPTO_BYTES + sizeof(msg)] = {0};
    unsigned long long sig_len = sizeof(sig);

    time = -cpucycles();
    crypto_sign(sig, &sig_len, (const unsigned char *)msg, sizeof(msg), sk);
    time += cpucycles();

    if (time < sign_time_simd) sign_time_simd = time;

    unsigned char msg_out[4];
    unsigned long long msg_out_len = sizeof(msg_out);


    time = -cpucycles();
    int ret = crypto_sign_open(msg_out, &msg_out_len, sig, sizeof(sig), pk);
    time += cpucycles();

    if (time < verify_time_simd) verify_time_simd = time;

    printf("\n");
    if (ret == 0)
      printf("(Optimized) success\n");
    else
      printf("!!! (Optimized) FAILED !!!\n");

    printf("\n");
    printf("(Optimized) Time (min of %i runs):\n", rounds);
    printf("(Optimized) total_keygen: %f   (%llu cycles)\n", keygen_time_simd / freq, keygen_time_simd);
    printf("(Optimized) total_sign:   %f   (%llu cycles)\n", sign_time_simd / freq, sign_time_simd);
    printf("(Optimized) total_verify: %f   (%llu cycles)\n", verify_time_simd / freq, verify_time_simd);
    printf("\n");
  }

  printf("-----Unoptimized version-----\n");

  //original speed
  long long keygen_time = 0xfffffffffffffff;
  long long sign_time = 0xfffffffffffffff;
  long long verify_time = 0xfffffffffffffff;
  for (int round = 0; round < rounds; round++)
  {
    uint8_t sk[CRYPTO_SECRETKEYBYTES] = {0};
    uint8_t pk[CRYPTO_PUBLICKEYBYTES] = {0};

    time = -cpucycles();
    crypto_sign_keypair(pk, sk);
    time += cpucycles();

    if (time < keygen_time) keygen_time = time;

    uint8_t sig[CRYPTO_BYTES + sizeof(msg)] = {0};
    unsigned long long sig_len = sizeof(sig);

    time = -cpucycles();
    crypto_sign(sig, &sig_len, (const unsigned char *)msg, sizeof(msg), sk);
    time += cpucycles();

    if (time < sign_time) sign_time = time;

    unsigned char msg_out[4];
    unsigned long long msg_out_len = sizeof(msg_out);


    time = -cpucycles();
    int ret = crypto_sign_open(msg_out, &msg_out_len, sig, sizeof(sig), pk);
    time += cpucycles();

    if (time < verify_time) verify_time = time;

    printf("\n");
    if (ret == 0)
      printf("(Unoptimized) success\n");
    else
      printf("!!!(Unoptimized) FAILED !!!\n");
  }

  printf("\n");
  printf("(Unoptimized) Time (min of %i runs):\n", rounds);
  printf("(Unoptimized) total_keygen: %f   (%llu cycles)\n", keygen_time / freq, keygen_time);
  printf("(Unoptimized) total_sign:   %f   (%llu cycles)\n", sign_time / freq, sign_time);
  printf("(Unoptimized) total_verify: %f   (%llu cycles)\n", verify_time / freq, verify_time);

  return 0;
}

