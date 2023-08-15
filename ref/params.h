#ifndef PARAMS_H
#define PARAMS_H

#ifdef MEDS9923
  #define MEDS_name "MEDS9923"

  #define MEDS_digest_bytes 32
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 16
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 4093
  #define GFq_t uint32_t
  #define GFq_bits 12
  #define GFq_bytes 2

  #define MEDS_m 14
  #define MEDS_n 14
  #define MEDS_k 14

  #define MEDS_s 4
  #define MEDS_t 1152
  #define MEDS_w 14

  #define MEDS_seed_tree_height 11
  #define SEED_TREE_size 4095
  #define MEDS_max_path_len 100

  #define MEDS_t_mask 0x000007FF
  #define MEDS_t_bytes 2

  #define MEDS_s_mask 0x00000003

  #define MEDS_PK_BYTES 9923
  #define MEDS_SK_BYTES 1828
  #define MEDS_SIG_BYTES 9896
#endif

#ifdef MEDS13220
  #define MEDS_name "MEDS13220"

  #define MEDS_digest_bytes 32
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 16
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 4093
  #define GFq_t uint32_t
  #define GFq_bits 12
  #define GFq_bytes 2

  #define MEDS_m 14
  #define MEDS_n 14
  #define MEDS_k 14

  #define MEDS_s 5
  #define MEDS_t 192
  #define MEDS_w 20

  #define MEDS_seed_tree_height 8
  #define SEED_TREE_size 511
  #define MEDS_max_path_len 72

  #define MEDS_t_mask 0x000000FF
  #define MEDS_t_bytes 1

  #define MEDS_s_mask 0x00000007

  #define MEDS_PK_BYTES 13220
  #define MEDS_SK_BYTES 2416
  #define MEDS_SIG_BYTES 12976
#endif

#ifdef MEDS41711
  #define MEDS_name "MEDS41711"

  #define MEDS_digest_bytes 32
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 24
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 4093
  #define GFq_t uint32_t
  #define GFq_bits 12
  #define GFq_bytes 2

  #define MEDS_m 22
  #define MEDS_n 22
  #define MEDS_k 22

  #define MEDS_s 4
  #define MEDS_t 608
  #define MEDS_w 26

  #define MEDS_seed_tree_height 10
  #define SEED_TREE_size 2047
  #define MEDS_max_path_len 136

  #define MEDS_t_mask 0x000003FF
  #define MEDS_t_bytes 2

  #define MEDS_s_mask 0x00000003

  #define MEDS_PK_BYTES 41711
  #define MEDS_SK_BYTES 4420
  #define MEDS_SIG_BYTES 41080
#endif

#ifdef MEDS55604
  #define MEDS_name "MEDS55604"

  #define MEDS_digest_bytes 32
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 24
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 4093
  #define GFq_t uint32_t
  #define GFq_bits 12
  #define GFq_bytes 2

  #define MEDS_m 22
  #define MEDS_n 22
  #define MEDS_k 22

  #define MEDS_s 5
  #define MEDS_t 160
  #define MEDS_w 36

  #define MEDS_seed_tree_height 8
  #define SEED_TREE_size 511
  #define MEDS_max_path_len 100

  #define MEDS_t_mask 0x000000FF
  #define MEDS_t_bytes 1

  #define MEDS_s_mask 0x00000007

  #define MEDS_PK_BYTES 55604
  #define MEDS_SK_BYTES 5872
  #define MEDS_SIG_BYTES 54736
#endif

#ifdef MEDS134180
  #define MEDS_name "MEDS134180"

  #define MEDS_digest_bytes 32
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 32
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 2039
  #define GFq_t uint32_t
  #define GFq_bits 11
  #define GFq_bytes 2

  #define MEDS_m 30
  #define MEDS_n 30
  #define MEDS_k 30

  #define MEDS_s 5
  #define MEDS_t 192
  #define MEDS_w 52

  #define MEDS_seed_tree_height 8
  #define SEED_TREE_size 511
  #define MEDS_max_path_len 116

  #define MEDS_t_mask 0x000000FF
  #define MEDS_t_bytes 1

  #define MEDS_s_mask 0x00000007

  #define MEDS_PK_BYTES 134180
  #define MEDS_SK_BYTES 9968
  #define MEDS_SIG_BYTES 132528
#endif

#ifdef MEDS167717
  #define MEDS_name "MEDS167717"

  #define MEDS_digest_bytes 32
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 32
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 2039
  #define GFq_t uint32_t
  #define GFq_bits 11
  #define GFq_bytes 2

  #define MEDS_m 30
  #define MEDS_n 30
  #define MEDS_k 30

  #define MEDS_s 6
  #define MEDS_t 112
  #define MEDS_w 66

  #define MEDS_seed_tree_height 7
  #define SEED_TREE_size 255
  #define MEDS_max_path_len 62

  #define MEDS_t_mask 0x0000007F
  #define MEDS_t_bytes 1

  #define MEDS_s_mask 0x00000007

  #define MEDS_PK_BYTES 167717
  #define MEDS_SK_BYTES 12444
  #define MEDS_SIG_BYTES 165464
#endif

#ifdef toy
  #define MEDS_name "toy"

  #define MEDS_digest_bytes 16
  #define MEDS_pub_seed_bytes 32
  #define MEDS_sec_seed_bytes 32
  #define MEDS_st_seed_bytes 16
  #define MEDS_st_salt_bytes 32

  #define MEDS_p 8191
  #define GFq_t uint32_t
  #define GFq_bits 13
  #define GFq_bytes 2

  #define MEDS_m 10
  #define MEDS_n 10
  #define MEDS_k 10

  #define MEDS_s 4
  #define MEDS_t 16
  #define MEDS_w 6

  #define MEDS_seed_tree_height 4
  #define SEED_TREE_size 31
  #define MEDS_max_path_len 8

  #define MEDS_t_mask 0x0000000F
  #define MEDS_t_bytes 1

  #define MEDS_s_mask 0x00000003

  #define MEDS_PK_BYTES 3593
  #define MEDS_SK_BYTES 1042
  #define MEDS_SIG_BYTES 2132
#endif

#endif

