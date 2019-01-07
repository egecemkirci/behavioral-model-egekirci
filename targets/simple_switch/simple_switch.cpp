/* Copyright 2013-present Barefoot Networks, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Antonin Bas (antonin@barefootnetworks.com)
 *
 */

#include <bm/bm_sim/parser.h>
#include <bm/bm_sim/tables.h>
#include <bm/bm_sim/logger.h>

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <mutex>
#include <string>
#include <stdint.h>
#include <fstream>
#include <cstring>

#include "simple_switch.h"

#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
 
// These vars will contain the hash

uint64_t md5_64(const char *initial_msg, size_t initial_len) {
  
  uint32_t h0, h1, h2, h3;
 
    // Message (to prepare)
    uint8_t *msg = NULL;
 
    // Note: All variables are unsigned 32 bit and wrap modulo 2^32 when calculating
 
    // r specifies the per-round shift amounts
 
    uint32_t r[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
                    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    // Use binary integer part of the sines of integers (in radians) as constants// Initialize variables:
    uint32_t k[] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
 
    h0 = 0x67452301;
    h1 = 0xefcdab89;
    h2 = 0x98badcfe;
    h3 = 0x10325476;
 
    // Pre-processing: adding a single 1 bit
    //append "1" bit to message    
    /* Notice: the input bytes are considered as bits strings,
       where the first bit is the most significant bit of the byte.[37] */
 
    // Pre-processing: padding with zeros
    //append "0" bit until message length in bit ≡ 448 (mod 512)
    //append length mod (2 pow 64) to message
 
    int new_len;
    for(new_len = initial_len*8 + 1; new_len%512!=448; new_len++);
    new_len /= 8;
 
    msg = static_cast<uint8_t *>(calloc(new_len + 64, 1)); // also appends "0" bits 
                                   // (we alloc also 64 extra bytes...)
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 128; // write the "1" bit
 
    uint32_t bits_len = 8*initial_len; // note, we append the len
    memcpy(msg + new_len, &bits_len, 4);           // in bits at the end of the buffer
 
    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    int offset;
    for(offset=0; offset<new_len; offset += (512/8)) {
 
        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        uint32_t *w = (uint32_t *) (msg + offset);
 
        // Initialize hash value for this chunk:
        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
 
        // Main loop:
        uint32_t i;
        for(i = 0; i<64; i++) {

            uint32_t f, g;
 
             if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5*i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3*i + 5) % 16;          
            } else {
                f = c ^ (b | (~d));
                g = (7*i) % 16;
            }

            uint32_t temp = d;
            d = c;
            c = b;
            //printf("rotateLeft(%x + %x + %x + %x, %d)\n", a, f, k[i], w[g], r[i]);
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;
        }

    // Add this chunk's hash to result so far:
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;

    //var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)

    h0 = ((h0>>24)&0xff) | // move byte 3 to byte 0
                    ((h0<<8)&0xff0000) | // move byte 1 to byte 2
                    ((h0>>8)&0xff00) | // move byte 2 to byte 1
                    ((h0<<24)&0xff000000); // byte 0 to byte 3
    h1 = ((h1>>24)&0xff) | // move byte 3 to byte 0
                    ((h1<<8)&0xff0000) | // move byte 1 to byte 2
                    ((h1>>8)&0xff00) | // move byte 2 to byte 1
                    ((h1<<24)&0xff000000); // byte 0 to byte 3
    h2 = ((h2>>24)&0xff) | // move byte 3 to byte 0
                    ((h2<<8)&0xff0000) | // move byte 1 to byte 2
                    ((h2>>8)&0xff00) | // move byte 2 to byte 1
                    ((h2<<24)&0xff000000); // byte 0 to byte 3
    h3 = ((h3>>24)&0xff) | // move byte 3 to byte 0
                    ((h3<<8)&0xff0000) | // move byte 1 to byte 2
                    ((h3>>8)&0xff00) | // move byte 2 to byte 1
                    ((h3<<24)&0xff000000); // byte 0 to byte 3

    printf("%x", h0);
    printf("%x", h1);
    printf("%x", h2);
    printf("%x \n", h3);
 
    } 
    // cleanup
    free(msg); 

    uint64_t answer = ((uint64_t) h0 << 32) | ((uint64_t) h1);

    return answer;
}

#define ROT32(x, y) ((x << y) | (x >> (32 - y))) // avoid effort
static inline uint32_t murmur3_32(const char *key, uint32_t len, uint32_t seed) {
  static const uint32_t c1 = 0xcc9e2d51;
  static const uint32_t c2 = 0x1b873593;
  static const uint32_t r1 = 15;
  static const uint32_t r2 = 13;
  static const uint32_t m = 5;
  static const uint32_t n = 0xe6546b64;
  uint32_t hash = seed;
  const int nblocks = len / 4;
  const uint32_t *blocks = (const uint32_t *) key;
  int i;
  uint32_t k;
  for (i = 0; i < nblocks; i++) {
    k = blocks[i];
    k *= c1;
    k = ROT32(k, r1);
    k *= c2;
    hash ^= k;
    hash = ROT32(hash, r2) * m + n;
  }
  const uint8_t *tail = (const uint8_t *) (key + nblocks * 4);
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = ROT32(k1, r1);
      k1 *= c2;
      hash ^= k1;
  }
  hash ^= len;
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);
  return hash;
}

using std::string;
using std::cout;
using std::endl;
 
class SHA256
{
protected:
    typedef unsigned char uint8;
    typedef unsigned int uint32;
    typedef unsigned long long uint64;
 
    const static uint32 sha256_k[];
    static const unsigned int SHA224_256_BLOCK_SIZE = (512/8);
public:
    void init();
    void update(const unsigned char *message, unsigned int len);
    void final(unsigned char *digest);
    static const unsigned int DIGEST_SIZE = ( 256 / 8);
 
protected:
    void transform(const unsigned char *message, unsigned int block_nb);
    unsigned int m_tot_len;
    unsigned int m_len;
    unsigned char m_block[2*SHA224_256_BLOCK_SIZE];
    uint32 m_h[8];
};
 
std::string sha256(std::string input);
 
#define SHA2_SHFR(x, n)    (x >> n)
#define SHA2_ROTR(x, n)   ((x >> n) | (x << ((sizeof(x) << 3) - n)))
#define SHA2_ROTL(x, n)   ((x << n) | (x >> ((sizeof(x) << 3) - n)))
#define SHA2_CH(x, y, z)  ((x & y) ^ (~x & z))
#define SHA2_MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define SHA256_F1(x) (SHA2_ROTR(x,  2) ^ SHA2_ROTR(x, 13) ^ SHA2_ROTR(x, 22))
#define SHA256_F2(x) (SHA2_ROTR(x,  6) ^ SHA2_ROTR(x, 11) ^ SHA2_ROTR(x, 25))
#define SHA256_F3(x) (SHA2_ROTR(x,  7) ^ SHA2_ROTR(x, 18) ^ SHA2_SHFR(x,  3))
#define SHA256_F4(x) (SHA2_ROTR(x, 17) ^ SHA2_ROTR(x, 19) ^ SHA2_SHFR(x, 10))
#define SHA2_UNPACK32(x, str)                 \
{                                             \
    *((str) + 3) = (uint8) ((x)      );       \
    *((str) + 2) = (uint8) ((x) >>  8);       \
    *((str) + 1) = (uint8) ((x) >> 16);       \
    *((str) + 0) = (uint8) ((x) >> 24);       \
}
#define SHA2_PACK32(str, x)                   \
{                                             \
    *(x) =   ((uint32) *((str) + 3)      )    \
           | ((uint32) *((str) + 2) <<  8)    \
           | ((uint32) *((str) + 1) << 16)    \
           | ((uint32) *((str) + 0) << 24);   \
}


const unsigned int SHA256::sha256_k[64] = //UL = uint32
            {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
             0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
             0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
             0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
             0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
             0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
             0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
             0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
             0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
             0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
             0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
             0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
             0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
             0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
             0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
             0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
 
void SHA256::transform(const unsigned char *message, unsigned int block_nb)
{
    uint32 w[64];
    uint32 wv[8];
    uint32 t1, t2;
    const unsigned char *sub_block;
    int i;
    int j;
    for (i = 0; i < (int) block_nb; i++) {
        sub_block = message + (i << 6);
        for (j = 0; j < 16; j++) {
            SHA2_PACK32(&sub_block[j << 2], &w[j]);
        }
        for (j = 16; j < 64; j++) {
            w[j] =  SHA256_F4(w[j -  2]) + w[j -  7] + SHA256_F3(w[j - 15]) + w[j - 16];
        }
        for (j = 0; j < 8; j++) {
            wv[j] = m_h[j];
        }
        for (j = 0; j < 64; j++) {
            t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
                + sha256_k[j] + w[j];
            t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }
        for (j = 0; j < 8; j++) {
            m_h[j] += wv[j];
        }
    }
}
 
void SHA256::init()
{
    m_h[0] = 0x6a09e667;
    m_h[1] = 0xbb67ae85;
    m_h[2] = 0x3c6ef372;
    m_h[3] = 0xa54ff53a;
    m_h[4] = 0x510e527f;
    m_h[5] = 0x9b05688c;
    m_h[6] = 0x1f83d9ab;
    m_h[7] = 0x5be0cd19;
    m_len = 0;
    m_tot_len = 0;
}
 
void SHA256::update(const unsigned char *message, unsigned int len)
{
    unsigned int block_nb;
    unsigned int new_len, rem_len, tmp_len;
    const unsigned char *shifted_message;
    tmp_len = SHA224_256_BLOCK_SIZE - m_len;
    rem_len = len < tmp_len ? len : tmp_len;
    memcpy(&m_block[m_len], message, rem_len);
    if (m_len + len < SHA224_256_BLOCK_SIZE) {
        m_len += len;
        return;
    }
    new_len = len - rem_len;
    block_nb = new_len / SHA224_256_BLOCK_SIZE;
    shifted_message = message + rem_len;
    transform(m_block, 1);
    transform(shifted_message, block_nb);
    rem_len = new_len % SHA224_256_BLOCK_SIZE;
    memcpy(m_block, &shifted_message[block_nb << 6], rem_len);
    m_len = rem_len;
    m_tot_len += (block_nb + 1) << 6;
}
 
void SHA256::final(unsigned char *digest)
{
    unsigned int block_nb;
    unsigned int pm_len;
    unsigned int len_b;
    int i;
    block_nb = (1 + ((SHA224_256_BLOCK_SIZE - 9)
                     < (m_len % SHA224_256_BLOCK_SIZE)));
    len_b = (m_tot_len + m_len) << 3;
    pm_len = block_nb << 6;
    memset(m_block + m_len, 0, pm_len - m_len);
    m_block[m_len] = 0x80;
    SHA2_UNPACK32(len_b, m_block + pm_len - 4);
    transform(m_block, block_nb);
    for (i = 0 ; i < 8; i++) {
        SHA2_UNPACK32(m_h[i], &digest[i << 2]);
    }
}
 
uint64_t sha256(const char *input, size_t initial_len)
{
    unsigned char digest[SHA256::DIGEST_SIZE];
    memset(digest,0,SHA256::DIGEST_SIZE);
 
    SHA256 ctx = SHA256();
    ctx.init();
    ctx.update( (unsigned char*)input, initial_len);
    ctx.final(digest);
 
    char buf[2*SHA256::DIGEST_SIZE+1];
    buf[2*SHA256::DIGEST_SIZE] = 0;
    for (int i = 0; i < SHA256::DIGEST_SIZE; i++)
        sprintf(buf+i*2, "%02x", digest[i]);

    char temp[16];

    for(int i=0; i<16; i++) {
      temp[i] = buf[i];
    }

    for(int i=0 ; i<16 ; i++)
  {
      cout << temp[i];
  }
    cout << endl;


  uint64_t paquet = 0;
  for(int i=0; i<16; i++) {
    uint64_t u_temp = 0;
    switch (temp[i]) {
      case '0': u_temp = 0;
          break;
          case '1': u_temp = 1;
          break;
          case '2': u_temp = 2;
          break;
          case '3': u_temp = 3;
          break;
          case '4': u_temp = 4;
          break;
          case '5': u_temp = 5;
          break;
          case '6': u_temp = 6;
          break;
          case '7': u_temp = 7;
          break;
          case '8': u_temp = 8;
          break;
          case '9': u_temp = 9;
          break;
          case 'a': u_temp = 10;
          break;
          case 'b': u_temp = 11;
          break;
          case 'c': u_temp = 12;
          break;
          case 'd': u_temp = 13;
          break;
          case 'e': u_temp = 14;
          break;
          case 'f': u_temp = 15;
          break;
       }
       paquet = paquet | u_temp;
       if(i != 15){
        paquet <<= 4;
       }
  } 
    return paquet;
}

namespace {

struct my_murmur32 {
  uint32_t operator()(const char *buf, size_t s) const {
    return murmur3_32(buf, s, 1);
  }
};

struct my_md5_64 {
  uint64_t operator()(const char *buf, size_t s) const {
    return md5_64(buf, s);
  }
};  

struct my_sha256 {
  uint64_t operator()(const char *buf, size_t s) const {
    return sha256(buf, s);
  }
};

struct hash_ex {
  uint32_t operator()(const char *buf, size_t s) const {
    const uint32_t p = 16777619;
    uint32_t hash = 2166136261;

    for (size_t i = 0; i < s; i++)
      hash = (hash ^ buf[i]) * p;

    hash += hash << 13;
    hash ^= hash >> 7;
    hash += hash << 3;
    hash ^= hash >> 17;
    hash += hash << 5;
    return static_cast<uint32_t>(hash);
  }
};

struct bmv2_hash {
  uint64_t operator()(const char *buf, size_t s) const {
    return bm::hash::xxh64(buf, s);
  }
};

}  // namespace

// if REGISTER_HASH calls placed in the anonymous namespace, some compiler can
// give an unused variable warning
REGISTER_HASH(my_md5_64);
REGISTER_HASH(my_sha256);
REGISTER_HASH(hash_ex);
REGISTER_HASH(bmv2_hash);
REGISTER_HASH(my_murmur32);

extern int import_primitives();

packet_id_t SimpleSwitch::packet_id = 0;

class SimpleSwitch::MirroringSessions {
 public:
  bool add_session(mirror_id_t mirror_id,
                   const MirroringSessionConfig &config) {
    Lock lock(mutex);
    sessions_map[mirror_id] = config;
    return true;
  }

  bool delete_session(mirror_id_t mirror_id) {
    Lock lock(mutex);
    return sessions_map.erase(mirror_id) == 1;
  }

  bool get_session(mirror_id_t mirror_id,
                   MirroringSessionConfig *config) const {
    Lock lock(mutex);
    auto it = sessions_map.find(mirror_id);
    if (it == sessions_map.end()) return false;
    *config = it->second;
    return true;
  }

 private:
  using Mutex = std::mutex;
  using Lock = std::lock_guard<Mutex>;

  mutable std::mutex mutex;
  std::unordered_map<mirror_id_t, MirroringSessionConfig> sessions_map;
};

SimpleSwitch::SimpleSwitch(port_t max_port, bool enable_swap)
  : Switch(enable_swap),
    max_port(max_port),
    input_buffer(1024),
#ifdef SSWITCH_PRIORITY_QUEUEING_ON
    egress_buffers(max_port, nb_egress_threads,
                   64, EgressThreadMapper(nb_egress_threads),
                   SSWITCH_PRIORITY_QUEUEING_NB_QUEUES),
#else
    egress_buffers(max_port, nb_egress_threads,
                   64, EgressThreadMapper(nb_egress_threads)),
#endif
    output_buffer(128),
    // cannot use std::bind because of a clang bug
    // https://stackoverflow.com/questions/32030141/is-this-incorrect-use-of-stdbind-or-a-compiler-bug
    my_transmit_fn([this](port_t port_num, packet_id_t pkt_id,
                          const char *buffer, int len) {
        _BM_UNUSED(pkt_id);
        this->transmit_fn(port_num, buffer, len);
    }),
    pre(new McSimplePreLAG()),
    start(clock::now()),
    mirroring_sessions(new MirroringSessions()) {
  add_component<McSimplePreLAG>(pre);

  add_required_field("standard_metadata", "ingress_port");
  add_required_field("standard_metadata", "packet_length");
  add_required_field("standard_metadata", "instance_type");
  add_required_field("standard_metadata", "egress_spec");
  add_required_field("standard_metadata", "clone_spec");
  add_required_field("standard_metadata", "egress_port");

  force_arith_header("standard_metadata");
  force_arith_header("queueing_metadata");
  force_arith_header("intrinsic_metadata");

  import_primitives();
}

#define PACKET_LENGTH_REG_IDX 0

int
SimpleSwitch::receive_(port_t port_num, const char *buffer, int len) {
  // this is a good place to call this, because blocking this thread will not
  // block the processing of existing packet instances, which is a requirement
  if (do_swap() == 0) {
    check_queueing_metadata();
  }

  // we limit the packet buffer to original size + 512 bytes, which means we
  // cannot add more than 512 bytes of header data to the packet, which should
  // be more than enough
  auto packet = new_packet_ptr(port_num, packet_id++, len,
                               bm::PacketBuffer(len + 512, buffer, len));

  BMELOG(packet_in, *packet);

  PHV *phv = packet->get_phv();
  // many current P4 programs assume this
  // it is also part of the original P4 spec
  phv->reset_metadata();

  // setting standard metadata

  phv->get_field("standard_metadata.ingress_port").set(port_num);
  // using packet register 0 to store length, this register will be updated for
  // each add_header / remove_header primitive call
  packet->set_register(PACKET_LENGTH_REG_IDX, len);
  phv->get_field("standard_metadata.packet_length").set(len);
  Field &f_instance_type = phv->get_field("standard_metadata.instance_type");
  f_instance_type.set(PKT_INSTANCE_TYPE_NORMAL);

  if (phv->has_field("intrinsic_metadata.ingress_global_timestamp")) {
    phv->get_field("intrinsic_metadata.ingress_global_timestamp")
        .set(get_ts().count());
  }

  input_buffer.push_front(std::move(packet));
  return 0;
}

void
SimpleSwitch::start_and_return_() {
  check_queueing_metadata();

  threads_.push_back(std::thread(&SimpleSwitch::ingress_thread, this));
  for (size_t i = 0; i < nb_egress_threads; i++) {
    threads_.push_back(std::thread(&SimpleSwitch::egress_thread, this, i));
  }
  threads_.push_back(std::thread(&SimpleSwitch::transmit_thread, this));
}

SimpleSwitch::~SimpleSwitch() {
  input_buffer.push_front(nullptr);
  for (size_t i = 0; i < nb_egress_threads; i++) {
#ifdef SSWITCH_PRIORITY_QUEUEING_ON
    egress_buffers.push_front(i, 0, nullptr);
#else
    egress_buffers.push_front(i, nullptr);
#endif
  }
  output_buffer.push_front(nullptr);
  for (auto& thread_ : threads_) {
    thread_.join();
  }
}

void
SimpleSwitch::reset_target_state_() {
  bm::Logger::get()->debug("Resetting simple_switch target-specific state");
  get_component<McSimplePreLAG>()->reset_state();
}

bool
SimpleSwitch::mirroring_add_session(mirror_id_t mirror_id,
                                    const MirroringSessionConfig &config) {
  return mirroring_sessions->add_session(mirror_id, config);
}

bool
SimpleSwitch::mirroring_delete_session(mirror_id_t mirror_id) {
  return mirroring_sessions->delete_session(mirror_id);
}

bool
SimpleSwitch::mirroring_get_session(mirror_id_t mirror_id,
                                    MirroringSessionConfig *config) const {
  return mirroring_sessions->get_session(mirror_id, config);
}

int
SimpleSwitch::set_egress_queue_depth(size_t port, const size_t depth_pkts) {
  egress_buffers.set_capacity(port, depth_pkts);
  return 0;
}

int
SimpleSwitch::set_all_egress_queue_depths(const size_t depth_pkts) {
  for (uint32_t i = 0; i < max_port; i++) {
    set_egress_queue_depth(i, depth_pkts);
  }
  return 0;
}

int
SimpleSwitch::set_egress_queue_rate(size_t port, const uint64_t rate_pps) {
  egress_buffers.set_rate(port, rate_pps);
  return 0;
}

int
SimpleSwitch::set_all_egress_queue_rates(const uint64_t rate_pps) {
  for (uint32_t i = 0; i < max_port; i++) {
    set_egress_queue_rate(i, rate_pps);
  }
  return 0;
}

uint64_t
SimpleSwitch::get_time_elapsed_us() const {
  return get_ts().count();
}

uint64_t
SimpleSwitch::get_time_since_epoch_us() const {
  auto tp = clock::now();
  return duration_cast<ts_res>(tp.time_since_epoch()).count();
}

void
SimpleSwitch::set_transmit_fn(TransmitFn fn) {
  my_transmit_fn = std::move(fn);
}

void
SimpleSwitch::transmit_thread() {
  while (1) {
    std::unique_ptr<Packet> packet;
    output_buffer.pop_back(&packet);
    if (packet == nullptr) break;
    BMELOG(packet_out, *packet);
    BMLOG_DEBUG_PKT(*packet, "Transmitting packet of size {} out of port {}",
                    packet->get_data_size(), packet->get_egress_port());
    my_transmit_fn(packet->get_egress_port(), packet->get_packet_id(),
                   packet->data(), packet->get_data_size());
  }
}

ts_res
SimpleSwitch::get_ts() const {
  return duration_cast<ts_res>(clock::now() - start);
}

void
SimpleSwitch::enqueue(port_t egress_port, std::unique_ptr<Packet> &&packet) {
    if (egress_port >= max_port) {
      bm::Logger::get()->error("Invalid egress port %u, dropping packet",
                               egress_port);
      return;
    }

    packet->set_egress_port(egress_port);

    PHV *phv = packet->get_phv();

    if (with_queueing_metadata) {
      phv->get_field("queueing_metadata.enq_timestamp").set(get_ts().count());
      phv->get_field("queueing_metadata.enq_qdepth")
          .set(egress_buffers.size(egress_port));
    }

#ifdef SSWITCH_PRIORITY_QUEUEING_ON
    size_t priority = phv->has_field(SSWITCH_PRIORITY_QUEUEING_SRC) ?
        phv->get_field(SSWITCH_PRIORITY_QUEUEING_SRC).get<size_t>() : 0u;
    if (priority >= SSWITCH_PRIORITY_QUEUEING_NB_QUEUES) {
      bm::Logger::get()->error("Priority out of range, dropping packet");
      return;
    }
    egress_buffers.push_front(
        egress_port, SSWITCH_PRIORITY_QUEUEING_NB_QUEUES - 1 - priority,
        std::move(packet));
#else
    egress_buffers.push_front(egress_port, std::move(packet));
#endif
}

// used for ingress cloning, resubmit
void
SimpleSwitch::copy_field_list_and_set_type(
    const std::unique_ptr<Packet> &packet,
    const std::unique_ptr<Packet> &packet_copy,
    PktInstanceType copy_type, p4object_id_t field_list_id) {
  PHV *phv_copy = packet_copy->get_phv();
  phv_copy->reset_metadata();
  FieldList *field_list = this->get_field_list(field_list_id);
  field_list->copy_fields_between_phvs(phv_copy, packet->get_phv());
  phv_copy->get_field("standard_metadata.instance_type").set(copy_type);
}

void
SimpleSwitch::check_queueing_metadata() {
  // TODO(antonin): add qid in required fields
  bool enq_timestamp_e = field_exists("queueing_metadata", "enq_timestamp");
  bool enq_qdepth_e = field_exists("queueing_metadata", "enq_qdepth");
  bool deq_timedelta_e = field_exists("queueing_metadata", "deq_timedelta");
  bool deq_qdepth_e = field_exists("queueing_metadata", "deq_qdepth");
  if (enq_timestamp_e || enq_qdepth_e || deq_timedelta_e || deq_qdepth_e) {
    if (enq_timestamp_e && enq_qdepth_e && deq_timedelta_e && deq_qdepth_e)
      with_queueing_metadata = true;
    else
      bm::Logger::get()->warn(
          "Your JSON input defines some but not all queueing metadata fields");
  }
}

void
SimpleSwitch::multicast(Packet *packet, unsigned int mgid) {
  auto *phv = packet->get_phv();
  auto &f_rid = phv->get_field("intrinsic_metadata.egress_rid");
  const auto pre_out = pre->replicate({mgid});
  auto packet_size = packet->get_register(PACKET_LENGTH_REG_IDX);
  for (const auto &out : pre_out) {
    auto egress_port = out.egress_port;
    BMLOG_DEBUG_PKT(*packet, "Replicating packet on port {}", egress_port);
    f_rid.set(out.rid);
    std::unique_ptr<Packet> packet_copy = packet->clone_with_phv_ptr();
    packet_copy->set_register(PACKET_LENGTH_REG_IDX, packet_size);
    enqueue(egress_port, std::move(packet_copy));
  }
}

void
SimpleSwitch::ingress_thread() {
  PHV *phv;

  while (1) {
    std::unique_ptr<Packet> packet;
    input_buffer.pop_back(&packet);
    if (packet == nullptr) break;

    // TODO(antonin): only update these if swapping actually happened?
    Parser *parser = this->get_parser("parser");
    Pipeline *ingress_mau = this->get_pipeline("ingress");

    phv = packet->get_phv();

    port_t ingress_port = packet->get_ingress_port();
    (void) ingress_port;
    BMLOG_DEBUG_PKT(*packet, "Processing packet received on port {}",
                    ingress_port);

    auto ingress_packet_size = packet->get_register(PACKET_LENGTH_REG_IDX);

    /* This looks like it comes out of the blue. However this is needed for
       ingress cloning. The parser updates the buffer state (pops the parsed
       headers) to make the deparser's job easier (the same buffer is
       re-used). But for ingress cloning, the original packet is needed. This
       kind of looks hacky though. Maybe a better solution would be to have the
       parser leave the buffer unchanged, and move the pop logic to the
       deparser. TODO? */
    const Packet::buffer_state_t packet_in_state = packet->save_buffer_state();
    parser->parse(packet.get());

    if (phv->has_field("standard_metadata.parser_error")) {
      phv->get_field("standard_metadata.parser_error").set(
          packet->get_error_code().get());
    }

    if (phv->has_field("standard_metadata.checksum_error")) {
      phv->get_field("standard_metadata.checksum_error").set(
           packet->get_checksum_error() ? 1 : 0);
    }


    ingress_mau->apply(packet.get());

    packet->reset_exit();

    Field &f_egress_spec = phv->get_field("standard_metadata.egress_spec");
    port_t egress_spec = f_egress_spec.get_uint();

    Field &f_clone_spec = phv->get_field("standard_metadata.clone_spec");
    unsigned int clone_spec = f_clone_spec.get_uint();

    int learn_id = 0;
    unsigned int mgid = 0u;

    if (phv->has_field("intrinsic_metadata.lf_field_list")) {
      Field &f_learn_id = phv->get_field("intrinsic_metadata.lf_field_list");
      learn_id = f_learn_id.get_int();
    }

    // detect mcast support, if this is true we assume that other fields needed
    // for mcast are also defined
    if (phv->has_field("intrinsic_metadata.mcast_grp")) {
      Field &f_mgid = phv->get_field("intrinsic_metadata.mcast_grp");
      mgid = f_mgid.get_uint();
    }

    // INGRESS CLONING
    if (clone_spec) {
      BMLOG_DEBUG_PKT(*packet, "Cloning packet at ingress");
      f_clone_spec.set(0);
      MirroringSessionConfig config;
      bool is_session_configured = mirroring_get_session(
          static_cast<mirror_id_t>(clone_spec & 0xFFFF), &config);
      if (is_session_configured) {
        const Packet::buffer_state_t packet_out_state =
            packet->save_buffer_state();
        packet->restore_buffer_state(packet_in_state);
        p4object_id_t field_list_id = clone_spec >> 16;
        std::unique_ptr<Packet> packet_copy = packet->clone_no_phv_ptr();
        packet_copy->set_register(PACKET_LENGTH_REG_IDX, ingress_packet_size);
        // we need to parse again
        // the alternative would be to pay the (huge) price of PHV copy for
        // every ingress packet
        parser->parse(packet_copy.get());
        copy_field_list_and_set_type(packet, packet_copy,
                                     PKT_INSTANCE_TYPE_INGRESS_CLONE,
                                     field_list_id);
        if (config.mgid_valid) {
          BMLOG_DEBUG_PKT(*packet, "Cloning packet to MGID {}", config.mgid);
          multicast(packet_copy.get(), config.mgid);
        }
        if (config.egress_port_valid) {
          BMLOG_DEBUG_PKT(*packet, "Cloning packet to egress port {}",
                          config.egress_port);
          enqueue(config.egress_port, std::move(packet_copy));
        }
        packet->restore_buffer_state(packet_out_state);
      }
    }

    // LEARNING
    if (learn_id > 0) {
      get_learn_engine()->learn(learn_id, *packet.get());
    }

    // RESUBMIT
    if (phv->has_field("intrinsic_metadata.resubmit_flag")) {
      Field &f_resubmit = phv->get_field("intrinsic_metadata.resubmit_flag");
      if (f_resubmit.get_int()) {
        BMLOG_DEBUG_PKT(*packet, "Resubmitting packet");
        // get the packet ready for being parsed again at the beginning of
        // ingress
        packet->restore_buffer_state(packet_in_state);
        p4object_id_t field_list_id = f_resubmit.get_int();
        f_resubmit.set(0);
        // TODO(antonin): a copy is not needed here, but I don't yet have an
        // optimized way of doing this
        std::unique_ptr<Packet> packet_copy = packet->clone_no_phv_ptr();
        copy_field_list_and_set_type(packet, packet_copy,
                                     PKT_INSTANCE_TYPE_RESUBMIT,
                                     field_list_id);
        input_buffer.push_front(std::move(packet_copy));
        continue;
      }
    }

    // MULTICAST
    if (mgid != 0) {
      BMLOG_DEBUG_PKT(*packet, "Multicast requested for packet");
      auto &f_instance_type = phv->get_field("standard_metadata.instance_type");
      f_instance_type.set(PKT_INSTANCE_TYPE_REPLICATION);
      multicast(packet.get(), mgid);
      // when doing multicast, we discard the original packet
      continue;
    }

    port_t egress_port = egress_spec;
    BMLOG_DEBUG_PKT(*packet, "Egress port is {}", egress_port);

    if (egress_port == 511) {  // drop packet
      BMLOG_DEBUG_PKT(*packet, "Dropping packet at the end of ingress");
      continue;
    }

    enqueue(egress_port, std::move(packet));
  }
}

void
SimpleSwitch::egress_thread(size_t worker_id) {
  PHV *phv;

  while (1) {
    std::unique_ptr<Packet> packet;
    size_t port;
#ifdef SSWITCH_PRIORITY_QUEUEING_ON
    size_t priority;
    egress_buffers.pop_back(worker_id, &port, &priority, &packet);
#else
    egress_buffers.pop_back(worker_id, &port, &packet);
#endif
    if (packet == nullptr) break;

    Deparser *deparser = this->get_deparser("deparser");
    Pipeline *egress_mau = this->get_pipeline("egress");

    phv = packet->get_phv();

    if (phv->has_field("intrinsic_metadata.egress_global_timestamp")) {
      phv->get_field("intrinsic_metadata.egress_global_timestamp")
          .set(get_ts().count());
    }

    if (with_queueing_metadata) {
      auto enq_timestamp =
          phv->get_field("queueing_metadata.enq_timestamp").get<ts_res::rep>();
      phv->get_field("queueing_metadata.deq_timedelta").set(
          get_ts().count() - enq_timestamp);
      phv->get_field("queueing_metadata.deq_qdepth").set(
          egress_buffers.size(port));
      if (phv->has_field("queueing_metadata.qid")) {
        auto &qid_f = phv->get_field("queueing_metadata.qid");
#ifdef SSWITCH_PRIORITY_QUEUEING_ON
        qid_f.set(SSWITCH_PRIORITY_QUEUEING_NB_QUEUES - 1 - priority);
#else
        qid_f.set(0);
#endif
      }
    }

    phv->get_field("standard_metadata.egress_port").set(port);

    Field &f_egress_spec = phv->get_field("standard_metadata.egress_spec");
    f_egress_spec.set(0);

    phv->get_field("standard_metadata.packet_length").set(
        packet->get_register(PACKET_LENGTH_REG_IDX));

    egress_mau->apply(packet.get());

    Field &f_clone_spec = phv->get_field("standard_metadata.clone_spec");
    unsigned int clone_spec = f_clone_spec.get_uint();

    // EGRESS CLONING
    if (clone_spec) {
      BMLOG_DEBUG_PKT(*packet, "Cloning packet at egress");
      f_clone_spec.set(0);
      MirroringSessionConfig config;
      bool is_session_configured = mirroring_get_session(
          static_cast<mirror_id_t>(clone_spec & 0xFFFF), &config);
      if (is_session_configured) {
        p4object_id_t field_list_id = clone_spec >> 16;
        std::unique_ptr<Packet> packet_copy =
            packet->clone_with_phv_reset_metadata_ptr();
        PHV *phv_copy = packet_copy->get_phv();
        FieldList *field_list = this->get_field_list(field_list_id);
        field_list->copy_fields_between_phvs(phv_copy, phv);
        phv_copy->get_field("standard_metadata.instance_type")
            .set(PKT_INSTANCE_TYPE_EGRESS_CLONE);
        if (config.mgid_valid) {
          BMLOG_DEBUG_PKT(*packet, "Cloning packet to MGID {}", config.mgid);
          multicast(packet_copy.get(), config.mgid);
        }
        if (config.egress_port_valid) {
          BMLOG_DEBUG_PKT(*packet, "Cloning packet to egress port {}",
                          config.egress_port);
          enqueue(config.egress_port, std::move(packet_copy));
        }
      }
    }

    // TODO(antonin): should not be done like this in egress pipeline
    port_t egress_spec = f_egress_spec.get_uint();
    if (egress_spec == 511) {  // drop packet
      BMLOG_DEBUG_PKT(*packet, "Dropping packet at the end of egress");
      continue;
    }

    deparser->deparse(packet.get());

    // RECIRCULATE
    if (phv->has_field("intrinsic_metadata.recirculate_flag")) {
      Field &f_recirc = phv->get_field("intrinsic_metadata.recirculate_flag");
      if (f_recirc.get_int()) {
        BMLOG_DEBUG_PKT(*packet, "Recirculating packet");
        p4object_id_t field_list_id = f_recirc.get_int();
        f_recirc.set(0);
        FieldList *field_list = this->get_field_list(field_list_id);
        // TODO(antonin): just like for resubmit, there is no need for a copy
        // here, but it is more convenient for this first prototype
        std::unique_ptr<Packet> packet_copy = packet->clone_no_phv_ptr();
        PHV *phv_copy = packet_copy->get_phv();
        phv_copy->reset_metadata();
        field_list->copy_fields_between_phvs(phv_copy, phv);
        phv_copy->get_field("standard_metadata.instance_type")
            .set(PKT_INSTANCE_TYPE_RECIRC);
        size_t packet_size = packet_copy->get_data_size();
        packet_copy->set_register(PACKET_LENGTH_REG_IDX, packet_size);
        phv_copy->get_field("standard_metadata.packet_length").set(packet_size);
        // TODO(antonin): really it may be better to create a new packet here or
        // to fold this functionality into the Packet class?
        packet_copy->set_ingress_length(packet_size);
        input_buffer.push_front(std::move(packet_copy));
        continue;
      }
    }

    output_buffer.push_front(std::move(packet));
  }
}
