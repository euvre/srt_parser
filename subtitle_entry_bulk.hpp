#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include <linux/io_uring.h>
#include <sys/syscall.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class file_mapping {
  const char *data_ = nullptr;
  size_t size_ = 0;

public:
  explicit file_mapping(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error(std::format("cannot open input file: {}", path));
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) {
      close(fd);
      throw std::runtime_error(std::format("cannot stat input file: {}", path));
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ > 0) {
      void *p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (p == MAP_FAILED) {
        throw std::runtime_error(
            std::format("cannot mmap input file: {}", path));
      }
      madvise(p, size_, MADV_SEQUENTIAL);
      data_ = static_cast<const char *>(p);
    } else {
      close(fd);
    }
  }
  file_mapping(const file_mapping &) = delete;
  file_mapping &operator=(const file_mapping &) = delete;
  file_mapping(file_mapping &&other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  file_mapping &operator=(file_mapping &&other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~file_mapping() { reset(); }

  const char *data() const { return data_; }
  size_t size() const { return size_; }

private:
  void reset() {
    if (data_ != nullptr) {
      munmap(const_cast<char *>(data_), size_);
      data_ = nullptr;
    }
  }
};

// Async buffered writer: copies payload into a small pool of reusable
// buffers and hands them to the kernel with io_uring IORING_OP_WRITE.
// The user->page-cache copy runs in io-wq worker threads, off the caller's
// critical path, and the output mapping (and its page faults) disappears.
class uring_writer {
  static constexpr unsigned kRingEntries{16};
  static constexpr size_t kNumBufs{8};
  static constexpr size_t kBufSize{4u << 20};

  int fd_{-1};
  int ring_{-1};
  void *sq_ring_{}, *cq_ring_{};
  io_uring_sqe *sqes_{};
  size_t sq_ring_sz_{}, cq_ring_sz_{}, sqes_sz_{};
  unsigned *sq_tail_{}, *sq_mask_{}, *sq_array_{};
  unsigned *cq_head_{}, *cq_tail_{}, *cq_mask_{};
  io_uring_cqe *cqes_{};

  char *bufs_[kNumBufs]{};
  int free_[kNumBufs]{};
  size_t nfree_{};
  int cur_{-1};
  size_t pos_{};
  size_t offset_{};
  unsigned in_flight_{};
  int error_{};

  int take_free() {
    reap_ready();
    while (nfree_ == 0) {
      wait_one();
    }
    return free_[--nfree_];
  }

  void submit(int idx, size_t len) {
    const unsigned tail = *sq_tail_;
    const unsigned i = tail & *sq_mask_;
    io_uring_sqe &e = sqes_[i];
    std::memset(&e, 0, sizeof(e));
    e.opcode = IORING_OP_WRITE;
    e.fd = fd_;
    e.addr = reinterpret_cast<uint64_t>(bufs_[idx]);
    e.len = static_cast<unsigned>(len);
    e.off = offset_;
    e.user_data = static_cast<uint64_t>(idx);
    offset_ += len;
    sq_array_[i] = i;
    std::atomic_thread_fence(std::memory_order_release);
    *sq_tail_ = tail + 1;
    if (syscall(__NR_io_uring_enter, ring_, 1, 0, 0, nullptr, 0) < 0) {
      throw std::runtime_error(std::format("io_uring_enter submit failed: {}",
                                           std::strerror(errno)));
    }
    ++in_flight_;
  }

  void reap_ready() {
    unsigned h = *cq_head_;
    std::atomic_thread_fence(std::memory_order_acquire);
    const unsigned t = *cq_tail_;
    while (h != t) {
      const io_uring_cqe &c = cqes_[h & *cq_mask_];
      if (c.res < 0 && error_ == 0) {
        error_ = -c.res;
      }
      free_[nfree_++] = static_cast<int>(c.user_data);
      --in_flight_;
      ++h;
    }
    *cq_head_ = h;
    std::atomic_thread_fence(std::memory_order_release);
  }

  void wait_one() {
    for (;;) {
      reap_ready();
      if (nfree_ > 0) {
        return;
      }
      if (syscall(__NR_io_uring_enter, ring_, 0, 1, IORING_ENTER_GETEVENTS,
                  nullptr, 0) < 0 &&
          errno != EINTR) {
        throw std::runtime_error(std::format("io_uring_enter wait failed: {}",
                                             std::strerror(errno)));
      }
    }
  }

public:
  explicit uring_writer(const std::string &path) {
    fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
      throw std::runtime_error(
          std::format("cannot open output file: {}", path));
    }
    io_uring_params p{};
    ring_ = static_cast<int>(syscall(__NR_io_uring_setup, kRingEntries, &p));
    if (ring_ < 0) {
      const int e = errno;
      close(fd_);
      fd_ = -1;
      throw std::runtime_error(
          std::format("io_uring_setup failed: {}", std::strerror(e)));
    }
    sq_ring_sz_ = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    cq_ring_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    sqes_sz_ = p.sq_entries * sizeof(io_uring_sqe);
    sq_ring_ = mmap(nullptr, sq_ring_sz_, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_SQ_RING);
    cq_ring_ = mmap(nullptr, cq_ring_sz_, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_CQ_RING);
    sqes_ = static_cast<io_uring_sqe *>(
        mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, ring_, IORING_OFF_SQES));
    if (sq_ring_ == MAP_FAILED || cq_ring_ == MAP_FAILED ||
        sqes_ == MAP_FAILED) {
      throw std::runtime_error("cannot mmap io_uring rings");
    }
    sq_tail_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_) +
                                            p.sq_off.tail);
    sq_mask_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_) +
                                            p.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_) +
                                             p.sq_off.array);
    cq_head_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ring_) +
                                            p.cq_off.head);
    cq_tail_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ring_) +
                                            p.cq_off.tail);
    cq_mask_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ring_) +
                                            p.cq_off.ring_mask);
    cqes_ = reinterpret_cast<io_uring_cqe *>(static_cast<char *>(cq_ring_) +
                                             p.cq_off.cqes);
    for (size_t i = 0; i < kNumBufs; ++i) {
      bufs_[i] = static_cast<char *>(std::aligned_alloc(4096, kBufSize));
      if (bufs_[i] == nullptr) {
        throw std::runtime_error("cannot allocate io_uring buffers");
      }
      free_[nfree_++] = static_cast<int>(i);
    }
  }
  uring_writer(const uring_writer &) = delete;
  uring_writer &operator=(const uring_writer &) = delete;
  ~uring_writer() {
    for (size_t i = 0; i < kNumBufs; ++i) {
      std::free(bufs_[i]);
    }
    if (sqes_ != nullptr && sqes_ != MAP_FAILED) {
      munmap(sqes_, sqes_sz_);
    }
    if (sq_ring_ != nullptr && sq_ring_ != MAP_FAILED) {
      munmap(sq_ring_, sq_ring_sz_);
    }
    if (cq_ring_ != nullptr && cq_ring_ != MAP_FAILED) {
      munmap(cq_ring_, cq_ring_sz_);
    }
    if (ring_ >= 0) {
      close(ring_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  void write(std::string_view s) {
    while (!s.empty()) {
      if (cur_ < 0) {
        cur_ = take_free();
        pos_ = 0;
      }
      const size_t n = s.size() < kBufSize - pos_ ? s.size() : kBufSize - pos_;
      std::memcpy(bufs_[cur_] + pos_, s.data(), n);
      pos_ += n;
      s.remove_prefix(n);
      if (pos_ == kBufSize) {
        submit(cur_, kBufSize);
        cur_ = -1;
      }
    }
  }

  void finish() {
    if (cur_ >= 0) {
      submit(cur_, pos_);
      cur_ = -1;
    }
    while (in_flight_ > 0) {
      wait_one();
    }
    if (error_ != 0) {
      throw std::runtime_error(
          std::format("async write failed: {}", std::strerror(error_)));
    }
  }
};

inline void prefault_ahead(const char *begin, const char *end,
                           const std::atomic<const char *> &cursor) {
  constexpr size_t kPage = 4096;
  constexpr ptrdiff_t kWindow = 32L << 20;
  const char *p = begin;
  while (p < end) {
    const char *cur = cursor.load(std::memory_order_relaxed);
    while (cur != nullptr && p - cur > kWindow) {
      cursor.wait(cur, std::memory_order_relaxed);
      cur = cursor.load(std::memory_order_relaxed);
    }
    if (cur == nullptr) {
      return;
    }
    (void)*static_cast<volatile const char *>(p);
    p += kPage;
  }
}

inline const char *find_newline(const char *cur, const char *end) {
  if (end - cur < 8) {
    while (cur < end) {
      if (*cur == '\n') {
        return cur;
      }
      ++cur;
    }
    return nullptr;
  }
  while (end - cur >= 8) {
    uint64_t x;
    std::memcpy(&x, cur, 8);
    x ^= 0x0A0A0A0A0A0A0A0Aull;
    const uint64_t found =
        (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
    if (found != 0) {
      return cur + (__builtin_ctzll(found) >> 3);
    }
    cur += 8;
  }
  if (cur == end) {
    return nullptr;
  }
  uint64_t x;
  std::memcpy(&x, end - 8, 8);
  x ^= 0x0A0A0A0A0A0A0A0Aull;
  const uint64_t found =
      (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
  if (found != 0) {
    return (end - 8) + (__builtin_ctzll(found) >> 3);
  }
  return nullptr;
}

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#define SRT_SIMD_V 4
struct simd_ops {
  using vec_t = __m512i;
  using mask_t = uint64_t;
  static constexpr size_t width = 64;
  static vec_t broadcast(char c) { return _mm512_set1_epi8(c); }
  static vec_t load(const char *p) { return _mm512_loadu_si512(p); }
  static mask_t cmpeq_mask(vec_t v, vec_t ref) {
    return _mm512_cmpeq_epi8_mask(v, ref);
  }
  static vec_t sub_epi8(vec_t a, vec_t b) { return _mm512_sub_epi8(a, b); }
  static vec_t subs_epu8(vec_t a, vec_t b) { return _mm512_subs_epu8(a, b); }
  static size_t ctz(mask_t m) {
    return static_cast<size_t>(__builtin_ctzll(m));
  }
};
#elif defined(__AVX2__)
#define SRT_SIMD_V 3
struct simd_ops {
  using vec_t = __m256i;
  using mask_t = uint32_t;
  static constexpr size_t width = 32;
  static vec_t broadcast(char c) { return _mm256_set1_epi8(c); }
  static vec_t load(const char *p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  }
  static mask_t cmpeq_mask(vec_t v, vec_t ref) {
    return static_cast<mask_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, ref)));
  }
  static vec_t sub_epi8(vec_t a, vec_t b) { return _mm256_sub_epi8(a, b); }
  static vec_t subs_epu8(vec_t a, vec_t b) { return _mm256_subs_epu8(a, b); }
  static size_t ctz(mask_t m) { return static_cast<size_t>(__builtin_ctz(m)); }
};
#elif defined(__SSE2__) && defined(__POPCNT__)
#define SRT_SIMD_V 2
struct simd_ops {
  using vec_t = __m128i;
  using mask_t = uint32_t;
  static constexpr size_t width = 16;
  static vec_t broadcast(char c) { return _mm_set1_epi8(c); }
  static vec_t load(const char *p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  }
  static mask_t cmpeq_mask(vec_t v, vec_t ref) {
    return static_cast<mask_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(v, ref)));
  }
  static vec_t sub_epi8(vec_t a, vec_t b) { return _mm_sub_epi8(a, b); }
  static vec_t subs_epu8(vec_t a, vec_t b) { return _mm_subs_epu8(a, b); }
  static size_t ctz(mask_t m) { return static_cast<size_t>(__builtin_ctz(m)); }
};
#else
#define SRT_SIMD_V 0
#endif
