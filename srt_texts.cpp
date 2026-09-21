#include "subtitle_entry_bulk.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <meta>
#include <print>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr char kTimePattern[] = "..:..:..,... --> ..:..:..,...";
constexpr size_t kTimeLen = 29;
constexpr uint32_t kTimeCare = (1u << kTimeLen) - 1;

constexpr auto kTimeExpect = [] {
  std::array<uint8_t, 64> a{};
  for (size_t i = 0; i < kTimeLen; ++i) {
    a[i] = kTimePattern[i] == '.' ? static_cast<uint8_t>('0')
                                  : static_cast<uint8_t>(kTimePattern[i]);
  }
  return a;
}();

constexpr auto kTimeFixedPos = [] {
  uint32_t m = 0;
  for (size_t i = 0; i < kTimeLen; ++i)
    if (kTimePattern[i] != '.')
      m |= 1u << i;
  return m;
}();

constexpr auto kTimeDigitPos = [] {
  uint32_t m = 0;
  for (size_t i = 0; i < kTimeLen; ++i)
    if (kTimePattern[i] == '.')
      m |= 1u << i;
  return m;
}();

struct header_mask_template_dyn;

consteval std::meta::info named_member(std::meta::info type, const char *name) {
  return std::meta::data_member_spec(type, {.name = name,
                                            .alignment = std::nullopt,
                                            .bit_width = std::nullopt,
                                            .no_unique_address = false});
}

consteval void define_header_mask_template() {
  auto members = std::vector<std::meta::info>{};
  members.push_back(named_member(^^std::array<uint8_t, 64>, "expect_"));
  members.push_back(named_member(^^std::array<uint8_t, 64>, "limit_"));
  if constexpr (SRT_SIMD_V >= 4) {
    members.push_back(named_member(^^uint64_t, "care_mask_"));
  }
  members.push_back(named_member(^^uint8_t, "len_"));
  std::meta::define_aggregate(^^header_mask_template_dyn, members);
}

consteval { define_header_mask_template(); }

using header_mask_template = header_mask_template_dyn;

static_assert(sizeof(header_mask_template) == (SRT_SIMD_V >= 4 ? 144 : 129),
              "care_mask_ must cost storage only on x86-64-v4");

constexpr void set_care(auto &t, size_t i) {
  if constexpr (requires { t.care_mask_ |= 1ull; }) {
    t.care_mask_ |= 1ull << i;
  }
}

constexpr uint64_t care_of(const auto &t) {
  if constexpr (requires { t.care_mask_; }) {
    return static_cast<uint64_t>(t.care_mask_);
  }
  return 0;
}

constexpr auto kHeaderTmpls = [] {
  std::array<std::array<header_mask_template, 10>, 2> a{};
  for (size_t crlf = 0; crlf <= 1; ++crlf) {
    const size_t s = crlf + 1;
    for (size_t r = 1; r <= 9; ++r) {
      header_mask_template &t = a[crlf][r];
      const size_t time_off = r + s;
      t.len_ = static_cast<uint8_t>(time_off + kTimeLen + s);
      const auto mark = [&t](size_t i, uint8_t byte, bool dig) {
        t.expect_[i] = dig ? static_cast<uint8_t>('0') : byte;
        t.limit_[i] = dig ? 9 : 0;
        set_care(t, i);
      };
      for (size_t i = 0; i < r; ++i) {
        mark(i, 0, true);
      }
      if (crlf != 0) {
        mark(r, static_cast<uint8_t>('\r'), false);
      }
      mark(r + s - 1, static_cast<uint8_t>('\n'), false);
      for (size_t i = 0; i < kTimeLen; ++i)
        mark(time_off + i, static_cast<uint8_t>(kTimePattern[i]),
             kTimePattern[i] == '.');
      if (crlf != 0) {
        mark(time_off + kTimeLen, static_cast<uint8_t>('\r'), false);
        mark(time_off + kTimeLen + 1, static_cast<uint8_t>('\n'), false);
      } else {
        mark(time_off + kTimeLen, static_cast<uint8_t>('\n'), false);
      }
    }
  }
  return a;
}();

constexpr std::array<uint64_t, 8> kPow10 = {
    1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull, 10000000ull};

inline uint64_t parse_id_scaled(const char *p, uint32_t r) {
  [[assume(r >= 1 && r <= 9)]];
  const uint32_t kept = r < 8 ? r : 8;
  uint64_t x;
  std::memcpy(&x, p, 8);
  x ^= 0x3030303030303030ull;
  x &= ~0ull >> ((8 - kept) * 8);
  uint64_t v = (x * 10) + (x >> 8);
  v = (v & 0x000000FF000000FFull) * 100 + ((v >> 16) & 0x000000FF000000FFull);
  v = (v & 0xFFFFull) * 10000 + (v >> 32);
  if (r == 9) {
    v = v * 10 + static_cast<uint64_t>(p[8] - '0');
  }
  return v;
}

inline bool id_matches(const char *p, uint32_t r, uint32_t expected) {
  const uint32_t kept = r < 8 ? r : 8;
  return parse_id_scaled(p, r) ==
         static_cast<uint64_t>(expected) * kPow10[8 - kept];
}

inline uint32_t parse_id(const char *p, uint32_t r) {
  const uint32_t kept = r < 8 ? r : 8;
  return static_cast<uint32_t>(parse_id_scaled(p, r) / kPow10[8 - kept]);
}

struct header_checker_cache {
  header_mask_template t_{};
  uint32_t expected_id_{};
  uint8_t id_len_{};
  uint8_t crlf_{};

  void seed(const char *id_digits, uint8_t id_len, uint8_t crlf) {
    t_ = kHeaderTmpls[crlf][id_len];
    expected_id_ = parse_id(id_digits, id_len) + 1;
    id_len_ = id_len;
    crlf_ = crlf;
  }
};

inline void prefetch_header_ahead(const char *text_begin) {
#if defined(__x86_64__) || defined(__i386__)
  _mm_prefetch(text_begin + 64, _MM_HINT_T0);
  _mm_prefetch(text_begin + 128, _MM_HINT_T0);
  _mm_prefetch(text_begin + 192, _MM_HINT_T0);
  _mm_prefetch(text_begin + 256, _MM_HINT_T0);
#else
  (void)text_begin;
#endif
}

class text_cursor {
public:
  explicit text_cursor(const file_mapping &m)
      : line_(m.data()), end_(m.data() + m.size()) {
    if (end_ - line_ >= 3) {
      uint32_t w;
      std::memcpy(&w, line_, 4);
      if ((w & 0x00FF'FFFF) == 0x00BF'BB'EF) {
        line_ += 3;
      }
    }
  }

  const char *end() const { return end_; }
  const char *position() const { return line_; }

  size_t run(uring_writer &out, std::atomic<const char *> &cursor) {
    constexpr ptrdiff_t kWakeQuantum = 8L << 20;
    const char *last_in_wake = line_;
    size_t written{};
    const char *p = line_;
    header h = recognize_entry_start(p);
    if (!h.texts_line_) {
      return 0;
    }
    for (;;) {
      const char *text_begin = h.texts_line_;
      prefetch_header_ahead(text_begin);
      const char *text_end = end_;
      p = end_;
      const char *run_start{};
      const char *prev_after{};
      const char *scan = text_begin - 1;
#if SRT_SIMD_V >= 3
      const simd_ops::vec_t nls = simd_ops::broadcast('\n');
      while (scan + simd_ops::width <= end_) {
        const simd_ops::vec_t v = simd_ops::load(scan);
        const simd_ops::mask_t m = simd_ops::cmpeq_mask(v, nls);
        simd_ops::mask_t found = m & (m << 1);
        found |= m & (m << 2);
        while (found) {
          const size_t bit = simd_ops::ctz(found);
          const char *at = scan + bit;
          found &= found - 1;
          const uint32_t d =
              static_cast<uint32_t>(static_cast<unsigned char>(at[-1]) ^ 0x0Au);
          if (d * (d - 7) != 0) {
            continue;
          }
          const char *blank = at - 1 - (d == 7);
          if (!run_start || (prev_after && blank > prev_after)) {
            run_start = blank;
          }
          const char *after_blank = at + 1;
          if (after_blank >= end_) {
            text_end = run_start + 1;
            p = end_;
            goto body_done;
          }
          if (end_ - after_blank >= kHeaderMin && cache_.id_len_ != 0 &&
              is_header(after_blank, cache_.t_) &&
              id_matches(after_blank, cache_.id_len_, cache_.expected_id_)) {
            text_end = run_start + 1;
            p = after_blank;
            h = header{p + cache_.t_.len_, cache_.crlf_ != 0};
            ++cache_.expected_id_;
            goto body_done;
          }
          const header r = recognize(after_blank);
          if (r.texts_line_) {
            text_end = run_start + 1;
            p = after_blank;
            h = r;
            goto body_done;
          }
          prev_after = after_blank;
        }
        scan += simd_ops::width - 2;
      }
    body_done:;
#endif
      while (text_end == end_) {
        const char *blank = find_blank_line(scan);
        if (blank == nullptr) {
          break;
        }
        if (!run_start || (prev_after != nullptr && blank > prev_after)) {
          run_start = blank;
        }
        const char *after_blank = blank + (blank[1] == '\r' ? 3 : 2);
        if (after_blank >= end_) {
          text_end = run_start + 1;
          p = end_;
          break;
        }
        const header r = recognize(after_blank);
        if (r.texts_line_) {
          text_end = run_start + 1;
          p = after_blank;
          h = r;
          break;
        }
        prev_after = after_blank;
        scan = blank + 1;
      }
      out.write(std::string_view(text_begin,
                                 static_cast<size_t>(text_end - text_begin)));
      ++written;
      if ((written & 1023) == 0) {
        cursor.store(p, std::memory_order_relaxed);
        if (p - last_in_wake >= kWakeQuantum) {
          last_in_wake = p;
          cursor.notify_one();
        }
      }
      if (p == end_) {
        return written;
      }
    }
  }

private:
  struct header {
    const char *texts_line_{};
    bool is_time_crlf_{};
  };

  header recognize_entry_start(const char *&p) {
    for (;;) {
      if (p >= end_) {
        return {};
      }
      const header h = recognize(p);
      if (h.texts_line_) {
        return h;
      }
      const char *new_line = find_newline(p, end_);
      p = new_line ? new_line + 1 : end_;
    }
  }

  static bool is_time_line_swar(const char *begin, const char *end) {
    struct masks {
      uint64_t fixed_mask;
      uint64_t fixed_value;
      uint64_t digit_hi;
    };
    constexpr auto kGroups = [] {
      std::array<masks, 4> a{};
      for (size_t g = 0; g < 4; ++g) {
        const size_t off = g < 3 ? g * 8 : kTimeLen - 8;
        for (size_t i = 0; i < 8; ++i) {
          const char p = kTimePattern[off + i];
          if (p == '.') {
            a[g].digit_hi |= 0x80ull << (i * 8);
          } else {
            a[g].fixed_mask |= 0xFFull << (i * 8);
            a[g].fixed_value |= static_cast<uint64_t>(static_cast<uint8_t>(p))
                                << (i * 8);
          }
        }
      }
      return a;
    }();

    if (end - begin != static_cast<ptrdiff_t>(kTimeLen)) {
      return false;
    }
    uint64_t bad = 0;
    for (size_t g = 0; g < 4; ++g) {
      const size_t off = g < 3 ? g * 8 : kTimeLen - 8;
      uint64_t w;
      std::memcpy(&w, begin + off, 8);
      const auto &m = kGroups[g];
      bad |= (w ^ m.fixed_value) & m.fixed_mask;

      const uint64_t x = (w & 0x7F7F7F7F7F7F7F7Full) ^ 0x3030303030303030ull;
      bad |= (x + 0x7676767676767676ull) & m.digit_hi;
      bad |=
          ~((w & 0x7F7F7F7F7F7F7F7Full) + 0x5050505050505050ull) & m.digit_hi;
      bad |= w & 0x8080808080808080ull & m.digit_hi;
    }
    return bad == 0;
  }

#if SRT_SIMD_V >= 3
  bool is_time_line_simd(const char *begin) const {
    const simd_ops::vec_t v = simd_ops::load(begin);
    const simd_ops::vec_t expect =
        simd_ops::load(reinterpret_cast<const char *>(kTimeExpect.data()));
    const simd_ops::vec_t d = simd_ops::sub_epi8(v, expect);
    const simd_ops::mask_t fix_ok =
        simd_ops::cmpeq_mask(d, simd_ops::broadcast(0));
    const simd_ops::mask_t dig_ok = simd_ops::cmpeq_mask(
        simd_ops::subs_epu8(d, simd_ops::broadcast(9)), simd_ops::broadcast(0));
    const uint64_t ok = static_cast<uint64_t>((fix_ok & kTimeFixedPos) |
                                              (dig_ok & kTimeDigitPos));
    return (ok & kTimeCare) == kTimeCare;
  }
#endif

  bool is_time_line(const char *begin, const char *end) const {
    if (end - begin != static_cast<ptrdiff_t>(kTimeLen))
      return false;
#if SRT_SIMD_V >= 3
    if (begin + simd_ops::width <= end_) {
      return is_time_line_simd(begin);
    }
#endif
    return is_time_line_swar(begin, end);
  }

  const char *verify_header_tail(const char *digits_end,
                                 bool &time_crlf) const {
    const char *p = digits_end;
    if (*p == '\r') {
      ++p;
    }
    if (p >= end_ || *p != '\n') {
      return nullptr;
    }
    const char *time_begin = p + 1;
    if (end_ - time_begin < 30 || !is_time_line(time_begin, time_begin + 29)) {
      return nullptr;
    }
    const char *after = time_begin + 29;
    time_crlf = *after == '\r';
    if (time_crlf) {
      ++after;
    }
    if (after >= end_ || *after != '\n') {
      return nullptr;
    }
    return after + 1;
  }

  header recognize(const char *p) const {
#if SRT_SIMD_V >= 3
    if (end_ - p >= kHeaderMin) {
      return recognize_hot(p);
    }
#endif
    const char *digits_end = p;
    while (digits_end < end_ && digits_end - p < 9 &&
           static_cast<unsigned>(*digits_end - '0') <= 9) {
      ++digits_end;
    }
    if (digits_end == p || digits_end >= end_) {
      return {};
    }
    bool is_time_crlf{};
    const char *texts_line = verify_header_tail(digits_end, is_time_crlf);
    if (texts_line == nullptr) {
      return {};
    }
    return {texts_line, is_time_crlf};
  }

#if SRT_SIMD_V >= 3
  static constexpr ptrdiff_t kHeaderMin = 9 + 2 + simd_ops::width;

  header recognize_hot(const char *p) const {
    if (cache_.id_len_ != 0 && is_header(p, cache_.t_) &&
        id_matches(p, cache_.id_len_, cache_.expected_id_)) {
      const header h{p + cache_.t_.len_, cache_.crlf_ != 0};
      ++cache_.expected_id_;
      return h;
    }
    if (sel_r_ != 0) {
      const size_t r0 = sel_r_, crlf = sel_crlf_;
      for (size_t id_len = r0; id_len <= r0 + 1 && id_len <= 9; ++id_len) {
        const bool pos_ok = crlf == 0
                                ? p[id_len] == '\n'
                                : (p[id_len] == '\r' && p[id_len + 1] == '\n');
        if (!pos_ok) {
          continue;
        }
        const header_mask_template &t = kHeaderTmpls[crlf][id_len];
        if (is_header(p, t)) {
          sel_r_ = static_cast<uint8_t>(id_len);
          cache_.seed(p, static_cast<uint8_t>(id_len),
                      static_cast<uint8_t>(crlf));
          return {p + t.len_, crlf != 0};
        }
        break;
      }
    }
    const simd_ops::vec_t v0 = simd_ops::load(p);
    const simd_ops::mask_t nls =
        simd_ops::cmpeq_mask(v0, simd_ops::broadcast('\n'));
    if (nls == 0) {
      return {};
    }
    const size_t k = simd_ops::ctz(nls);
    size_t r = k, crlf = 0;
    if (k > 0 && p[k - 1] == '\r') {
      crlf = 1;
      r = k - 1;
    }
    if (r < 1 || r > 9) {
      return {};
    }
    const header_mask_template &t = kHeaderTmpls[crlf][r];
    if (!is_header(p, t)) {
      return {};
    }
    sel_r_ = static_cast<uint8_t>(r);
    sel_crlf_ = static_cast<uint8_t>(crlf);
    cache_.seed(p, static_cast<uint8_t>(r), static_cast<uint8_t>(crlf));
    return {p + t.len_, crlf != 0};
  }

  [[gnu::always_inline]] bool is_header(const char *p,
                                        const header_mask_template &t) const {
    const auto block_ok = [](simd_ops::vec_t v, const uint8_t *expect,
                             const uint8_t *limit, simd_ops::mask_t care) {
      const simd_ops::vec_t e =
          simd_ops::load(reinterpret_cast<const char *>(expect));
      const simd_ops::vec_t d = simd_ops::sub_epi8(v, e);
      const simd_ops::vec_t lim =
          simd_ops::load(reinterpret_cast<const char *>(limit));
      const simd_ops::mask_t ok = simd_ops::cmpeq_mask(
          simd_ops::subs_epu8(d, lim), simd_ops::broadcast(0));
      if constexpr (simd_ops::width >= 64) {
        return ok == care;
      } else {
        return ok == ~simd_ops::mask_t{0};
      }
    };
    if constexpr (simd_ops::width >= 64) {
      return block_ok(simd_ops::load(p), t.expect_.data(), t.limit_.data(),
                      care_of(t));
    } else {
      if (!block_ok(simd_ops::load(p), t.expect_.data(), t.limit_.data(), 0)) {
        return false;
      }
      const size_t off = t.len_ - simd_ops::width;
      return block_ok(simd_ops::load(p + off), t.expect_.data() + off,
                      t.limit_.data() + off, 0);
    }
  }
#endif

  const char *find_blank_line(const char *p) {
#if SRT_SIMD_V
    const simd_ops::vec_t nls = simd_ops::broadcast('\n');
    while (p + simd_ops::width <= end_) {
      const simd_ops::vec_t v = simd_ops::load(p);
      const simd_ops::mask_t m = simd_ops::cmpeq_mask(v, nls);
      simd_ops::mask_t found = m & (m << 1);
      found |= m & (m << 2);
      if (found != 0) {
        for (;;) {
          const size_t bit = simd_ops::ctz(found);
          const char *at = p + bit;
          if (at[-1] == '\n') {
            return at - 1;
          }
          if (at[-1] == '\r') {
            return at - 2;
          }
          found &= found - 1;
          if (found == 0) {
            break;
          }
        }
      }
      p += simd_ops::width - 2;
    }
#endif
    while (p + 1 < end_) {
      if (p[0] == '\n' &&
          (p[1] == '\n' || (p[1] == '\r' && p + 2 < end_ && p[2] == '\n'))) {
        return p;
      }
      ++p;
    }
    return nullptr;
  }

  const char *line_;
  const char *end_;
  mutable uint8_t sel_r_ = 0, sel_crlf_ = 0;
  mutable header_checker_cache cache_{};
};

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::println(stderr, "usage: {} <input.srt> <texts.txt>",
                 argc > 0 ? argv[0] : "srt_texts");
    return 1;
  }
  try {
    const file_mapping in(argv[1]);
    text_cursor cur(in);
    uring_writer out(argv[2]);

    std::atomic<const char *> cursor{cur.position()};
    std::thread p1(prefault_ahead, in.data(), cur.end(), std::cref(cursor));

    size_t written = 0;
    try {
      written = cur.run(out, cursor);
    } catch (...) {
      cursor.store(nullptr, std::memory_order_relaxed);
      cursor.notify_all();
      p1.join();
      throw;
    }
    out.finish();
    cursor.store(nullptr, std::memory_order_relaxed);
    cursor.notify_all();
    p1.join();
    std::println(stderr, "wrote {} text entries to {}", written, argv[2]);
  } catch (const std::exception &e) {
    std::println(stderr, "error: {}", e.what());
    return 2;
  }
  return 0;
}
