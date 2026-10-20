// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <omp.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/library.h>
#include <vector>

static int64_t set_worker_threads(int64_t requested) {
  TORCH_CHECK(requested > 0 && requested <= std::numeric_limits<int>::max(),
              "PLE CPU update threads must fit a positive OpenMP integer");
  at::internal::lazy_init_num_threads();
  const auto previous = omp_get_max_threads();
  omp_set_num_threads(static_cast<int>(requested));
  return previous;
}

class ScopedThreads {
 public:
  explicit ScopedThreads(int64_t requested) : previous_(0) {
    TORCH_CHECK(requested >= 0, "PLE CPU update threads must be nonnegative");
    if (requested) {
      previous_ = set_worker_threads(requested);
    }
  }
  ~ScopedThreads() {
    if (previous_) omp_set_num_threads(previous_);
  }

 private:
  int previous_;
};

static void store_gradient(float *destination, __m512 gradient) {
  // Preserve all bits, including signed zero and NaN payload changes.
  const auto changed = _mm512_cmpneq_epi32_mask(
      _mm512_castps_si512(_mm512_loadu_ps(destination)),
      _mm512_castps_si512(gradient));
  _mm512_mask_storeu_ps(destination, changed, gradient);
}

static void store_bfloat16(at::BFloat16 *destination, __m512 value, bool streaming) {
  const auto bits = _mm512_castps_si512(value);
  const auto lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
  auto rounded = _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb));
  rounded = _mm512_mask_mov_epi32(
      rounded, _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q),
      _mm512_set1_epi32(static_cast<int32_t>(0xffff0000u)));
  const auto result = _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
  if (streaming && (reinterpret_cast<uintptr_t>(destination) % 32 == 0))
    _mm256_stream_si256(reinterpret_cast<__m256i *>(destination), result);
  else
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination), result);
}

// Match the installed native CPU FP32 vector AdamW arithmetic, including
// clipping's intermediate rounding and both native FMA placements.
static void validate(const at::Tensor &p, const at::Tensor &m,
                     const at::Tensor &v, const at::Tensor &g) {
  for (const auto &value : {p, m, v, g}) {
    TORCH_CHECK(value.device().is_cpu() && value.scalar_type() == at::kFloat,
                "PLE AdamW inputs must be CPU FP32 tensors");
    TORCH_CHECK(value.is_contiguous() && value.sizes() == p.sizes(),
                "PLE AdamW inputs must have matching contiguous shapes");
  }
  TORCH_CHECK(p.numel() % 16 == 0, "PLE AdamW requires complete FP32 vectors");
}

void update_dense(at::Tensor p, at::Tensor m, at::Tensor v, at::Tensor g,
                  double lr, double b1, double b2, double wd, double eps,
                  double step, double clipping, int64_t num_threads) {
  validate(p, m, v, g);
  ScopedThreads threads(num_threads);
  auto *pp = p.data_ptr<float>();
  auto *mp = m.data_ptr<float>();
  auto *vp = v.data_ptr<float>();
  auto *gp = g.data_ptr<float>();
  const __m512 decay = _mm512_set1_ps(float(1 - lr * wd));
  const float lw = float(1 - b1);
  const __m512 weight = _mm512_set1_ps(lw < .5f ? lw : lw - 1.f);
  const __m512 vb = _mm512_set1_ps(float(b2)),
               vw = _mm512_set1_ps(float(1 - b2));
  const __m512 sb = _mm512_set1_ps(float(std::sqrt(1 - std::pow(b2, step))));
  const __m512 epsilon = _mm512_set1_ps(float(eps));
  const __m512 ss = _mm512_set1_ps(float(-lr / (1 - std::pow(b1, step))));
  const __m512 coef = _mm512_set1_ps(float(clipping));
  at::parallel_for(0, p.numel() / 16, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin * 16; i < end * 16; i += 16) {
      __m512 param = _mm512_mul_ps(_mm512_loadu_ps(pp + i), decay);
      __m512 grad = _mm512_mul_ps(_mm512_loadu_ps(gp + i), coef);
      __m512 first = _mm512_loadu_ps(mp + i);
      first = _mm512_fmadd_ps(weight, _mm512_sub_ps(grad, first),
                              lw < .5f ? first : grad);
      __m512 second =
          _mm512_fmadd_ps(_mm512_mul_ps(vw, grad), grad,
                          _mm512_mul_ps(_mm512_loadu_ps(vp + i), vb));
      __m512 denom =
          _mm512_add_ps(_mm512_div_ps(_mm512_sqrt_ps(second), sb), epsilon);
      param =
          _mm512_add_ps(param, _mm512_div_ps(_mm512_mul_ps(ss, first), denom));
      _mm512_storeu_ps(pp + i, param);
      _mm512_storeu_ps(mp + i, first);
      _mm512_storeu_ps(vp + i, second);
      store_gradient(gp + i, grad);
    }
  });
  for (const auto &value : {p, m, v, g})
    torch::autograd::impl::bump_version(value);
}
void update_seen(at::Tensor p, at::Tensor m, at::Tensor v, at::Tensor g,
                 double lr, double b1, double b2, double wd, double eps,
                 double step, double clipping, at::Tensor seen, int64_t width,
                 int64_t num_threads) {
  validate(p, m, v, g);
  ScopedThreads threads(num_threads);
  auto *pp = p.data_ptr<float>();
  auto *mp = m.data_ptr<float>();
  auto *vp = v.data_ptr<float>();
  auto *gp = g.data_ptr<float>();
  const __m512 decay = _mm512_set1_ps(float(1 - lr * wd));
  const float lw = float(1 - b1);
  const __m512 weight = _mm512_set1_ps(lw < .5f ? lw : lw - 1.f);
  const __m512 vb = _mm512_set1_ps(float(b2)),
               vw = _mm512_set1_ps(float(1 - b2));
  const __m512 sb = _mm512_set1_ps(float(std::sqrt(1 - std::pow(b2, step))));
  const __m512 epsilon = _mm512_set1_ps(float(eps));
  const __m512 ss = _mm512_set1_ps(float(-lr / (1 - std::pow(b1, step))));
  const __m512 coef = _mm512_set1_ps(float(clipping));
  TORCH_CHECK(p.dim() == 2 && width == p.size(1) && width > 0 &&
                  width % 16 == 0,
              "PLE AdamW requires a 16-aligned row width");
  TORCH_CHECK(seen.device().is_cpu() && seen.scalar_type() == at::kBool &&
                  seen.is_contiguous() && seen.numel() == p.size(0),
              "PLE AdamW row mask must contain one CPU bool per row");
  const bool *visited = seen.data_ptr<bool>();
  const bool zero_state_safe = std::isfinite(clipping) &&
                               !std::signbit(clipping) && eps > 0 && lr >= 0 &&
                               !std::signbit(lr);
  at::parallel_for(0, p.numel() / width, 0, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      if (!visited[row] && zero_state_safe) {
        for (int64_t i = row * width; i < (row + 1) * width; i += 16)
          _mm512_storeu_ps(pp + i,
                           _mm512_mul_ps(_mm512_loadu_ps(pp + i), decay));
        continue;
      }
      for (int64_t i = row * width; i < (row + 1) * width; i += 16) {
        __m512 param = _mm512_mul_ps(_mm512_loadu_ps(pp + i), decay);
        __m512 grad = _mm512_mul_ps(_mm512_loadu_ps(gp + i), coef);
        __m512 first = _mm512_loadu_ps(mp + i);
        first = _mm512_fmadd_ps(weight, _mm512_sub_ps(grad, first),
                                lw < .5f ? first : grad);
        __m512 second =
            _mm512_fmadd_ps(_mm512_mul_ps(vw, grad), grad,
                            _mm512_mul_ps(_mm512_loadu_ps(vp + i), vb));
        __m512 denom =
            _mm512_add_ps(_mm512_div_ps(_mm512_sqrt_ps(second), sb), epsilon);
        param = _mm512_add_ps(param,
                              _mm512_div_ps(_mm512_mul_ps(ss, first), denom));
        _mm512_storeu_ps(pp + i, param);
        _mm512_storeu_ps(mp + i, first);
        _mm512_storeu_ps(vp + i, second);
        store_gradient(gp + i, grad);
      }
    }
  });
  for (const auto &value : {p, m, v, g})
    torch::autograd::impl::bump_version(value);
}

void update_prepared(at::Tensor p, at::Tensor m, at::Tensor v, at::Tensor g,
                 double lr, double b1, double b2, double wd, double eps,
                 double step, double clipping, at::Tensor seen, int64_t width,
                 int64_t num_threads, at::Tensor prepared_p, at::Tensor prepared_m,
                 at::Tensor prepared_v, at::Tensor active, std::optional<at::Tensor> cache,
                 std::optional<at::Tensor> packed, std::optional<at::Tensor> active_rows, bool streaming) {
  validate(p, m, v, g);
  validate(p, prepared_p, prepared_m, prepared_v);
  TORCH_CHECK(active.device().is_cpu() && active.scalar_type() == at::kBool &&
                  active.is_contiguous() && active.numel() == p.size(0),
              "Prepared AdamW requires one active bool per row");
  at::BFloat16 *cache_ptr = nullptr;
  if (cache.has_value()) {
    TORCH_CHECK(cache->device().is_cpu() && cache->scalar_type() == at::kBFloat16 &&
                    cache->is_contiguous() && cache->sizes() == p.sizes(),
                "Prepared cache must be a matching contiguous CPU BF16 tensor");
    cache_ptr = cache->data_ptr<at::BFloat16>();
  }
  at::BFloat16 *packed_ptr = nullptr;
  const int64_t *active_ptr = nullptr;
  int64_t iterations = p.size(0);
  if (packed.has_value()) {
    TORCH_CHECK(packed->device().is_cpu() && packed->scalar_type() == at::kBFloat16 &&
                    packed->is_contiguous() && packed->sizes() == p.sizes() &&
                    active_rows.has_value() && active_rows->device().is_cpu() &&
                    active_rows->scalar_type() == at::kLong && active_rows->is_contiguous() && active_rows->dim() == 1,
                "Packed cache requires CPU BF16 storage and contiguous CPU int64 active rows");
    packed_ptr = packed->data_ptr<at::BFloat16>();
    active_ptr = active_rows->data_ptr<int64_t>();
    iterations = active_rows->numel();
    TORCH_CHECK(iterations <= p.size(0), "Too many prepared cache rows");
  }
  const auto *ap = active.data_ptr<bool>();
  const auto *np = prepared_p.data_ptr<float>();
  const auto *nm = prepared_m.data_ptr<float>();
  const auto *nv = prepared_v.data_ptr<float>();
  ScopedThreads threads(num_threads);
  auto *pp = p.data_ptr<float>();
  auto *mp = m.data_ptr<float>();
  auto *vp = v.data_ptr<float>();
  auto *gp = g.data_ptr<float>();
  const bool stream_state = streaming &&
      (reinterpret_cast<uintptr_t>(pp) % 64 == 0) &&
      (reinterpret_cast<uintptr_t>(mp) % 64 == 0) &&
      (reinterpret_cast<uintptr_t>(vp) % 64 == 0);
  const __m512 decay = _mm512_set1_ps(float(1 - lr * wd));
  const float lw = float(1 - b1);
  const __m512 weight = _mm512_set1_ps(lw < .5f ? lw : lw - 1.f);
  const __m512 vb = _mm512_set1_ps(float(b2)),
               vw = _mm512_set1_ps(float(1 - b2));
  const __m512 sb = _mm512_set1_ps(float(std::sqrt(1 - std::pow(b2, step))));
  const __m512 epsilon = _mm512_set1_ps(float(eps));
  const __m512 ss = _mm512_set1_ps(float(-lr / (1 - std::pow(b1, step))));
  const __m512 coef = _mm512_set1_ps(float(clipping));
  TORCH_CHECK(p.dim() == 2 && width == p.size(1) && width > 0 &&
                  width % 16 == 0,
              "PLE AdamW requires a 16-aligned row width");
  TORCH_CHECK(seen.device().is_cpu() && seen.scalar_type() == at::kBool &&
                  seen.is_contiguous() && seen.numel() == p.size(0),
              "PLE AdamW row mask must contain one CPU bool per row");
  at::parallel_for(0, iterations, 0, [&](int64_t begin, int64_t end) {
    for (int64_t ordinal = begin; ordinal < end; ++ordinal) {
      const int64_t row = active_ptr ? active_ptr[ordinal] : ordinal;
      TORCH_CHECK(row >= 0 && row < p.size(0), "Prepared cache row outside table");
      if (!ap[row]) continue;
      for (int64_t i = row * width; i < (row + 1) * width; i += 16) {
        __m512 param = _mm512_mul_ps(_mm512_loadu_ps(np + i), decay);
        __m512 grad = _mm512_mul_ps(_mm512_loadu_ps(gp + i), coef);
        __m512 first = _mm512_loadu_ps(nm + i);
        first = _mm512_fmadd_ps(weight, _mm512_sub_ps(grad, first),
                                lw < .5f ? first : grad);
        __m512 second =
            _mm512_fmadd_ps(_mm512_mul_ps(vw, grad), grad,
                            _mm512_mul_ps(_mm512_loadu_ps(nv + i), vb));
        __m512 denom =
            _mm512_add_ps(_mm512_div_ps(_mm512_sqrt_ps(second), sb), epsilon);
        param = _mm512_add_ps(param,
                              _mm512_div_ps(_mm512_mul_ps(ss, first), denom));
        if (stream_state) {
          _mm512_stream_ps(pp + i, param);
          _mm512_stream_ps(mp + i, first);
          _mm512_stream_ps(vp + i, second);
        } else {
          _mm512_storeu_ps(pp + i, param);
          _mm512_storeu_ps(mp + i, first);
          _mm512_storeu_ps(vp + i, second);
        }
        if (cache_ptr) store_bfloat16(cache_ptr + i, param, streaming);
        if (packed_ptr) store_bfloat16(packed_ptr + ordinal * width + i - row * width, param, streaming);
        store_gradient(gp + i, grad);
      }
    }
    if (streaming) _mm_sfence();
  });
  for (const auto &value : {p, m, v, g})
    torch::autograd::impl::bump_version(value);
  if (cache.has_value()) torch::autograd::impl::bump_version(*cache);
  if (packed.has_value()) torch::autograd::impl::bump_version(*packed);
}

std::tuple<at::Tensor, at::Tensor> unique_bounded(
    const at::Tensor &values, int64_t rows, bool return_inverse) {
  TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == at::kLong &&
                  values.is_contiguous() && values.dim() == 1 && rows > 0,
              "Bounded unique requires contiguous CPU int64 indices and positive rows");
  std::vector<int64_t> positions(rows, -1);
  const auto *input = values.data_ptr<int64_t>();
  for (int64_t i = 0; i < values.numel(); ++i) {
    const auto key = input[i];
    TORCH_CHECK(key >= 0 && key < rows, "Bounded unique index outside row range");
    positions[key] = 0;
  }
  int64_t count = 0;
  for (auto position : positions) count += position == 0;
  auto unique = at::empty({count}, values.options());
  auto *output = unique.data_ptr<int64_t>();
  int64_t cursor = 0;
  for (int64_t key = 0; key < rows; ++key) {
    if (positions[key] == 0) {
      output[cursor] = key;
      positions[key] = cursor++;
    }
  }
  auto inverse = at::empty({return_inverse ? values.numel() : 0}, values.options());
  if (return_inverse) {
    auto *mapping = inverse.data_ptr<int64_t>();
    at::parallel_for(0, values.numel(), 32768, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) mapping[i] = positions[input[i]];
    });
  }
  return {unique, inverse};
}

TORCH_LIBRARY(torchtitan_ple_cpu, m) {
  m.def("unique_bounded(Tensor values, int rows, bool return_inverse) -> (Tensor, Tensor)");
  m.def("set_worker_threads(int num_threads) -> int", &set_worker_threads);
  m.def("update_dense(Tensor(a!) p, Tensor(b!) m, Tensor(c!) v, Tensor(d!) g, "
        "float lr, float b1, float b2, float wd, float eps, float step, float "
        "clipping, int num_threads=0) -> ()");
  m.def("update_prepared(Tensor(a!) p, Tensor(b!) m, Tensor(c!) v, Tensor(d!) g, "
        "float lr, float b1, float b2, float wd, float eps, float step, float "
        "clipping, Tensor seen, int width, int num_threads, Tensor prepared_p, "
        "Tensor prepared_m, Tensor prepared_v, Tensor active, Tensor(e!)? cache=None, Tensor(f!)? packed=None, Tensor? active_rows=None, bool streaming=False) -> ()");
  m.def("update_seen(Tensor(a!) p, Tensor(b!) m, Tensor(c!) v, Tensor(d!) g, "
        "float lr, float b1, float b2, float wd, float eps, float step, float "
        "clipping, Tensor seen, int width, int num_threads=0) -> ()");
}
TORCH_LIBRARY_IMPL(torchtitan_ple_cpu, CPU, m) {
  m.impl("unique_bounded", unique_bounded);
  m.impl("update_dense", update_dense);
  m.impl("update_seen", update_seen);
  m.impl("update_prepared", update_prepared);
}
