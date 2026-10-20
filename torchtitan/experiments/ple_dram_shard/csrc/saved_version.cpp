// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>

class SavedInputVersion : public torch::CustomClassHolder {
 public:
  explicit SavedInputVersion(at::Tensor tensor)
      : counter_(tensor.unsafeGetTensorImpl()->version_counter()),
        expected_(counter_.current_version()) {}
  void check() const {
    TORCH_CHECK(counter_.current_version() == expected_,
                "PLE saved input was modified in-place: expected version ",
                expected_, ", got ", counter_.current_version());
  }
  void attach(at::Tensor tensor) const {
    tensor.unsafeGetTensorImpl()->set_version_counter(counter_);
  }

 private:
  c10::VariableVersion counter_;
  uint32_t expected_;
};

TORCH_LIBRARY(torchtitan_ple_saved_input, m) {
  m.class_<SavedInputVersion>("SavedVersion")
      .def(torch::init<at::Tensor>())
      .def("check", &SavedInputVersion::check)
      .def("attach", &SavedInputVersion::attach);
}
