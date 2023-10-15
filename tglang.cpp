#include "tglang.h"

#include <stdlib.h>
#include <string.h>

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

enum TglangLanguage tglang_detect_programming_language_backup(const char *text) {
  if (strstr(text, "std::") != NULL) {
    return TGLANG_LANGUAGE_CPLUSPLUS;
  }
  if (strstr(text, "let ") != NULL) {
    return TGLANG_LANGUAGE_JAVASCRIPT;
  }
  if (strstr(text, "int ") != NULL) {
    return TGLANG_LANGUAGE_C;
  }
  if (strstr(text, ";") == NULL) {
    return TGLANG_LANGUAGE_PYTHON;
  }
  return TGLANG_LANGUAGE_OTHER;
}

enum TglangLanguage tglang_detect_programming_language(const char *text) {
    torch::jit::script::Module module;
    try {
      module = torch::jit::load("../detect_model.pt");
    } catch (const c10::Error& e) {
      std::cerr << "Error loading the model: " << e.what() << std::endl;
      return tglang_detect_programming_language_backup(text);
    }

    std::vector<int64_t> sizes = {3, 4};
    torch::Tensor input = torch::randn(sizes, torch::kFloat);
    torch::Tensor output = module.forward({input}).toTensor();
    int predicted_language = 0;
    return static_cast<TglangLanguage>(predicted_language);
    return TGLANG_LANGUAGE_OTHER;
}

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << tglang_detect_programming_language("abcdefg") << std::endl;
    return 0;
}
