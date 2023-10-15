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
        torch::Tensor output = module.forward({text}).toTensor();
        return static_cast<TglangLanguage>(output.item<int>());
    } catch (const c10::Error& e) {
        return tglang_detect_programming_language_backup(text);
    }
}

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << tglang_detect_programming_language("input text") << std::endl;
    return 0;
}
