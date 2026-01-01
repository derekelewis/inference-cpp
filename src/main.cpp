#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "qwen3.h"
#include "tokenizer.h"

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " <model_path> [prompt] [max_tokens]\n";
  std::cerr << "  model_path: Path to Qwen3 model directory\n";
  std::cerr << "  prompt: Text prompt (default: \"Hello, I am\")\n";
  std::cerr << "  max_tokens: Maximum tokens to generate (default: 20)\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  std::string prompt = (argc > 2) ? argv[2] : "Hello, I am";
  size_t max_tokens = (argc > 3) ? std::stoul(argv[3]) : 20;

  try {
    std::cout << "Loading model from: " << model_path << std::endl;

    auto load_start = std::chrono::high_resolution_clock::now();
    Qwen3Model model(model_path);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start).count();
    std::cout << "Model loaded in " << load_ms << " ms" << std::endl;

    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer(model_path + "/tokenizer.json");
    std::cout << "Tokenizer loaded (vocab size: " << tokenizer.vocab_size() << ")\n";

    // Encode prompt
    std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;
    std::vector<uint32_t> prompt_tokens = tokenizer.encode(prompt);
    std::cout << "Prompt tokens (" << prompt_tokens.size() << "): ";
    for (uint32_t tok : prompt_tokens) {
      std::cout << tok << " ";
    }
    std::cout << std::endl;

    // Generate
    std::cout << "\nGenerating " << max_tokens << " tokens..." << std::endl;
    auto gen_start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> output = model.generate(prompt_tokens, max_tokens, 0.0f);
    auto gen_end = std::chrono::high_resolution_clock::now();
    auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        gen_end - gen_start).count();

    size_t tokens_generated = output.size() - prompt_tokens.size();
    float tokens_per_sec = tokens_generated * 1000.0f / gen_ms;

    std::cout << "Generated " << tokens_generated << " tokens in "
              << gen_ms << " ms (" << tokens_per_sec << " tok/s)\n";

    // Decode and print output
    std::cout << "\nOutput tokens: ";
    for (uint32_t tok : output) {
      std::cout << tok << " ";
    }
    std::cout << std::endl;

    std::string decoded = tokenizer.decode(output);
    std::cout << "\nGenerated text:\n" << decoded << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
