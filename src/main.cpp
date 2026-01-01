#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include <omp.h>

#include "qwen3.h"
#include "tokenizer.h"

void print_usage(const char* program) {
  std::cerr << "Usage: " << program << " <model_path> [options]\n";
  std::cerr << "  model_path: Path to Qwen3 model directory\n";
  std::cerr << "\nOptions:\n";
  std::cerr << "  -p, --prompt <text>    Text prompt (default: \"Hello, I am\")\n";
  std::cerr << "  -n, --max-tokens <N>   Maximum tokens to generate (default: 20)\n";
  std::cerr << "  -t, --threads <N>      Number of threads (default: all cores)\n";
  std::cerr << "  -h, --help             Show this help message\n";
  std::cerr << "\nLegacy positional arguments are also supported for backward compatibility.\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  // Handle --help as first argument
  std::string first_arg = argv[1];
  if (first_arg == "-h" || first_arg == "--help") {
    print_usage(argv[0]);
    return 0;
  }

  std::string model_path = argv[1];
  std::string prompt = "Hello, I am";
  size_t max_tokens = 20;
  int num_threads = omp_get_max_threads();

  // Parse optional arguments
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
      num_threads = std::stoi(argv[++i]);
      if (num_threads < 1) {
        std::cerr << "Error: Thread count must be at least 1\n";
        return 1;
      }
    } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
      prompt = argv[++i];
    } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
      max_tokens = std::stoul(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      // Legacy positional arguments for backward compatibility
      if (prompt == "Hello, I am") {
        prompt = arg;
      } else {
        max_tokens = std::stoul(arg);
      }
    }
  }

  // Set thread count
  omp_set_num_threads(num_threads);

  try {
    std::cout << "Using " << num_threads << " threads\n";
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
