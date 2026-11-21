// Runtime level in C++ on an NVIDIA GPU: load a model onto GPU 0 and stream generated text.
//
//     generate_gpu model.gguf ["prompt"]
//
// A library built without GPU support, or a machine without a supported GPU, makes Model::open_gpu throw quintrit::Error.
#include "quintrit.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " model.gguf [\"prompt\"]\n";
        return 2;
    }
    const std::string prompt = argc > 2 ? argv[2] : "<|user|>\nWhat is ternary computing?\n<|assistant|>\n";

    try {
        // The weights are transcoded into the GPU layout while the file is read; sessions keep their KV cache on the GPU.
        quintrit::Model model = quintrit::Model::open_gpu(argv[1], /*device=*/0);
        std::cout << "running on the GPU: " << (model.on_gpu() ? "yes" : "no") << "\n";

        quintrit::Session session = model.create_session();
        quintrit::GenerateOptions options;  // greedy by default
        options.max_tokens = 64;
        const quintrit::Generation result = session.generate(prompt, options, [](const char* text, std::size_t len) {
            std::cout.write(text, static_cast<std::streamsize>(len)).flush();
            return true;
        });
        std::cout << "\n[" << result.prompt_tokens << " prompt tokens, " << result.generated_tokens << " generated]\n";
        return 0;
    } catch (const quintrit::Error& e) {
        std::cerr << "quintrit error " << e.code() << ": " << e.what() << "\n";
        return 1;
    }
}
