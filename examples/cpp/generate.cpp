// Runtime level in C++: load a ternary GGUF model and stream generated text to the terminal.
//
//     generate model.gguf ["prompt"]
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
        quintrit::Model model(argv[1]);  // transcodes the weights into RAM; silent unless verbose is requested
        const quintrit_model_info_t info = model.info();
        std::cout << info.architecture << " model, " << info.num_layers << " layers, "
                  << quintrit::kernel_name(static_cast<quintrit::Kernel>(info.kernel)) << " kernel\n";

        quintrit::Session session = model.create_session();
        quintrit::GenerateOptions options;  // greedy by default
        options.max_tokens = 64;

        // The callback receives pieces of UTF-8 text; return false to stop.
        const quintrit::Generation result = session.generate(prompt, options, [](const char* text, std::size_t len) {
            std::cout.write(text, static_cast<std::streamsize>(len)).flush();
            return true;
        });
        std::cout << "\n[" << result.prompt_tokens << " prompt tokens, " << result.generated_tokens
                  << " generated]\n";
        return 0;
    } catch (const quintrit::Error& e) {
        std::cerr << "quintrit error " << e.code() << ": " << e.what() << "\n";
        return 1;
    }
}
