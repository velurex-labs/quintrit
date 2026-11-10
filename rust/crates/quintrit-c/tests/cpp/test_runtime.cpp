// C++ test of the wrapper on a tiny synthetic model (argv[1]). Prints the greedy text as "TEXT_HEX=<hex>".
#include "quintrit.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond << "\n"; \
            ++failures;                                                                   \
        }                                                                                 \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf>\n";
        return 2;
    }
    const std::string path = argv[1];
    const std::string prompt = "<|user|>\nHello there\n<|assistant|>\n";

    try {
        // Format level.
        quintrit::Gguf gguf(path);
        CHECK(gguf.tensor_count() > 10);
        CHECK(gguf.metadata_string("general.architecture") == "llama");
        CHECK(gguf.metadata_u64("llama.context_length") == 48);
        std::string ternary;
        for (const quintrit::TensorInfo& t : gguf.tensors()) {
            if (t.is_ternary && ternary.empty()) ternary = t.name;
        }
        CHECK(!ternary.empty());
        const quintrit::TiledTensor tiled = gguf.convert_tensor(ternary, quintrit::Kernel::Auto);
        CHECK(tiled.data.size() == tiled.layout.total_bytes);
        CHECK(tiled.scale > 0.0f);
        try {
            gguf.find("does.not.exist");
            CHECK(false);
        } catch (const quintrit::Error& e) {
            CHECK(e.code() == QUINTRIT_ERR_NOT_FOUND);
        }

        // Runtime level.
        quintrit::Model model(path, 2);
        const quintrit_model_info_t info = model.info();
        CHECK(info.num_layers == 2 && std::string(info.architecture) == "llama");

        const std::vector<std::uint32_t> ids = model.tokenize("Hello");
        CHECK(ids.size() == 5);
        CHECK(model.detokenize(ids) == "Hello");
        CHECK(model.tokenize("Hello", true).size() == 6);
        CHECK(model.format_chat({{"user", "Hi"}}) == "### User:\nHi\n\n### Assistant:\n");

        quintrit::Session session = model.create_session();
        quintrit::GenerateOptions options;
        options.max_tokens = 8;

        std::string streamed;
        const quintrit::Generation g = session.generate(prompt, options, [&](const char* text, std::size_t len) {
            streamed.append(text, len);
            return true;
        });
        CHECK(streamed == g.text);
        CHECK(g.generated_tokens > 0);
        CHECK(g.prompt_tokens == prompt.size() + 1);
        std::printf("TEXT_HEX=");
        for (unsigned char c : g.text) std::printf("%02x", static_cast<unsigned>(c));
        std::printf("\n");

        // Same text again (the cache is restarted), and with a callback that stops early.
        CHECK(session.generate(prompt, options).text == g.text);
        int calls = 0;
        const quintrit::Generation stopped = session.generate(prompt, options, [&](const char*, std::size_t) { return ++calls < 2; });
        // A token that ends inside a UTF-8 character yields no piece of its own: two pieces can take more tokens.
        CHECK(stopped.finish_reason == QUINTRIT_FINISH_CALLBACK && calls == 2);
        CHECK(stopped.generated_tokens >= 2 && stopped.generated_tokens < options.max_tokens);

        // An exception thrown by the callback crosses the C boundary safely and reaches the caller.
        bool caught = false;
        try {
            session.generate(prompt, options, [](const char*, std::size_t) -> bool { throw std::runtime_error("from callback"); });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "from callback";
        }
        CHECK(caught);
        CHECK(session.generate(prompt, options).text == g.text);  // the session is still usable

        // Errors are exceptions.
        try {
            session.generate(std::string(10000, 'x'), options);
            CHECK(false);
        } catch (const quintrit::Error& e) {
            CHECK(e.code() == QUINTRIT_ERR_CONTEXT_OVERFLOW);
        }
        try {
            model.set_threads(1);  // a session exists
            CHECK(false);
        } catch (const quintrit::Error& e) {
            CHECK(e.code() == QUINTRIT_ERR_BUSY);
        }
    } catch (const std::exception& e) {
        std::cerr << "unexpected exception: " << e.what() << "\n";
        return 1;
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "C++ runtime test passed\n";
    return 0;
}
