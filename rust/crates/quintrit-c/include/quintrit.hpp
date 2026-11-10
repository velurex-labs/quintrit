// quintrit.hpp - header-only C++11 wrapper around the Quintrit C API (quintrit.h).
//
// Design
//   - RAII classes own the opaque handles: quintrit::Gguf, quintrit::Model, quintrit::Session. They are movable and
//     not copyable; the destructors free the handle and never throw.
//   - Errors are reported with exceptions: every failing C call throws quintrit::Error, which carries the status code
//     (code()) and the message of quintrit_last_error_message(). Nothing returns a status the caller could forget to
//     check. Code that must not throw can use the C API directly.
//   - The wrapper allocates the output buffers (std::vector, std::string) so that the size-query dance of the C API
//     stays internal.
//   - The streaming callback of Session::generate may throw; the exception is carried across the C boundary (which
//     must not see it) and rethrown to the caller after the generation has stopped.
//
// Thread safety is that of the C API: a Session is used by one thread at a time, a Model may be shared.
#ifndef QUINTRIT_HPP
#define QUINTRIT_HPP

#include "quintrit.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quintrit {

/// Thrown when a call of the C API fails.
class Error : public std::runtime_error {
public:
    Error(int code, const std::string& message) : std::runtime_error(message), code_(code) {}
    /// One of the QUINTRIT_ERR_* status codes.
    int code() const noexcept { return code_; }

private:
    int code_;
};

namespace detail {

[[noreturn]] inline void throw_last(int status) {
    const char* msg = quintrit_last_error_message();
    std::string text = (msg != nullptr && msg[0] != '\0') ? msg : quintrit_status_string(status);
    throw Error(status, text);
}

/// Throws quintrit::Error unless `status` is QUINTRIT_OK.
inline void check(int status) {
    if (status != QUINTRIT_OK) {
        throw_last(status);
    }
}

/// State of one streaming generation: collects the text, forwards pieces to the user callback and carries exceptions.
struct Trampoline {
    const std::function<bool(const char*, std::size_t)>* fn = nullptr;
    std::string text;
    std::exception_ptr error;
};

/// C callback that feeds a Trampoline; never lets an exception escape into the C library.
inline int stream_thunk(const char* text, std::size_t len, void* user) {
    Trampoline* t = static_cast<Trampoline*>(user);
    try {
        t->text.append(text, len);
        if (t->fn != nullptr && !(*t->fn)(text, len)) {
            return 1;
        }
        return 0;
    } catch (...) {
        t->error = std::current_exception();
        return 1;
    }
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------------------------
// Library and kernel queries
// -------------------------------------------------------------------------------------------------------------------

/// Kernel ids as a typed enumeration (values equal the QUINTRIT_KERNEL_* constants).
enum class Kernel : std::int32_t {
    Auto = QUINTRIT_KERNEL_AUTO,
    Scalar = QUINTRIT_KERNEL_SCALAR,
    Simd32 = QUINTRIT_KERNEL_SIMD32,
    Avx512 = QUINTRIT_KERNEL_AVX512,
};

inline std::string version() { return quintrit_version(); }

/// Throws if the loaded library does not match the header this file was compiled against.
inline void check_abi() {
    if (quintrit_abi_version() != QUINTRIT_ABI_VERSION) {
        throw Error(QUINTRIT_ERR_INVALID_ARGUMENT, "quintrit library ABI version differs from the header");
    }
}

inline std::uint64_t cpu_features() {
    std::uint64_t mask = 0;
    detail::check(quintrit_cpu_features(&mask));
    return mask;
}

inline bool kernel_supported(Kernel k) {
    std::int32_t s = 0;
    detail::check(quintrit_kernel_supported(static_cast<std::int32_t>(k), &s));
    return s != 0;
}

inline Kernel resolve_kernel(Kernel k) {
    std::int32_t r = 0;
    detail::check(quintrit_kernel_resolve(static_cast<std::int32_t>(k), &r));
    return static_cast<Kernel>(r);
}

inline const char* kernel_name(Kernel k) {
    const char* n = quintrit_kernel_name(static_cast<std::int32_t>(k));
    return n != nullptr ? n : "unknown";
}

/// Kernels usable on this CPU, best first.
inline std::vector<Kernel> supported_kernels() {
    std::size_t n = 0;
    int st = quintrit_supported_kernels(nullptr, 0, &n);
    if (st != QUINTRIT_OK && st != QUINTRIT_ERR_BUFFER_TOO_SMALL) {
        detail::throw_last(st);
    }
    std::vector<std::int32_t> raw(n);
    detail::check(quintrit_supported_kernels(raw.data(), raw.size(), &n));
    std::vector<Kernel> out;
    for (std::int32_t v : raw) {
        out.push_back(static_cast<Kernel>(v));
    }
    return out;
}

// -------------------------------------------------------------------------------------------------------------------
// Kernel level
// -------------------------------------------------------------------------------------------------------------------

inline quintrit_tiled_layout_t tiled_layout(Kernel k, std::size_t rows, std::size_t cols) {
    quintrit_tiled_layout_t l;
    l.struct_size = sizeof(l);
    detail::check(quintrit_tiled_layout(static_cast<std::int32_t>(k), rows, cols, &l));
    return l;
}

/// Packs a row-major rows x cols matrix of trits (-1, 0, +1) into the tiled layout of `k`.
inline std::vector<std::uint8_t> pack_matrix(Kernel k, const std::vector<std::int8_t>& trits, std::size_t rows,
                                             std::size_t cols) {
    std::size_t bytes = 0;
    detail::check(quintrit_tiled_size(static_cast<std::int32_t>(k), rows, cols, &bytes));
    std::vector<std::uint8_t> out(bytes);
    detail::check(quintrit_pack_matrix(static_cast<std::int32_t>(k), trits.data(), trits.size(), rows, cols, out.data(),
                                       out.size()));
    return out;
}

inline std::vector<std::int8_t> unpack_matrix(Kernel k, const std::vector<std::uint8_t>& tiled, std::size_t rows,
                                              std::size_t cols) {
    std::vector<std::int8_t> out(rows * cols);
    detail::check(quintrit_unpack_matrix(static_cast<std::int32_t>(k), tiled.data(), tiled.size(), rows, cols, out.data(),
                                         out.size()));
    return out;
}

/// out[i] = sum_j W[i][j] * activations[j], exact.
inline std::vector<std::int32_t> gemv_i32(Kernel k, const std::vector<std::uint8_t>& tiled, std::size_t rows,
                                          std::size_t cols, const std::vector<std::int8_t>& activations,
                                          std::uint32_t threads = 1) {
    std::vector<std::int32_t> out(rows);
    detail::check(quintrit_gemv_i32(static_cast<std::int32_t>(k), tiled.data(), tiled.size(), rows, cols,
                                    activations.data(), activations.size(), out.data(), out.size(), threads));
    return out;
}

/// out[i] = scale * sum_j W[i][j] * activations[j].
inline std::vector<float> gemv_f32(Kernel k, const std::vector<std::uint8_t>& tiled, std::size_t rows, std::size_t cols,
                                   const std::vector<std::int8_t>& activations, float scale = 1.0f,
                                   std::uint32_t threads = 1) {
    std::vector<float> out(rows);
    detail::check(quintrit_gemv_f32(static_cast<std::int32_t>(k), tiled.data(), tiled.size(), rows, cols,
                                    activations.data(), activations.size(), scale, out.data(), out.size(), threads));
    return out;
}

// -------------------------------------------------------------------------------------------------------------------
// Format level
// -------------------------------------------------------------------------------------------------------------------

/// Description of a tensor of a GGUF file (value type; independent of the file handle).
struct TensorInfo {
    std::string name;
    std::string type_name;
    std::int32_t ggml_type = 0;
    std::vector<std::uint64_t> shape;  // as stored: shape[0] is the innermost dimension
    std::uint64_t num_elements = 0;
    std::uint64_t data_bytes = 0;
    bool is_ternary = false;
};

/// A converted ternary tensor.
struct TiledTensor {
    quintrit_tiled_layout_t layout;
    float scale = 1.0f;
    std::vector<std::uint8_t> data;
};

/// An open GGUF file.
class Gguf {
public:
    explicit Gguf(const std::string& path) {
        quintrit_gguf_t* h = nullptr;
        detail::check(quintrit_gguf_open(path.c_str(), &h));
        handle_.reset(h);
    }

    std::size_t tensor_count() const {
        quintrit_gguf_info_t info;
        info.struct_size = sizeof(info);
        detail::check(quintrit_gguf_info(handle_.get(), &info));
        return static_cast<std::size_t>(info.tensor_count);
    }

    TensorInfo tensor(std::size_t index) const {
        quintrit_tensor_info_t t;
        t.struct_size = sizeof(t);
        detail::check(quintrit_gguf_tensor_at(handle_.get(), index, &t));
        return convert(t);
    }

    TensorInfo find(const std::string& name) const {
        quintrit_tensor_info_t t;
        t.struct_size = sizeof(t);
        detail::check(quintrit_gguf_tensor_find(handle_.get(), name.c_str(), &t));
        return convert(t);
    }

    std::vector<TensorInfo> tensors() const {
        std::vector<TensorInfo> out;
        const std::size_t n = tensor_count();
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(tensor(i));
        }
        return out;
    }

    std::string metadata_string(const std::string& key) const {
        std::size_t len = 0;
        int st = quintrit_gguf_metadata_string(handle_.get(), key.c_str(), nullptr, 0, &len);
        if (st != QUINTRIT_OK && st != QUINTRIT_ERR_BUFFER_TOO_SMALL) {
            detail::throw_last(st);
        }
        std::string out(len + 1, '\0');
        detail::check(quintrit_gguf_metadata_string(handle_.get(), key.c_str(), &out[0], out.size(), &len));
        out.resize(len);
        return out;
    }

    std::uint64_t metadata_u64(const std::string& key) const {
        std::uint64_t v = 0;
        detail::check(quintrit_gguf_metadata_u64(handle_.get(), key.c_str(), &v));
        return v;
    }

    /// Converts a ternary tensor into the tiled layout of `k`.
    TiledTensor convert_tensor(const std::string& name, Kernel k) const {
        TiledTensor t;
        t.layout.struct_size = sizeof(t.layout);
        detail::check(quintrit_gguf_tensor_layout(handle_.get(), name.c_str(), static_cast<std::int32_t>(k), &t.layout,
                                                  &t.scale));
        t.data.resize(static_cast<std::size_t>(t.layout.total_bytes));
        detail::check(quintrit_gguf_convert_tensor(handle_.get(), name.c_str(), static_cast<std::int32_t>(k),
                                                   t.data.data(), t.data.size(), &t.layout, &t.scale));
        return t;
    }

    quintrit_gguf_t* get() const noexcept { return handle_.get(); }

private:
    static TensorInfo convert(const quintrit_tensor_info_t& t) {
        TensorInfo i;
        i.name = t.name;
        i.type_name = t.type_name;
        i.ggml_type = t.ggml_type;
        for (std::uint32_t d = 0; d < t.n_dims && d < QUINTRIT_MAX_DIMS; ++d) {
            i.shape.push_back(t.shape[d]);
        }
        i.num_elements = t.num_elements;
        i.data_bytes = t.data_bytes;
        i.is_ternary = t.is_ternary != 0;
        return i;
    }

    struct Deleter {
        void operator()(quintrit_gguf_t* h) const noexcept { quintrit_gguf_close(h); }
    };
    std::unique_ptr<quintrit_gguf_t, Deleter> handle_;
};

// -------------------------------------------------------------------------------------------------------------------
// Runtime level
// -------------------------------------------------------------------------------------------------------------------

/// Generation settings (defaults: greedy, 128 tokens).
struct GenerateOptions {
    std::uint32_t max_tokens = 128;
    float temperature = 0.0f;  // 0 = greedy
    std::uint32_t top_k = 1;
    float top_p = 1.0f;
    std::uint64_t seed = 0;
    bool keep_context = false;
    std::vector<std::string> stop_sequences;
};

struct Generation {
    std::string text;
    int finish_reason = QUINTRIT_FINISH_LENGTH;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t generated_tokens = 0;
};

/// Streaming callback: return false to stop. May throw; the exception is rethrown by Session::generate.
using TokenCallback = std::function<bool(const char* text, std::size_t len)>;

class Model;

/// A conversation with a model. Not thread-safe.
class Session {
public:
    Generation generate(const std::string& prompt, const GenerateOptions& options = GenerateOptions(),
                        const TokenCallback& callback = TokenCallback()) {
        quintrit_generate_params_t p;
        detail::check(quintrit_generate_params_default(&p));
        p.flags = options.keep_context ? QUINTRIT_GENERATE_KEEP_CONTEXT : 0;
        p.max_tokens = options.max_tokens;
        p.temperature = options.temperature;
        p.top_k = options.top_k;
        p.top_p = options.top_p;
        p.seed = options.seed;
        std::vector<const char*> stops;
        for (const std::string& s : options.stop_sequences) {
            stops.push_back(s.c_str());
        }
        p.stop_sequences = stops.empty() ? nullptr : stops.data();
        p.stop_count = stops.size();

        detail::Trampoline tramp;
        tramp.fn = callback ? &callback : nullptr;

        Generation g;
        quintrit_generate_result_t r;
        r.struct_size = sizeof(r);
        detail::check(quintrit_session_generate(handle_.get(), prompt.data(), prompt.size(), &p, &detail::stream_thunk,
                                                &tramp, nullptr, 0, nullptr, &r));
        if (tramp.error) {
            std::rethrow_exception(tramp.error);
        }
        g.text = std::move(tramp.text);
        g.finish_reason = r.finish_reason;
        g.prompt_tokens = r.prompt_tokens;
        g.generated_tokens = r.generated_tokens;
        return g;
    }

    /// Forgets the conversation.
    void reset() { detail::check(quintrit_session_reset(handle_.get())); }

    quintrit_session_info_t info() const {
        quintrit_session_info_t i;
        i.struct_size = sizeof(i);
        detail::check(quintrit_session_info(handle_.get(), &i));
        return i;
    }

    quintrit_session_t* get() const noexcept { return handle_.get(); }

private:
    friend class Model;
    explicit Session(quintrit_session_t* h) : handle_(h) {}

    struct Deleter {
        void operator()(quintrit_session_t* h) const noexcept { quintrit_session_free(h); }
    };
    std::unique_ptr<quintrit_session_t, Deleter> handle_;
};

/// A loaded model. May be shared by several threads (each with its own Session).
class Model {
public:
    /// Loads a GGUF model. `threads` = 0 selects automatic; `kernel` = Kernel::Auto selects the best. Loading is
    /// silent; pass `verbose = true` to print progress lines to the standard output.
    explicit Model(const std::string& path, std::uint32_t threads = 0, Kernel kernel = Kernel::Auto, bool verbose = false) {
        quintrit_model_options_t o;
        o.struct_size = sizeof(o);
        o.num_threads = threads;
        o.kernel = static_cast<std::int32_t>(kernel);
        o.flags = verbose ? QUINTRIT_MODEL_OPEN_VERBOSE : 0;
        quintrit_model_t* h = nullptr;
        detail::check(quintrit_model_open(path.c_str(), &o, &h));
        handle_.reset(h);
    }

    quintrit_model_info_t info() const {
        quintrit_model_info_t i;
        i.struct_size = sizeof(i);
        detail::check(quintrit_model_info(handle_.get(), &i));
        return i;
    }

    void set_threads(std::uint32_t threads) { detail::check(quintrit_model_set_threads(handle_.get(), threads)); }

    std::vector<std::uint32_t> tokenize(const std::string& text, bool add_bos = false) const {
        std::size_t n = 0;
        const std::uint32_t flags = add_bos ? QUINTRIT_TOKENIZE_ADD_BOS : 0;
        int st = quintrit_model_tokenize(handle_.get(), text.data(), text.size(), flags, nullptr, 0, &n);
        if (st != QUINTRIT_OK && st != QUINTRIT_ERR_BUFFER_TOO_SMALL) {
            detail::throw_last(st);
        }
        std::vector<std::uint32_t> out(n);
        detail::check(quintrit_model_tokenize(handle_.get(), text.data(), text.size(), flags, out.data(), out.size(), &n));
        return out;
    }

    std::string detokenize(const std::vector<std::uint32_t>& tokens) const {
        std::size_t n = 0;
        int st = quintrit_model_detokenize(handle_.get(), tokens.data(), tokens.size(), nullptr, 0, &n);
        if (st != QUINTRIT_OK && st != QUINTRIT_ERR_BUFFER_TOO_SMALL) {
            detail::throw_last(st);
        }
        std::string out(n + 1, '\0');
        detail::check(quintrit_model_detokenize(handle_.get(), tokens.data(), tokens.size(), &out[0], out.size(), &n));
        out.resize(n);
        return out;
    }

    /// Formats {role, content} pairs into the prompt format of the model.
    std::string format_chat(const std::vector<std::pair<std::string, std::string>>& messages) const {
        std::vector<quintrit_chat_message_t> raw;
        for (const auto& m : messages) {
            quintrit_chat_message_t c;
            c.role = m.first.c_str();
            c.content = m.second.c_str();
            raw.push_back(c);
        }
        std::size_t n = 0;
        int st = quintrit_model_format_chat(handle_.get(), raw.data(), raw.size(), nullptr, 0, &n);
        if (st != QUINTRIT_OK && st != QUINTRIT_ERR_BUFFER_TOO_SMALL) {
            detail::throw_last(st);
        }
        std::string out(n + 1, '\0');
        detail::check(quintrit_model_format_chat(handle_.get(), raw.data(), raw.size(), &out[0], out.size(), &n));
        out.resize(n);
        return out;
    }

    /// Creates a session; `context_length` = 0 selects the model's maximum.
    Session create_session(std::uint32_t context_length = 0) const {
        quintrit_session_options_t o;
        o.struct_size = sizeof(o);
        o.context_length = context_length;
        quintrit_session_t* s = nullptr;
        detail::check(quintrit_session_create(handle_.get(), &o, &s));
        return Session(s);
    }

    quintrit_model_t* get() const noexcept { return handle_.get(); }

private:
    struct Deleter {
        void operator()(quintrit_model_t* h) const noexcept { quintrit_model_free(h); }
    };
    std::unique_ptr<quintrit_model_t, Deleter> handle_;
};

}  // namespace quintrit

#endif  // QUINTRIT_HPP
