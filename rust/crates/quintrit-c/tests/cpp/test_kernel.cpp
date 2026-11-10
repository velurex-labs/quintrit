// C++ test of the quintrit.hpp wrapper: kernel level (pack / GEMV against a brute-force product), error handling through
// exceptions, and the RAII classes on files that do not exist. Exit code 0 means success.
#include "quintrit.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond << "\n";   \
            ++failures;                                                                     \
        }                                                                                   \
    } while (0)

#define CHECK_THROWS_CODE(expr, expected_code)                                              \
    do {                                                                                    \
        bool threw_ = false;                                                                \
        try {                                                                               \
            expr;                                                                           \
        } catch (const quintrit::Error& e) {                                                \
            threw_ = true;                                                                  \
            if (e.code() != (expected_code)) {                                              \
                std::cerr << __FILE__ << ":" << __LINE__ << ": code " << e.code()            \
                          << " instead of " << (expected_code) << ": " << e.what() << "\n"; \
                ++failures;                                                                 \
            }                                                                               \
            if (std::string(e.what()).empty()) {                                            \
                std::cerr << __FILE__ << ":" << __LINE__ << ": empty message\n";            \
                ++failures;                                                                 \
            }                                                                               \
        }                                                                                   \
        if (!threw_) {                                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << ": no exception from " #expr "\n";  \
            ++failures;                                                                     \
        }                                                                                   \
    } while (0)

std::uint64_t state = 0x13198A2E03707344ULL;
std::uint32_t next() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return static_cast<std::uint32_t>((state * 0x2545F4914F6CDD1DULL) >> 32);
}

void gemv_case(quintrit::Kernel kernel, std::size_t rows, std::size_t cols, std::uint32_t threads) {
    std::vector<std::int8_t> trits(rows * cols);
    for (auto& t : trits) t = static_cast<std::int8_t>(static_cast<int>(next() % 3) - 1);
    std::vector<std::int8_t> acts(cols);
    for (auto& a : acts) a = static_cast<std::int8_t>(next() & 0xFF);
    acts[0] = -128;

    const std::vector<std::uint8_t> tiled = quintrit::pack_matrix(kernel, trits, rows, cols);
    CHECK(quintrit::unpack_matrix(kernel, tiled, rows, cols) == trits);

    const std::vector<std::int32_t> got = quintrit::gemv_i32(kernel, tiled, rows, cols, acts, threads);
    const std::vector<float> gotf = quintrit::gemv_f32(kernel, tiled, rows, cols, acts, 2.0f, threads);
    CHECK(got.size() == rows && gotf.size() == rows);
    for (std::size_t r = 0; r < rows; ++r) {
        std::int64_t sum = 0;
        for (std::size_t c = 0; c < cols; ++c) sum += static_cast<std::int64_t>(trits[r * cols + c]) * acts[c];
        CHECK(got[r] == sum);
        CHECK(gotf[r] == static_cast<float>(sum) * 2.0f);
    }
}

}  // namespace

int main() {
    quintrit::check_abi();
    std::cout << "quintrit " << quintrit::version() << ", best kernel "
              << quintrit::kernel_name(quintrit::supported_kernels().front()) << "\n";
    CHECK(quintrit::version() == QUINTRIT_VERSION_STRING);
    CHECK(quintrit::kernel_supported(quintrit::Kernel::Scalar));
    CHECK(quintrit::supported_kernels().back() == quintrit::Kernel::Scalar);
    CHECK(quintrit::resolve_kernel(quintrit::Kernel::Auto) == quintrit::supported_kernels().front());
    (void)quintrit::cpu_features();

    const std::size_t shapes[][2] = {{1, 1}, {5, 7}, {33, 161}, {64, 320}, {70, 1000}, {96, 3072}};
    for (quintrit::Kernel k : quintrit::supported_kernels()) {
        for (const auto& s : shapes) {
            gemv_case(k, s[0], s[1], 1);
            gemv_case(k, s[0], s[1], 3);
        }
    }
    gemv_case(quintrit::Kernel::Auto, 100, 777, 0);

    // Errors become quintrit::Error with the status code and the message of the library.
    CHECK_THROWS_CODE(quintrit::pack_matrix(quintrit::Kernel::Simd32, std::vector<std::int8_t>(10), 4, 4), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    CHECK_THROWS_CODE(quintrit::pack_matrix(quintrit::Kernel::Simd32, std::vector<std::int8_t>(16, 5), 4, 4), QUINTRIT_ERR_INVALID_VALUE);
    CHECK_THROWS_CODE(quintrit::tiled_layout(quintrit::Kernel::Simd32, 0, 4), QUINTRIT_ERR_INVALID_SHAPE);
    CHECK_THROWS_CODE(quintrit::kernel_supported(static_cast<quintrit::Kernel>(99)), QUINTRIT_ERR_INVALID_ARGUMENT);
    CHECK_THROWS_CODE(quintrit::Gguf("this-file-does-not-exist.gguf"), QUINTRIT_ERR_IO);
    CHECK_THROWS_CODE(quintrit::Model("this-file-does-not-exist.gguf"), QUINTRIT_ERR_IO);

    // The wrapper classes are move-only.
    static_assert(std::is_move_constructible<quintrit::Model>::value, "Model must be movable");
    static_assert(!std::is_copy_constructible<quintrit::Model>::value, "Model must not be copyable");
    static_assert(std::is_move_constructible<quintrit::Session>::value, "Session must be movable");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "C++ test passed\n";
    return 0;
}
