// Kernel level in C++: pack a ternary matrix, multiply it with int8 activations and check the result.
//
//     gemv [rows] [cols]          (defaults: 512 x 2048)
#include "quintrit.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const std::size_t rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 512;
    const std::size_t cols = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 2048;

    try {
        quintrit::check_abi();  // the loaded library must match the header this file was compiled against
        std::cout << "Quintrit " << quintrit::version() << "\n";

        // The best kernel of this CPU; errors are reported as quintrit::Error exceptions.
        const quintrit::Kernel kernel = quintrit::resolve_kernel(quintrit::Kernel::Auto);
        std::cout << "kernel: " << quintrit::kernel_name(kernel) << "\n";

        std::mt19937 gen(1234);
        std::uniform_int_distribution<int> trit(-1, 1), act(-128, 127);
        std::vector<std::int8_t> trits(rows * cols), activations(cols);
        for (auto& t : trits) t = static_cast<std::int8_t>(trit(gen));
        for (auto& a : activations) a = static_cast<std::int8_t>(act(gen));

        const std::vector<std::uint8_t> packed = quintrit::pack_matrix(kernel, trits, rows, cols);
        std::cout << rows * cols << " trits -> " << packed.size() << " bytes ("
                  << 8.0 * static_cast<double>(packed.size()) / static_cast<double>(rows * cols)
                  << " bits per weight)\n";

        const std::vector<std::int32_t> out = quintrit::gemv_i32(kernel, packed, rows, cols, activations);

        std::size_t mismatches = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            std::int64_t sum = 0;
            for (std::size_t c = 0; c < cols; ++c) {
                sum += static_cast<std::int64_t>(trits[r * cols + c]) * activations[c];
            }
            if (sum != out[r]) ++mismatches;
        }
        std::cout << (mismatches == 0 ? "OK: matches the integer product" : "MISMATCH") << " (" << mismatches
                  << " mismatching rows)\n";
        return mismatches == 0 ? 0 : 1;
    } catch (const quintrit::Error& e) {
        std::cerr << "quintrit error " << e.code() << ": " << e.what() << "\n";
        return 1;
    }
}
