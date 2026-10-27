#include "sage/token_file.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        const auto ids = sage::strata::parse_token_file_text(" 0, 42,2147483647\r\n");
        if (ids != std::vector<int32_t>{0, 42, 2147483647}) throw std::runtime_error("ID mismatch");
        for (const std::string bad : {"", " ", "1,", "-1", "2147483648", "1 2", "1,,2", "1x", "+1"}) {
            bool rejected = false;
            try { sage::strata::parse_token_file_text(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("accepted invalid input: " + bad);
        }
        std::string large = "123";
        for (int i = 1; i < 262144; ++i) large += ",123";
        if (sage::strata::parse_token_file_text(large).size() != 262144)
            throw std::runtime_error("long prompt truncated");
        std::cout << "PASS token files including 256K IDs\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
