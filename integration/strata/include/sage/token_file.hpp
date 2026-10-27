#pragma once
#include <charconv>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sage::strata {
// Exact pretokenized input, independent of the Windows command-line length limit.
inline std::vector<int32_t> parse_token_file_text(const std::string& text) {
    std::vector<int32_t> ids;
    const char* p = text.data();
    const char* end = p + text.size();
    const auto space = [&] { while (p != end && std::isspace(static_cast<unsigned char>(*p))) ++p; };
    space();
    for (;;) {
        int32_t id = 0;
        const auto result = std::from_chars(p, end, id);
        if (result.ec != std::errc{} || result.ptr == p || id < 0)
            throw std::invalid_argument("token file requires nonnegative int32 IDs separated by commas");
        ids.push_back(id);
        if (ids.size() > 1048576) throw std::invalid_argument("token file exceeds 1048576 IDs");
        p = result.ptr;
        space();
        if (p == end) return ids;
        if (*p++ != ',') throw std::invalid_argument("token file requires comma separators");
        space();
    }
}

inline std::vector<int32_t> read_token_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open token file: " + path);
    std::string text;
    char chunk[8192];
    while (input.read(chunk, sizeof(chunk)) || input.gcount()) {
        if (text.size() + static_cast<size_t>(input.gcount()) > 16 * 1024 * 1024)
            throw std::invalid_argument("token file exceeds 16 MiB");
        text.append(chunk, static_cast<size_t>(input.gcount()));
    }
    if (!input.eof()) throw std::runtime_error("cannot read token file: " + path);
    return parse_token_file_text(text);
}
} // namespace sage::strata
