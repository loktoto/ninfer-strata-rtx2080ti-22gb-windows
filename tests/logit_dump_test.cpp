#include "sage/logit_dump.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

int main() {
    const std::string path = "sage-logit-dump-test.bin";
    {
        sage::strata::LogitDump dump(path, 2, {0, 1}, {1});
        const float row[] = {1.0f, -2.0f};
        dump.append(row);
        dump.finish();
    }
    std::ifstream f(path, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), {});
    const std::vector<unsigned char> expected = {
        'S','A','G','E','L','O','G','1', 1,0,0,0, 2,0,0,0, 2,0,0,0, 1,0,0,0,
        0,0,0,0, 1,0,0,0, 1,0,0,0, 0,0,128,63, 0,0,0,192};
    assert(bytes == expected);
    bool rejected = false;
    try {
        sage::strata::LogitDump dump(path, 2, {0}, {1});
        const float row[] = {0, std::numeric_limits<float>::infinity()};
        dump.append(row);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try {
        sage::strata::LogitDump dump(path, 2, {2}, {1});
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try {
        sage::strata::LogitDump dump(path, 2, {0}, {1});
        dump.finish();
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::remove(path.c_str());
}
