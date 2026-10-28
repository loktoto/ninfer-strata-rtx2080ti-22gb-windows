#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sage::strata {
// Version 1: all integers and IEEE-754 binary32 values are little endian.
class LogitDump {
    std::ofstream out_;
    uint32_t vocab_, steps_, written_ = 0;
    void u32(uint32_t n) {
        const char b[4] = {char(n), char(n >> 8), char(n >> 16), char(n >> 24)};
        out_.write(b, 4);
    }
public:
    LogitDump(const std::string& path, int vocab, const std::vector<int32_t>& prompt,
              const std::vector<int32_t>& follow) : vocab_(uint32_t(vocab)), steps_(uint32_t(follow.size())) {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "binary32 required");
        if (vocab <= 0 || vocab > 10000000 || prompt.empty() || prompt.size() > 1000000 ||
            follow.empty() || follow.size() > 1000000) throw std::runtime_error("invalid logit dump dimensions");
        for (int32_t id : prompt) if (id < 0 || id >= vocab) throw std::runtime_error("prompt token outside vocabulary");
        for (int32_t id : follow) if (id < 0 || id >= vocab) throw std::runtime_error("follow token outside vocabulary");
        out_.open(path, std::ios::binary | std::ios::trunc);
        if (!out_) throw std::runtime_error("cannot open logit dump: " + path);
        out_.write("SAGELOG1", 8);
        u32(1); u32(vocab_); u32(uint32_t(prompt.size())); u32(steps_);
        for (int32_t id : prompt) u32(uint32_t(id));
        for (int32_t id : follow) u32(uint32_t(id));
        if (!out_) throw std::runtime_error("cannot write logit dump header");
    }
    void append(const float* logits) {
        if (!logits || written_ >= steps_) throw std::runtime_error("too many logit rows");
        for (uint32_t i = 0; i < vocab_; ++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite logit");
            uint32_t bits; std::memcpy(&bits, logits + i, 4); u32(bits);
        }
        ++written_;
        if (!out_) throw std::runtime_error("cannot write logit row");
    }
    void finish() {
        if (written_ != steps_) throw std::runtime_error("incomplete logit dump");
        out_.flush();
        if (!out_) throw std::runtime_error("cannot flush logit dump");
        out_.close();
        if (out_.fail()) throw std::runtime_error("cannot close logit dump");
    }
};
}
