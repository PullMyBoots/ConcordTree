// Exact tuple ordering when a three-child mixed-radix key exceeds uint64.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

inline bool repeat_tuple_space(uint64_t a, uint64_t b, uint64_t c, uint64_t& out) {
    const auto limit = std::numeric_limits<uint64_t>::max();
    if (a==0 || b==0 || c==0) { out=0; return true; }
    if (a > limit / b || a*b > limit / c) return false;
    out = a*b*c;
    return true;
}

template<class Get>
void repeat_tuple_order(std::vector<int32_t>& order, std::vector<int32_t>& temp,
                        int64_t length, Get get) {
    for (int64_t s=0; s<length; ++s) order[s]=int32_t(s);
    // LSD radix on each component preserves lexicographic order and the
    // original site order within equal tuples. Never hash or truncate keys.
    for (int component=2; component>=0; --component) {
        uint32_t maximum=0;
        for (int64_t s=0; s<length; ++s)
            maximum=std::max(maximum,uint32_t(get(s,component)));
        for (int shift=0; shift<32 && (maximum>>shift); shift+=8) {
            std::array<int64_t,256> count{};
            for (auto s:order) ++count[(uint32_t(get(s,component))>>shift)&255];
            int64_t cursor=0;
            for (auto& n:count) { auto old=n; n=cursor; cursor+=old; }
            for (auto s:order) temp[count[(uint32_t(get(s,component))>>shift)&255]++]=s;
            order.swap(temp);
        }
    }
}
