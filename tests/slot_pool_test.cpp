#include "sage/slot_pool.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using sage::MemoryBudget;
using sage::PolicyStatus;
using sage::SlotHandle;
using sage::SlotPool;
using sage::SlotState;

void check(bool condition, const char* expression, int line) {
    if (!condition)
        throw std::runtime_error("line " + std::to_string(line) + ": " + expression);
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

SlotHandle ready(SlotPool& pool, std::size_t index, std::uint64_t token = 7) {
    const auto load = pool.begin_load(index, 64, token);
    CHECK(load);
    CHECK(pool.confirm_copy(load.value, token) == PolicyStatus::ok);
    CHECK(pool.publish(load.value) == PolicyStatus::ok);
    return load.value;
}

void delayed_copy() {
    const std::array<std::uint64_t, 1> capacities{128};
    SlotPool pool(capacities);
    const auto load = pool.begin_load(0, 96, 41);
    CHECK(load);
    CHECK(pool.snapshot(0)->state == SlotState::loading);
    CHECK(pool.snapshot(0)->copy_pending);
    CHECK(pool.publish(load.value) == PolicyStatus::copy_pending);
    CHECK(pool.evict(load.value) == PolicyStatus::copy_pending);
    CHECK(pool.acquire_reader(load.value).status == PolicyStatus::invalid_state);
    CHECK(pool.begin_load(0, 32, 42).status == PolicyStatus::invalid_state);
    CHECK(pool.confirm_copy(load.value, 42) == PolicyStatus::invalid_token);
    CHECK(pool.confirm_copy(load.value, 0) == PolicyStatus::invalid_token);
    CHECK(pool.snapshot(0)->copy_pending);
    CHECK(pool.confirm_copy(load.value, 41) == PolicyStatus::ok);
    CHECK(!pool.snapshot(0)->copy_pending);
    CHECK(pool.snapshot(0)->state == SlotState::loading);
    CHECK(pool.acquire_reader(load.value).status == PolicyStatus::invalid_state);
    CHECK(pool.confirm_copy(load.value, 41) == PolicyStatus::invalid_state);
    CHECK(pool.publish(load.value) == PolicyStatus::ok);
    CHECK(pool.snapshot(0)->state == SlotState::ready);
    CHECK(pool.snapshot(0)->content_bytes == 96);
    CHECK(pool.publish(load.value) == PolicyStatus::invalid_state);
}

void stale_generation() {
    const std::array<std::uint64_t, 1> capacities{128};
    SlotPool pool(capacities);
    const auto old = ready(pool, 0);
    const auto old_lease = pool.acquire_reader(old);
    CHECK(old_lease);
    CHECK(pool.release_reader(old_lease.value) == PolicyStatus::ok);
    CHECK(pool.evict(old) == PolicyStatus::ok);
    CHECK(pool.snapshot(0)->state == SlotState::empty);
    CHECK(pool.snapshot(0)->content_bytes == 0);
    const auto next = pool.begin_load(0, 128, 7); // Reusing a transfer ID is safe only with its handle.
    CHECK(next);
    CHECK(next.value.generation != old.generation);
    CHECK(pool.confirm_copy(old, 7) == PolicyStatus::stale_handle);
    CHECK(pool.publish(old) == PolicyStatus::stale_handle);
    CHECK(pool.evict(old) == PolicyStatus::stale_handle);
    CHECK(pool.acquire_reader(old).status == PolicyStatus::stale_handle);
    CHECK(pool.release_reader(old_lease.value) == PolicyStatus::stale_handle);
    CHECK(pool.snapshot(0)->copy_pending);
    CHECK(pool.confirm_copy(next.value, 7) == PolicyStatus::ok);
    CHECK(pool.evict(next.value) == PolicyStatus::ok); // Completed unpublished load can be discarded.
}

void pinned_readers_and_double_release() {
    const std::array<std::uint64_t, 2> capacities{128, 128};
    SlotPool pool(capacities);
    const auto handle = ready(pool, 0);
    const auto other = ready(pool, 1);
    const auto first = pool.acquire_reader(handle);
    const auto second = pool.acquire_reader(handle);
    const auto elsewhere = pool.acquire_reader(other);
    CHECK(first && second && elsewhere);
    CHECK(first.value.token != second.value.token);
    CHECK(pool.snapshot(0)->readers == 2);
    CHECK(pool.evict(handle) == PolicyStatus::readers_active);
    CHECK(pool.begin_load(0, 64, 8).status == PolicyStatus::invalid_state);
    auto wrong_slot = elsewhere.value;
    wrong_slot.slot = handle;
    CHECK(pool.release_reader(wrong_slot) == PolicyStatus::invalid_lease);
    CHECK(pool.release_reader(first.value) == PolicyStatus::ok);
    CHECK(pool.release_reader(first.value) == PolicyStatus::invalid_lease);
    CHECK(pool.snapshot(0)->readers == 1);
    CHECK(pool.evict(handle) == PolicyStatus::readers_active);
    CHECK(pool.release_reader(second.value) == PolicyStatus::ok);
    CHECK(pool.snapshot(0)->readers == 0);
    CHECK(pool.evict(handle) == PolicyStatus::ok);
    CHECK(pool.release_reader(elsewhere.value) == PolicyStatus::ok);
}

void invalid_inputs_and_owner() {
    const std::array<std::uint64_t, 2> capacities{128, 0};
    SlotPool pool(capacities);
    SlotPool other_pool(capacities);
    CHECK(pool.size() == 2);
    CHECK(!pool.snapshot(2));
    CHECK(pool.begin_load(2, 1, 1).status == PolicyStatus::invalid_slot);
    CHECK(pool.begin_load(0, 0, 1).status == PolicyStatus::invalid_size);
    CHECK(pool.begin_load(0, 1, 0).status == PolicyStatus::invalid_token);
    CHECK(pool.begin_load(0, 129, 1).status == PolicyStatus::capacity_exceeded);
    CHECK(pool.begin_load(1, 1, 1).status == PolicyStatus::capacity_exceeded);
    CHECK(pool.snapshot(0)->state == SlotState::empty);
    CHECK(pool.publish({}) == PolicyStatus::stale_handle);
    const auto handle = ready(pool, 0);
    const auto lease = pool.acquire_reader(handle);
    CHECK(lease);
    CHECK(other_pool.publish(handle) == PolicyStatus::stale_handle);
    CHECK(other_pool.release_reader(lease.value) == PolicyStatus::stale_handle);
    CHECK(pool.snapshot(0)->readers == 1);
    CHECK(pool.release_reader(lease.value) == PolicyStatus::ok);
}

void budget_reservations() {
    MemoryBudget budget(1024, 512);
    MemoryBudget other(1024);
    CHECK(budget.total_bytes() == 1024);
    CHECK(budget.fixed_bytes() == 512);
    CHECK(budget.kv_bytes() == 0);
    CHECK(budget.reserve_kv(0).status == PolicyStatus::invalid_size);
    const auto a = budget.reserve_kv(256);
    const auto b = budget.reserve_kv(256);
    CHECK(a && b);
    CHECK(a.value.token != b.value.token);
    CHECK(budget.available_bytes() == 0);
    CHECK(budget.kv_bytes() == 512);
    CHECK(budget.reserve_kv(1).status == PolicyStatus::capacity_exceeded);
    auto changed = a.value;
    changed.bytes = 255;
    CHECK(budget.release_kv(changed) == PolicyStatus::invalid_reservation);
    CHECK(other.release_kv(a.value) == PolicyStatus::invalid_reservation);
    CHECK(budget.release_kv({}) == PolicyStatus::invalid_reservation);
    CHECK(budget.kv_bytes() == 512);
    CHECK(budget.release_kv(a.value) == PolicyStatus::ok);
    CHECK(budget.release_kv(a.value) == PolicyStatus::invalid_reservation);
    CHECK(budget.kv_bytes() == 256);
    CHECK(budget.available_bytes() == 256);
    CHECK(budget.release_kv(b.value) == PolicyStatus::ok);
    CHECK(budget.available_bytes() == 512);
}

void budget_overflow() {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    MemoryBudget budget(max, 1);
    CHECK(budget.reserve_kv(max).status == PolicyStatus::capacity_exceeded);
    const auto huge = budget.reserve_kv(max - 2);
    CHECK(huge);
    CHECK(budget.available_bytes() == 1);
    CHECK(budget.reserve_kv(2).status == PolicyStatus::capacity_exceeded);
    const auto last = budget.reserve_kv(1);
    CHECK(last);
    CHECK(budget.kv_bytes() == max - 1);
    CHECK(budget.available_bytes() == 0);
    CHECK(budget.release_kv(huge.value) == PolicyStatus::ok);
    CHECK(budget.release_kv(last.value) == PolicyStatus::ok);
    CHECK(budget.available_bytes() == max - 1);
    MemoryBudget full(max);
    const auto all = full.reserve_kv(max);
    CHECK(all);
    CHECK(full.kv_bytes() == max);
    CHECK(full.reserve_kv(1).status == PolicyStatus::capacity_exceeded);
    CHECK(full.release_kv(all.value) == PolicyStatus::ok);
    CHECK(full.available_bytes() == max);
    MemoryBudget zero(0);
    CHECK(zero.reserve_kv(1).status == PolicyStatus::capacity_exceeded);
    bool rejected = false;
    try { MemoryBudget invalid(1, max); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}
} // namespace

int main() {
    const std::array tests{
        std::pair{"delayed copy", &delayed_copy},
        std::pair{"stale generation", &stale_generation},
        std::pair{"reader leases", &pinned_readers_and_double_release},
        std::pair{"invalid inputs and ownership", &invalid_inputs_and_owner},
        std::pair{"KV reservations", &budget_reservations},
        std::pair{"budget overflow", &budget_overflow}
    };
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
