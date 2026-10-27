#include "sage/slot_pool.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace sage {

SlotPool::SlotPool(std::span<const std::uint64_t> capacities) {
    slots_.reserve(capacities.size());
    for (const auto capacity : capacities) {
        Slot slot{};
        slot.capacity = capacity;
        slots_.push_back(std::move(slot));
    }
}

PolicyStatus SlotPool::validate(SlotHandle handle) const noexcept {
    if (handle.owner != this || handle.index >= slots_.size())
        return PolicyStatus::stale_handle;
    if (handle.generation != slots_[handle.index].generation)
        return PolicyStatus::stale_handle;
    return PolicyStatus::ok;
}

PolicyResult<SlotHandle> SlotPool::begin_load(std::size_t index,
                                           std::uint64_t bytes,
                                           std::uint64_t copy_token) {
    if (index >= slots_.size()) return {PolicyStatus::invalid_slot};
    if (!bytes) return {PolicyStatus::invalid_size};
    if (!copy_token) return {PolicyStatus::invalid_token};
    auto& slot = slots_[index];
    if (slot.state != SlotState::empty) return {PolicyStatus::invalid_state};
    if (bytes > slot.capacity) return {PolicyStatus::capacity_exceeded};
    slot.content_bytes = bytes;
    slot.copy_token = copy_token;
    slot.pending = true;
    slot.state = SlotState::loading;
    return {PolicyStatus::ok, {this, index, slot.generation}};
}

PolicyStatus SlotPool::confirm_copy(SlotHandle handle, std::uint64_t token) {
    const auto status = validate(handle);
    if (status != PolicyStatus::ok) return status;
    auto& slot = slots_[handle.index];
    if (slot.state != SlotState::loading || !slot.pending)
        return PolicyStatus::invalid_state;
    if (!token || token != slot.copy_token) return PolicyStatus::invalid_token;
    slot.pending = false;
    return PolicyStatus::ok;
}

PolicyStatus SlotPool::publish(SlotHandle handle) {
    const auto status = validate(handle);
    if (status != PolicyStatus::ok) return status;
    auto& slot = slots_[handle.index];
    if (slot.state != SlotState::loading) return PolicyStatus::invalid_state;
    if (slot.pending) return PolicyStatus::copy_pending;
    slot.state = SlotState::ready;
    return PolicyStatus::ok;
}

PolicyResult<ReaderLease> SlotPool::acquire_reader(SlotHandle handle) {
    const auto status = validate(handle);
    if (status != PolicyStatus::ok) return {status};
    auto& slot = slots_[handle.index];
    if (slot.state != SlotState::ready) return {PolicyStatus::invalid_state};
    if (next_lease_ == std::numeric_limits<std::uint64_t>::max())
        return {PolicyStatus::counter_exhausted};
    const auto token = next_lease_;
    slot.readers.insert(token); // Commit the counter only after allocation succeeds.
    ++next_lease_;
    return {PolicyStatus::ok, {handle, token}};
}

PolicyStatus SlotPool::release_reader(ReaderLease lease) {
    const auto status = validate(lease.slot);
    if (status != PolicyStatus::ok) return status;
    auto& slot = slots_[lease.slot.index];
    if (!lease.token || slot.readers.erase(lease.token) != 1)
        return PolicyStatus::invalid_lease;
    return PolicyStatus::ok;
}

PolicyStatus SlotPool::evict(SlotHandle handle) {
    const auto status = validate(handle);
    if (status != PolicyStatus::ok) return status;
    auto& slot = slots_[handle.index];
    if (slot.state == SlotState::empty) return PolicyStatus::invalid_state;
    if (slot.pending) return PolicyStatus::copy_pending;
    if (!slot.readers.empty()) return PolicyStatus::readers_active;
    if (slot.generation == std::numeric_limits<std::uint64_t>::max())
        return PolicyStatus::counter_exhausted;
    ++slot.generation;
    slot.state = SlotState::empty;
    slot.content_bytes = 0;
    slot.copy_token = 0;
    return PolicyStatus::ok;
}

std::optional<SlotSnapshot> SlotPool::snapshot(std::size_t index) const {
    if (index >= slots_.size()) return std::nullopt;
    const auto& slot = slots_[index];
    return SlotSnapshot{slot.state, slot.generation, slot.capacity,
                        slot.content_bytes, slot.pending, slot.readers.size()};
}

MemoryBudget::MemoryBudget(std::uint64_t total, std::uint64_t fixed)
    : total_(total), fixed_(fixed) {
    if (fixed > total) throw std::invalid_argument("fixed memory exceeds total budget");
}

PolicyResult<KvReservation> MemoryBudget::reserve_kv(std::uint64_t bytes) {
    if (!bytes) return {PolicyStatus::invalid_size};
    // Subtraction precedes addition, so even UINT64_MAX requests cannot wrap.
    if (bytes > available_bytes()) return {PolicyStatus::capacity_exceeded};
    if (next_reservation_ == std::numeric_limits<std::uint64_t>::max())
        return {PolicyStatus::counter_exhausted};
    const auto token = next_reservation_;
    reservations_.emplace(token, bytes);
    ++next_reservation_;
    kv_ += bytes;
    return {PolicyStatus::ok, {this, token, bytes}};
}

PolicyStatus MemoryBudget::release_kv(KvReservation reservation) {
    if (reservation.owner != this) return PolicyStatus::invalid_reservation;
    const auto found = reservations_.find(reservation.token);
    if (found == reservations_.end() || found->second != reservation.bytes)
        return PolicyStatus::invalid_reservation;
    kv_ -= found->second;
    reservations_.erase(found);
    return PolicyStatus::ok;
}

} // namespace sage
