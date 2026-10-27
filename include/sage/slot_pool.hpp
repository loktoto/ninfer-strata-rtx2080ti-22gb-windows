#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sage {

// Host scheduling policy only. All buffers and completion observations belong to
// the caller. These classes perform no device-buffer allocation, transfer or GPU
// synchronization; their bookkeeping uses ordinary host allocations.
// Serialize access externally, including access to snapshots and budget queries.
enum class PolicyStatus {
    ok, invalid_slot, stale_handle, invalid_state, invalid_token,
    copy_pending, readers_active, capacity_exceeded, invalid_lease,
    invalid_reservation, invalid_size, counter_exhausted
};

template <class T> struct PolicyResult {
    PolicyStatus status;
    T value{};
    explicit operator bool() const noexcept { return status == PolicyStatus::ok; }
};

enum class SlotState { empty, loading, ready };
class SlotPool;

struct SlotHandle {
    const SlotPool* owner = nullptr;
    std::size_t index = 0;
    std::uint64_t generation = 0;
};

struct ReaderLease {
    SlotHandle slot;
    std::uint64_t token = 0;
};

struct SlotSnapshot {
    SlotState state;
    std::uint64_t generation;
    std::uint64_t capacity_bytes;
    std::uint64_t content_bytes;
    bool copy_pending;
    std::size_t readers;
};

class SlotPool {
public:
    // Index i refers to an externally allocated buffer of capacities[i] bytes.
    // This copies capacities, owns only metadata, and never touches those buffers.
    explicit SlotPool(std::span<const std::uint64_t> capacities);
    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;
    SlotPool(SlotPool&&) = delete;
    SlotPool& operator=(SlotPool&&) = delete;

    // copy_token is a nonzero caller-issued transfer identifier, not a GPU event.
    PolicyResult<SlotHandle> begin_load(std::size_t index,
                                      std::uint64_t content_bytes,
                                      std::uint64_t copy_token);
    // Call only after the caller independently verifies this transfer completed.
    // Retain a pending slot after a transfer error until its writes have stopped.
    PolicyStatus confirm_copy(SlotHandle handle, std::uint64_t copy_token);
    PolicyStatus publish(SlotHandle handle);
    PolicyResult<ReaderLease> acquire_reader(SlotHandle handle);
    // Release only after the consumer has stopped using the underlying buffer.
    PolicyStatus release_reader(ReaderLease lease);
    // A completed but unpublished load may also be discarded. Success advances
    // the generation, permanently invalidating handles/leases from that content.
    PolicyStatus evict(SlotHandle handle);
    std::optional<SlotSnapshot> snapshot(std::size_t index) const;
    std::size_t size() const noexcept { return slots_.size(); }

private:
    struct Slot {
        std::uint64_t capacity;
        std::uint64_t generation = 1;
        std::uint64_t content_bytes = 0;
        std::uint64_t copy_token = 0;
        SlotState state = SlotState::empty;
        bool pending = false;
        std::unordered_set<std::uint64_t> readers;
    };
    PolicyStatus validate(SlotHandle handle) const noexcept;
    std::vector<Slot> slots_;
    std::uint64_t next_lease_ = 1;
};

class MemoryBudget;
struct KvReservation {
    const MemoryBudget* owner = nullptr;
    std::uint64_t token = 0;
    std::uint64_t bytes = 0;
};

class MemoryBudget {
public:
    // fixed_bytes includes external slot allocations and other committed memory.
    // Throws invalid_argument if it exceeds total_bytes. Neither value is probed.
    explicit MemoryBudget(std::uint64_t total_bytes, std::uint64_t fixed_bytes = 0);
    MemoryBudget(const MemoryBudget&) = delete;
    MemoryBudget& operator=(const MemoryBudget&) = delete;
    MemoryBudget(MemoryBudget&&) = delete;
    MemoryBudget& operator=(MemoryBudget&&) = delete;
    PolicyResult<KvReservation> reserve_kv(std::uint64_t bytes);
    PolicyStatus release_kv(KvReservation reservation);
    std::uint64_t total_bytes() const noexcept { return total_; }
    std::uint64_t fixed_bytes() const noexcept { return fixed_; }
    std::uint64_t kv_bytes() const noexcept { return kv_; }
    std::uint64_t available_bytes() const noexcept { return total_ - fixed_ - kv_; }

private:
    std::uint64_t total_, fixed_, kv_ = 0, next_reservation_ = 1;
    std::unordered_map<std::uint64_t, std::uint64_t> reservations_;
};

} // namespace sage
