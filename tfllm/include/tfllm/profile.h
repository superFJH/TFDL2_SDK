#pragma once
#include <stdexcept>

// Propagated by the matching CMake library target. Normal builds discard the
// entire argument list, including names, shapes and metadata expressions.
#ifndef TFLLM_ENABLE_PROFILE
#define TFLLM_ENABLE_PROFILE 0
#endif
#if TFLLM_ENABLE_PROFILE
#define TFLLM_PROFILE(...) __VA_ARGS__
#else
#define TFLLM_PROFILE(...)
#endif

namespace tfllm {
inline constexpr bool ProfileCompiled = TFLLM_ENABLE_PROFILE != 0;
// Validate options before model loading, device access or output file creation.
inline void RequireProfileBuild() {
    if(!ProfileCompiled)
        throw std::invalid_argument("profiling is not compiled in; use the executable with the _profile suffix and --profile");
}
}

#if TFLLM_ENABLE_PROFILE
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tfllm {
// Host wall spans. Parent spans INCLUDE their children; never sum all kinds.
// A recorder belongs to one request, not to a process-global profiling switch.
struct ProfileShape {size_t m=0,k=0,n=0;};
struct ProfileEvent {
    uint64_t id=0,parent=0,thread=0;
    std::string kind,name,operation;
    ProfileShape shape;
    double start_ms=0,ms=0;
    int pair=-1,cache_hit=-1,head=-1;
    size_t first_row=0;
    size_t padded_m=0,descriptors=0,bytes=0,evictions=0;
    bool failed=false;
};
class ProfileRecorder {
public:
    using Clock=std::chrono::steady_clock;
    explicit ProfileRecorder(size_t max_events=100000);
    std::vector<ProfileEvent> Events() const;
    size_t Dropped() const noexcept {return dropped_.load();}
    // Also used for queue intervals whose endpoints are on different threads.
    void Record(ProfileEvent event,Clock::time_point begin,Clock::time_point end) noexcept;
    uint64_t NextId() noexcept {return next_.fetch_add(1);}
private:
    friend class ProfileScope;
    const size_t limit_;
    const Clock::time_point epoch_=Clock::now();
    mutable std::mutex mutex_;
    std::vector<ProfileEvent> events_;
    std::atomic<uint64_t> next_{1};
    std::atomic<size_t> dropped_{0};
};
struct ProfileContext {
    std::shared_ptr<ProfileRecorder> recorder;
    uint64_t parent=0;
    std::string operation;
};
ProfileContext CurrentProfile();
bool Profiling() noexcept;
// Capture CurrentProfile() on submission and bind it on the receiving worker.
// Binding an empty context explicitly clears any previous request attribution.
class ProfileBinding {
public:
    explicit ProfileBinding(ProfileContext context);
    ~ProfileBinding();
    ProfileBinding(const ProfileBinding&)=delete;
    ProfileBinding& operator=(const ProfileBinding&)=delete;
private:
    ProfileContext previous_;
};
class ProfileScope {
public:
    ProfileScope(const char* kind,const char* name,ProfileShape shape={}) noexcept;
    ~ProfileScope() {End();}
    void End() noexcept;
    ProfileEvent& Info() noexcept {return event_;}
    ProfileScope(const ProfileScope&)=delete;
    ProfileScope& operator=(const ProfileScope&)=delete;
private:
    ProfileContext previous_;
    ProfileEvent event_;
    ProfileRecorder::Clock::time_point begin_;
    int exceptions_=0;
    bool active_=false;
};
// Snapshot only after all request workers have completed. File I/O is outside
// recorded operator spans; the caller controls request/file naming.
void WriteProfileCsv(const ProfileRecorder& recorder,const std::string& path);
}
#endif
