/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
//
// aicpu_pingpong — host launcher that measures the atomic ping-pong round-trip
// latency between every pair of AICPU cpu_ids and prints the matrix.
//
// Usage: aicpu_pingpong <device_id> [launch_count] [iters] [reps] [warmup]
//
// Bootstrap path is aicpu-thread-spread's verbatim (dispatcher upload + inner
// SO via rtsBinaryLoadFromFile + rtsLaunchCpuKernel). On top of it:
//   1. one discovery launch learns which cpu_ids CANN dispatched threads to
//   2. one launch per unordered pair; only the two threads that landed on that
//      pair's cpu_ids take part, every other thread returns immediately

#include <acl/acl.h>
#include <runtime/rt.h>
#include <runtime/runtime/rts/rts_kernel.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fp {

bool ElfBuildId(const char *data, size_t len, uint64_t *out) {
    if (len < 64) return false;
    if (std::memcmp(
            data,
            "\x7f"
            "ELF",
            4
        ) != 0)
        return false;
    if (data[4] != 2) return false;
    uint64_t e_shoff;
    uint16_t e_shentsize, e_shnum, e_shstrndx;
    std::memcpy(&e_shoff, data + 40, 8);
    std::memcpy(&e_shentsize, data + 58, 2);
    std::memcpy(&e_shnum, data + 60, 2);
    std::memcpy(&e_shstrndx, data + 62, 2);
    if (e_shentsize != 64 || e_shoff + (uint64_t)e_shentsize * e_shnum > len) return false;
    const char *strtab = nullptr;
    {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * e_shstrndx;
        uint64_t off;
        uint64_t sz;
        std::memcpy(&off, sh + 24, 8);
        std::memcpy(&sz, sh + 32, 8);
        if (off + sz > len) return false;
        strtab = data + off;
    }
    for (uint16_t i = 0; i < e_shnum; ++i) {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * i;
        uint32_t sh_name;
        uint32_t sh_type;
        uint64_t sh_off;
        uint64_t sh_size;
        std::memcpy(&sh_name, sh + 0, 4);
        std::memcpy(&sh_type, sh + 4, 4);
        std::memcpy(&sh_off, sh + 24, 8);
        std::memcpy(&sh_size, sh + 32, 8);
        if (sh_type != 7) continue;
        const char *nm = strtab + sh_name;
        if (std::strcmp(nm, ".note.gnu.build-id") != 0) continue;
        if (sh_size < 16 || sh_off + sh_size > len) return false;
        const char *p = data + sh_off;
        uint32_t namesz, descsz, type;
        std::memcpy(&namesz, p + 0, 4);
        std::memcpy(&descsz, p + 4, 4);
        std::memcpy(&type, p + 8, 4);
        if (type != 3 || descsz < 8) return false;
        size_t name_aligned = (namesz + 3u) & ~3u;
        const char *desc = p + 12 + name_aligned;
        std::memcpy(out, desc, 8);
        return true;
    }
    return false;
}

uint64_t Fnv1a(const char *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (unsigned char)data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t Compute(const char *data, size_t len) {
    uint64_t fp;
    if (ElfBuildId(data, len, &fp)) return fp;
    return Fnv1a(data, len);
}

}  // namespace fp

namespace {

constexpr size_t kDeviceArgsBytes = 160;
constexpr uint32_t kMaxSlots = 32;
constexpr uint32_t kMaxReps = 16;
constexpr uint32_t kBankLines = 64;

#define ACL_CHECK(call, msg)                                                    \
    do {                                                                        \
        aclError _rc = (call);                                                  \
        if (_rc != ACL_SUCCESS) {                                               \
            std::fprintf(stderr, "%s failed: %d (%s)\n", #call, (int)_rc, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define RT_CHECK(call, msg)                                                     \
    do {                                                                        \
        rtError_t _rc = (call);                                                 \
        if (_rc != RT_ERROR_NONE) {                                             \
            std::fprintf(stderr, "%s failed: %d (%s)\n", #call, (int)_rc, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

bool ReadFile(const std::string &path, std::vector<char> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "open %s failed: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    f.seekg(0, std::ios::end);
    out->resize((size_t)f.tellg());
    f.seekg(0);
    f.read(out->data(), out->size());
    return f.good() || f.eof();
}

struct DevBuf {
    void *ptr{nullptr};
    aclError Alloc(size_t n) {
        aclError rc = aclrtMalloc(&ptr, n, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) ptr = nullptr;
        return rc;
    }
    ~DevBuf() {
        if (ptr) aclrtFree(ptr);
    }
};

// Must match the device-side layout byte for byte.
#pragma pack(push, 4)
struct PingPongOutput {
    uint32_t claim_counter;
    int32_t status_a;
    int32_t status_b;
    int32_t role_a_seen;
    int32_t role_b_seen;
    int32_t cpu_a_start;
    int32_t cpu_a_end;
    int32_t cpu_b_start;
    int32_t cpu_b_end;
    uint32_t reps_done;
    uint64_t freq_hz;
    uint64_t final_counter;
    uint64_t ticks[kMaxReps];
    int32_t thread_cpu[kMaxSlots];
    uint32_t reps_done_relax;
    uint64_t ticks_relax[kMaxReps];
    int32_t thread_core_id[kMaxSlots];
    int32_t thread_package_id[kMaxSlots];
    int32_t thread_cluster_id[kMaxSlots];
    int32_t thread_die_id[kMaxSlots];
    char cpu_present[64];
    char cpu_online[64];
    char cpu_possible[64];
    char node_info[512];
    char slot_topo[kMaxSlots][112];
    uint64_t probe_ops;
    uint64_t probe_ticks_a;
    uint64_t probe_ticks_b;
    uint64_t shared_va;
    uint64_t shared_pfn;
    uint64_t page_bytes;
    int32_t khz_a_start;
    int32_t khz_a_end;
    int32_t khz_b_start;
    int32_t khz_b_end;
    uint64_t bank_va;
    uint32_t lines_done;
    uint64_t line_ticks[kBankLines];
};
#pragma pack(pop)

// DeviceArgs offsets. 96..128 are the dispatcher bootstrap slots; after the
// bootstrap launch the block is rewritten with this tool's own fields.
constexpr size_t kBootstrapDispBin = 96;
constexpr size_t kBootstrapDispLen = 104;
constexpr size_t kBootstrapDevId = 112;
constexpr size_t kBootstrapInnerBin = 120;
constexpr size_t kBootstrapInnerLen = 128;
constexpr size_t kArgOutputAddr = 96;
constexpr size_t kArgMode = 104;
constexpr size_t kArgCpuA = 112;
constexpr size_t kArgCpuB = 120;
constexpr size_t kArgIters = 128;
constexpr size_t kArgWarmup = 136;
constexpr size_t kArgReps = 144;
constexpr size_t kArgSpinBudget = 152;

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], pp_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(pp_op, sizeof(pp_op), "simpler_aicpu_pingpong_%016lx", fp);
    auto entry = [&](const char *op, const char *fn) {
        std::string s = "  \"";
        s += op;
        s += "\": {\n    \"opInfo\": {\n";
        s += "      \"functionName\": \"";
        s += fn;
        s += "\",\n      \"kernelSo\": \"";
        s += so_basename;
        s += "\",\n      \"opKernelLib\": \"AICPUKernel\",\n";
        s += "      \"computeCost\": \"100\",\n";
        s += "      \"engine\": \"DNN_VM_AICPU\",\n";
        s += "      \"flagAsync\": \"False\",\n";
        s += "      \"flagPartial\": \"False\",\n";
        s += "      \"userDefined\": \"False\"\n";
        s += "    }\n  }";
        return s;
    };
    std::string s = "{\n";
    s += entry(init_op, "simpler_aicpu_init");
    s += ",\n";
    s += entry(pp_op, "simpler_aicpu_pingpong");
    s += "\n}\n";
    return s;
}

std::string StatusText(int32_t st) {
    if (st == 0) return "ok";
    std::string s;
    if (st & 1) s += "partner-absent ";
    if (st & 2) s += "spin-budget ";
    if (st & 4) s += "migrated ";
    return s;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <device_id> [launch_count] [iters] [reps] [warmup]\n", argv[0]);
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    int launch_count = (argc > 2) ? std::atoi(argv[2]) : 0;  // 0 = use discovered pool size
    const uint64_t iters = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1000;
    uint32_t reps = (argc > 4) ? (uint32_t)std::atoi(argv[4]) : 5;
    const uint64_t warmup = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 200;
    if (reps > kMaxReps) reps = kMaxReps;
    const uint64_t spin_budget = 100000000ULL;

    const char *dispatcher_path_env = std::getenv("SIMPLER_DISPATCHER_SO");
    if (dispatcher_path_env == nullptr) {
        std::fprintf(stderr, "SIMPLER_DISPATCHER_SO env var required\n");
        return 1;
    }
    const char *inner_path_env = std::getenv("SIMPLER_AICPU_PINGPONG_SO");
    std::string inner_path =
        inner_path_env ? inner_path_env : "tools/cann-examples/aicpu-pingpong/device/build/libaicpu_pingpong.so";

    std::vector<char> dispatcher_bytes, inner_bytes;
    if (!ReadFile(dispatcher_path_env, &dispatcher_bytes)) return 1;
    if (!ReadFile(inner_path, &inner_bytes)) return 1;

    ACL_CHECK(aclInit(nullptr), "aclInit");
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    DevBuf dev_dispatcher, dev_inner, dev_args, dev_out;
    ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "dispatcher GM");
    ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "inner GM");
    ACL_CHECK(dev_args.Alloc(kDeviceArgsBytes), "DeviceArgs GM");
    ACL_CHECK(dev_out.Alloc(sizeof(PingPongOutput)), "PingPongOutput GM");

    ACL_CHECK(
        aclrtMemcpy(
            dev_dispatcher.ptr, dispatcher_bytes.size(), dispatcher_bytes.data(), dispatcher_bytes.size(),
            ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "H2D dispatcher"
    );
    ACL_CHECK(
        aclrtMemcpy(
            dev_inner.ptr, inner_bytes.size(), inner_bytes.data(), inner_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "H2D inner"
    );

    char hostargs[kDeviceArgsBytes] = {};
    WriteU64(hostargs, kBootstrapDispBin, reinterpret_cast<uint64_t>(dev_dispatcher.ptr));
    WriteU64(hostargs, kBootstrapDispLen, dispatcher_bytes.size());
    WriteU64(hostargs, kBootstrapDevId, (uint64_t)device_id);
    WriteU64(hostargs, kBootstrapInnerBin, reinterpret_cast<uint64_t>(dev_inner.ptr));
    WriteU64(hostargs, kBootstrapInnerLen, inner_bytes.size());
    ACL_CHECK(
        aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, hostargs, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D DeviceArgs(bootstrap)"
    );

    {
        struct Args {
            struct {
                uint64_t unused[5] = {0};
                uint64_t device_args_ptr = 0;
                uint64_t pad[20] = {0};
            } k_args;
            char kernel_name[32];
            char so_name[32];
            char op_name[32];
        } args = {};
        args.k_args.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
        std::strncpy(args.kernel_name, "DynTileFwkKernelServerInit", sizeof(args.kernel_name) - 1);
        std::strncpy(args.so_name, "libaicpu_extend_kernels.so", sizeof(args.so_name) - 1);
        args.op_name[0] = '\0';

        rtAicpuArgsEx_t rt_args = {};
        rt_args.args = &args;
        rt_args.argsSize = sizeof(args);
        rt_args.kernelNameAddrOffset = offsetof(Args, kernel_name);
        rt_args.soNameAddrOffset = offsetof(Args, so_name);

        RT_CHECK(
            rtAicpuKernelLaunchExWithArgs(
                rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &rt_args, nullptr, stream, 0
            ),
            "rtAicpuKernelLaunchExWithArgs(bootstrap)"
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream), "sync after bootstrap");

    const uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
    std::string so_basename = "simpler_inner_" + [&] {
        char b[24];
        std::snprintf(b, sizeof(b), "%016lx_%d.so", fp, device_id);
        return std::string(b);
    }();
    std::printf("[bootstrap] inner SO uploaded as %s (fp=%016lx)\n", so_basename.c_str(), fp);

    char json_path_buf[128];
    std::snprintf(json_path_buf, sizeof(json_path_buf), "/tmp/simpler_pingpong_%016lx_%d.json", fp, getpid());
    const std::string json_path = json_path_buf;
    {
        std::ofstream f(json_path);
        if (!f.is_open()) {
            std::fprintf(stderr, "open %s failed\n", json_path.c_str());
            return 1;
        }
        f << MakeJsonDescriptor(fp, so_basename);
    }

    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;
    void *binary_handle = nullptr;
    RT_CHECK(rtsBinaryLoadFromFile(json_path.c_str(), &load_config, &binary_handle), "rtsBinaryLoadFromFile");
    std::remove(json_path.c_str());

    rtFuncHandle init_handle = nullptr, pp_handle = nullptr;
    {
        char init_op[128], pp_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(pp_op, sizeof(pp_op), "simpler_aicpu_pingpong_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "rtsFuncGetByName init");
        RT_CHECK(rtsFuncGetByName(binary_handle, pp_op, &pp_handle), "rtsFuncGetByName pingpong");
    }

    struct LaunchArgs {
        uint64_t _pad[5] = {0};
        uint64_t device_args_ptr = 0;
        uint64_t reserved[20] = {0};
    } la = {};
    la.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = &la;
    cpu_args.baseArgs.argsSize = sizeof(la);
    rtLaunchKernelAttr_t attr = {};
    rtKernelLaunchCfg_t cfg = {&attr, 0};

    PingPongOutput host_out{};
    const PingPongOutput zero_out{};

    // --- one round: reset statics, set args, launch N threads, read back -----
    auto run_round = [&](uint64_t mode, int cpu_a, int cpu_b, int threads) -> int {
        ACL_CHECK(
            aclrtMemcpy(
                dev_out.ptr, sizeof(PingPongOutput), &zero_out, sizeof(PingPongOutput), ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "H2D zero out"
        );
        char a[kDeviceArgsBytes] = {};
        WriteU64(a, kArgOutputAddr, reinterpret_cast<uint64_t>(dev_out.ptr));
        WriteU64(a, kArgMode, mode);
        WriteU64(a, kArgCpuA, (uint64_t)(int64_t)cpu_a);
        WriteU64(a, kArgCpuB, (uint64_t)(int64_t)cpu_b);
        WriteU64(a, kArgIters, iters);
        WriteU64(a, kArgWarmup, warmup);
        WriteU64(a, kArgReps, reps);
        WriteU64(a, kArgSpinBudget, spin_budget);
        ACL_CHECK(
            aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, a, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D args"
        );
        RT_CHECK(rtsLaunchCpuKernel(init_handle, 1u, stream, &cfg, &cpu_args), "launch init");
        ACL_CHECK(aclrtSynchronizeStream(stream), "sync init");
        RT_CHECK(rtsLaunchCpuKernel(pp_handle, (uint32_t)threads, stream, &cfg, &cpu_args), "launch pingpong");
        ACL_CHECK(aclrtSynchronizeStream(stream), "sync pingpong");
        ACL_CHECK(
            aclrtMemcpy(
                &host_out, sizeof(PingPongOutput), dev_out.ptr, sizeof(PingPongOutput), ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "D2H out"
        );
        return 0;
    };

    // --- discovery ----------------------------------------------------------
    const int discover_threads = launch_count > 0 ? launch_count : (int)kMaxSlots / 2;
    if (run_round(0, -1, -1, discover_threads) != 0) return 1;

    std::vector<int> cpus;
    for (uint32_t i = 0; i < host_out.claim_counter && i < kMaxSlots; ++i)
        cpus.push_back(host_out.thread_cpu[i]);
    std::sort(cpus.begin(), cpus.end());
    std::printf("\n=== discovery: launched %d threads, %u reported ===\n", discover_threads, host_out.claim_counter);
    std::printf("  cpu_id spread (sorted, duplicates = CANN over-subscription):\n    ");
    for (size_t i = 0; i < cpus.size(); ++i)
        std::printf("%s%d", (i ? " " : ""), cpus[i]);
    std::printf("\n");
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    const int n = (int)cpus.size();
    std::printf("  unique user-schedulable cpu_ids: %d  ->  %d unordered pairs\n", n, n * (n - 1) / 2);

    // The set above is where CANN put threads, which is a lower bound on the
    // pool. The AICPU OS's own inventory is the upper bound, and the gap between
    // them is the difference between a part with fewer AICPU cores and a CANN
    // that dispatches to fewer of them.
    auto show = [](const char *s) {
        return (s[0] != '\0') ? s : "(unreadable)";
    };
    std::printf("\n=== AICPU OS cpu inventory ===\n");
    std::printf("  /sys/devices/system/cpu/present  : %s\n", show(host_out.cpu_present));
    std::printf("  /sys/devices/system/cpu/online   : %s\n", show(host_out.cpu_online));
    std::printf("  /sys/devices/system/cpu/possible : %s\n", show(host_out.cpu_possible));
    std::printf("  /sys/devices/system/node         : %s\n", show(host_out.node_info));

    // Cache sharing is the direct answer to "cluster or NUMA domain": the
    // sharers of the outermost level are the group, whatever it is called.
    std::printf("\n=== AICPU OS cache and sibling topology ===\n");
    {
        std::vector<int> seen;
        for (uint32_t i = 0; i < host_out.claim_counter && i < kMaxSlots; ++i) {
            const int c = host_out.thread_cpu[i];
            if (std::find(seen.begin(), seen.end(), c) != seen.end()) continue;
            seen.push_back(c);
            std::printf("  cpu%-3d %s\n", c, show(host_out.slot_topo[i]));
        }
    }

    // cpu_id -> cluster/die is unresolvable from the driver on this part
    // (CPU_TOPO returns DRV_ERROR_NOT_SUPPORT), so report the AICPU OS's own
    // sysfs view. -1 means the file is absent, which is itself the answer.
    std::printf("\n=== AICPU OS sysfs topology (-1 = file absent) ===\n");
    std::printf("  %-8s %-10s %-20s %-12s %-8s\n", "cpu_id", "core_id", "physical_package_id", "cluster_id", "die_id");
    for (uint32_t i = 0; i < host_out.claim_counter && i < kMaxSlots; ++i) {
        std::printf(
            "  %-8d %-10d %-20d %-12d %-8d\n", host_out.thread_cpu[i], host_out.thread_core_id[i],
            host_out.thread_package_id[i], host_out.thread_cluster_id[i], host_out.thread_die_id[i]
        );
    }
    if (n < 2) {
        std::fprintf(stderr, "need at least 2 distinct cpu_ids to measure a pair\n");
        return 1;
    }
    if (launch_count <= 0) launch_count = n;

    // --- pair sweep ---------------------------------------------------------
    std::vector<double> med(n * n, -1.0), best(n * n, -1.0);
    std::vector<double> rmed(n * n, -1.0), rbest(n * n, -1.0), ratio(n * n, -1.0);
    std::vector<double> probe_us(n * n, -1.0);
    std::printf(
        "\n=== sweep: iters=%llu reps=%u warmup=%llu launch_count=%d ===\n", (unsigned long long)iters, reps,
        (unsigned long long)warmup, launch_count
    );
    std::printf(
        "  %-6s %-6s %9s %9s %9s %9s %7s %9s %9s  %s\n", "cpu_a", "cpu_b", "bare_min", "bare_med", "isb_min", "isb_med",
        "isb/bare", "probe_a", "probe_b", "status"
    );
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (run_round(1, cpus[i], cpus[j], launch_count) != 0) return 1;
            const uint32_t done = host_out.reps_done > kMaxReps ? kMaxReps : host_out.reps_done;
            std::string st = StatusText(host_out.status_a) + "/" + StatusText(host_out.status_b);
            if (host_out.role_a_seen == 0 || host_out.role_b_seen == 0) st += " [role-missing]";
            const uint32_t rdone = host_out.reps_done_relax > kMaxReps ? kMaxReps : host_out.reps_done_relax;
            // Both roles time the same dependent-ALU chain inside this launch,
            // so a round trip that moves while the probe does not is not the
            // core clock moving.
            const double tick_hz = host_out.freq_hz ? (double)host_out.freq_hz : 1e9;
            const double probe_a_us = (double)host_out.probe_ticks_a * 1e6 / tick_hz;
            const double probe_b_us = (double)host_out.probe_ticks_b * 1e6 / tick_hz;
            probe_us[i * n + j] = probe_us[j * n + i] = probe_a_us;
            if (done == 0 || host_out.freq_hz == 0) {
                std::printf(
                    "  %-6d %-6d %9s %9s %9s %9s %7s %9.1f %9.1f  %s\n", cpus[i], cpus[j], "-", "-", "-", "-", "-",
                    probe_a_us, probe_b_us, st.c_str()
                );
                continue;
            }
            // Both phases come from one launch, so they share a fabric clock
            // state and their ratio is clock-independent.
            auto summarise = [&](const uint64_t *ticks, uint32_t count, double *mn, double *md) {
                std::vector<double> ns;
                for (uint32_t r = 0; r < count; ++r)
                    ns.push_back((double)ticks[r] * 1e9 / (double)host_out.freq_hz / (double)iters);
                std::sort(ns.begin(), ns.end());
                *mn = ns.front();
                *md = ns[ns.size() / 2];
            };
            double mn = 0, md = 0;
            summarise(host_out.ticks, done, &mn, &md);
            med[i * n + j] = med[j * n + i] = md;
            best[i * n + j] = best[j * n + i] = mn;

            double rmn = -1.0, rmd = -1.0, rat = -1.0;
            if (rdone > 0) {
                summarise(host_out.ticks_relax, rdone, &rmn, &rmd);
                rmed[i * n + j] = rmed[j * n + i] = rmd;
                rbest[i * n + j] = rbest[j * n + i] = rmn;
                rat = rmd / md;
                ratio[i * n + j] = ratio[j * n + i] = rat;
                std::printf(
                    "  %-6d %-6d %9.1f %9.1f %9.1f %9.1f %7.3f %9.1f %9.1f  %s\n", cpus[i], cpus[j], mn, md, rmn, rmd,
                    rat, probe_a_us, probe_b_us, st.c_str()
                );
            } else {
                std::printf(
                    "  %-6d %-6d %9.1f %9.1f %9s %9s %7s %9.1f %9.1f  %s\n", cpus[i], cpus[j], mn, md, "-", "-", "-",
                    probe_a_us, probe_b_us, st.c_str()
                );
            }
            std::printf(
                "         counter line: va=0x%llx pfn=0x%llx page=%llu\n", (unsigned long long)host_out.shared_va,
                (unsigned long long)host_out.shared_pfn, (unsigned long long)host_out.page_bytes
            );
            if (host_out.khz_a_start >= 0 || host_out.khz_b_start >= 0) {
                std::printf(
                    "         cpufreq kHz: cpu%d %d->%d  cpu%d %d->%d\n", cpus[i], host_out.khz_a_start,
                    host_out.khz_a_end, cpus[j], host_out.khz_b_start, host_out.khz_b_end
                );
            }
            std::fflush(stdout);
        }
    }

    auto print_matrix = [&](const char *title, const std::vector<double> &m) {
        std::printf("\n=== %s ===\n", title);
        std::printf("%8s", "cpu");
        for (int j = 0; j < n; ++j)
            std::printf("%9d", cpus[j]);
        std::printf("\n");
        for (int i = 0; i < n; ++i) {
            std::printf("%8d", cpus[i]);
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    std::printf("%9s", "-");
                } else if (m[i * n + j] < 0) {
                    std::printf("%9s", "x");
                } else {
                    std::printf("%9.1f", m[i * n + j]);
                }
            }
            std::printf("\n");
        }
    };
    print_matrix("bare spin: round-trip latency, ns (median of reps)", med);
    print_matrix("bare spin: round-trip latency, ns (min of reps)", best);
    print_matrix("isb spin: round-trip latency, ns (median of reps)", rmed);
    print_matrix("isb / bare ratio (same launch, so clock-independent)", ratio);

    // The pair matrix measures one counter, so a per-launch cost cannot be told
    // apart from a per-address one. This walks 64 counters 128 B apart in a
    // single launch, with the pair and the clock held fixed.
    std::printf("\n=== line sweep: round-trip latency per counter line, ns ===\n");
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (run_round(2, cpus[i], cpus[j], launch_count) != 0) return 1;
            if (host_out.freq_hz == 0) {
                std::printf("  cpu%d<->cpu%d: no timer\n", cpus[i], cpus[j]);
                continue;
            }
            const uint32_t nl = host_out.lines_done > kBankLines ? kBankLines : host_out.lines_done;
            std::vector<double> ns;
            for (uint32_t l = 0; l < nl; ++l)
                ns.push_back((double)host_out.line_ticks[l] * 1e9 / (double)host_out.freq_hz / (double)iters);
            std::printf(
                "  cpu%d<->cpu%d  bank_va=0x%llx  lines %u/%u  status %s/%s\n", cpus[i], cpus[j],
                (unsigned long long)host_out.bank_va, nl, kBankLines, StatusText(host_out.status_a).c_str(),
                StatusText(host_out.status_b).c_str()
            );
            if (nl == 0) continue;
            for (uint32_t l = 0; l < nl; ++l) {
                if (l % 8 == 0) std::printf("    line %2u:", l);
                std::printf(" %8.1f", ns[l]);
                if (l % 8 == 7) std::printf("\n");
            }
            if (nl % 8 != 0) std::printf("\n");
            std::vector<double> sorted(ns);
            std::sort(sorted.begin(), sorted.end());
            std::printf(
                "    min %.1f  median %.1f  max %.1f  spread %.1f ns\n", sorted.front(), sorted[sorted.size() / 2],
                sorted.back(), sorted.back() - sorted.front()
            );
            for (uint32_t l = 0; l < nl; ++l)
                std::printf("LINE,%d,%d,%u,%.2f\n", cpus[i], cpus[j], l, ns[l]);
            std::fflush(stdout);
        }
    }

    std::printf("\n=== CSV (cpu_a,cpu_b,min_ns,median_ns,relax_min_ns,relax_median_ns,ratio,probe_us) ===\n");
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            std::printf(
                "CSV,%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.1f\n", cpus[i], cpus[j], best[i * n + j], med[i * n + j],
                rbest[i * n + j], rmed[i * n + j], ratio[i * n + j], probe_us[i * n + j]
            );
    std::printf(
        "\nOne round trip = both threads increment once (2 fetch_adds).\n"
        "Halve for a one-way handoff. Timer: cntvct_el0 @ %llu Hz.\n",
        (unsigned long long)host_out.freq_hz
    );

    ACL_CHECK(aclrtDestroyStream(stream), "destroy stream");
    ACL_CHECK(aclrtResetDevice(device_id), "reset device");
    aclFinalize();
    return 0;
}
