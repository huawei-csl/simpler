// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//
// Host launcher for the AICPU <-> AICore ping-pong.
//
// Three modes:
//   census (0)  AICore only, blockDim wide. No block waits for anything, so it
//               cannot hang. Reports which of the 108 slots attended.
//   probe  (2)  AICore only. Publishes one magic via dcci and via st_dev so the
//               host can confirm both reach GM before the loop depends on it.
//   measure(1)  Both halves on their own streams: one AICPU thread and one
//               AICore block hand a token back and forth.
//
// The bootstrap half (dispatcher upload, ELF fingerprint, JSON descriptor,
// rtsBinaryLoadFromFile / rtsFuncGetByName / rtsLaunchCpuKernel) is taken from
// aicpu-pingpong, which is already proven on this silicon.

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
#include <random>
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


// DeviceArgs offsets. 96..128 are the dispatcher bootstrap slots; after the
// bootstrap launch the block is rewritten with this tool's own fields.
constexpr size_t kBootstrapDispBin = 96;
constexpr size_t kBootstrapDispLen = 104;
constexpr size_t kBootstrapDevId = 112;
constexpr size_t kBootstrapInnerBin = 120;
constexpr size_t kBootstrapInnerLen = 128;

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], pp_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(pp_op, sizeof(pp_op), "simpler_aicpu_run_%016lx", fp);
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
    s += entry(pp_op, "simpler_aicpu_run");
    s += "\n}\n";
    return s;
}
}  // namespace

#include "../shared/pingpong.h"

namespace {
const char *SlotKind(uint32_t slot) { return PP_SLOT_IS_AIC(slot) ? "AIC" : "AIV"; }
}  // namespace


int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <device_id> [mode] [blockDim] [slot_a] [slot_b] [cpu_a] [cpu_b] [iters] [warmup]\n"
                     "  mode: 0=census 1=measure 2=probe 3=core2core\n",
                     argv[0]);
        return 1;
    }
    const int device_id = std::atoi(argv[1]);
    const uint32_t mode = (argc > 2) ? (uint32_t)std::atoi(argv[2]) : PP_MODE_CENSUS;
    // 0, or no argument at all, means "ask the device". PP_MAX_BLOCKDIM is the
    // 950DT cluster count, and over-dispatching it on a part with fewer clusters
    // stalls the stream until the op-execute timeout with nothing in any log, so
    // the count is queried below rather than assumed.
    const uint32_t block_dim_arg = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 0u;
    const uint32_t slot_a = (argc > 4) ? (uint32_t)std::atoi(argv[4]) : 0u;    // AIC die 0
    const uint32_t slot_b = (argc > 5) ? (uint32_t)std::atoi(argv[5]) : 54u;   // AIC die 1
    const int32_t cpu_a = (argc > 6) ? std::atoi(argv[6]) : 2;                 // AICPU die 0 (never cpu1)
    const int32_t cpu_b = (argc > 7) ? std::atoi(argv[7]) : 4;                 // AICPU die 1
    const uint64_t iters = (argc > 8) ? std::strtoull(argv[8], nullptr, 10) : 1000;
    const uint64_t warmup = (argc > 9) ? std::strtoull(argv[9], nullptr, 10) : 200;

    const char *partner_env = std::getenv("PP_PARTNER_O");
    if (partner_env == nullptr) {
        std::fprintf(stderr, "PP_PARTNER_O must point at aicpu_aicore_pingpong_partner.o\n");
        return 1;
    }
    std::vector<char> partner_bytes;
    if (!ReadFile(partner_env, &partner_bytes)) return 1;

    const bool need_aicpu = (mode == PP_MODE_MEASURE);
    std::vector<char> dispatcher_bytes, inner_bytes;
    const char *dispatcher_path = std::getenv("SIMPLER_DISPATCHER_SO");
    const char *inner_path = std::getenv("PP_DRIVER_SO");
    if (need_aicpu) {
        if (dispatcher_path == nullptr || inner_path == nullptr) {
            std::fprintf(stderr, "measure mode needs SIMPLER_DISPATCHER_SO and PP_DRIVER_SO\n");
            return 1;
        }
        if (!ReadFile(dispatcher_path, &dispatcher_bytes)) return 1;
        if (!ReadFile(inner_path, &inner_bytes)) return 1;
    }

    ACL_CHECK(aclInit(nullptr), "aclInit");
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice");

    // One block holds one 1AIC+2AIV cluster, so blockDim is a bijection onto the
    // clusters only when it equals the cube-core count. That count is a property
    // of the part, not of the architecture: 950DT reports 36, 950PR parts report
    // 36 or 28.
    int64_t cube_cores = 0, vector_cores = 0;
    ACL_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_AI_CORE_NUM, &cube_cores), "AI_CORE_NUM");
    ACL_CHECK(
        aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vector_cores), "VECTOR_CORE_NUM"
    );
    const uint32_t block_dim = (block_dim_arg != 0u) ? block_dim_arg : (uint32_t)cube_cores;
    std::printf("[chip] %lld cube + %lld vector cores -> %lld slots; blockDim=%u (%s)\n",
                (long long)cube_cores, (long long)vector_cores, (long long)(cube_cores + vector_cores),
                block_dim, block_dim_arg != 0u ? "argv" : "queried");
    if ((int64_t)block_dim > cube_cores) {
        std::fprintf(stderr, "blockDim %u exceeds the %lld cube cores on this part; the launch would stall\n",
                     block_dim, (long long)cube_cores);
        return 1;
    }
    aclrtStream stream_aicore = nullptr, stream_aicpu = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream_aicore), "aclrtCreateStream(aicore)");
    if (need_aicpu) ACL_CHECK(aclrtCreateStream(&stream_aicpu), "aclrtCreateStream(aicpu)");

    // Several arenas, each a separate aclrtMalloc, all swept inside this one
    // launch. The AICore core ids, the AICPU pinning and the fabric clock are
    // all held fixed across the sweeps, so anything that differs between them
    // is a property of the allocation and of nothing else.
    const char *arenas_env = std::getenv("PP_ARENAS");
    const uint32_t n_arenas = arenas_env ? (uint32_t)std::atoi(arenas_env) : 1u;
    std::vector<void *> arena_ptrs(n_arenas ? n_arenas : 1u, nullptr);
    for (uint32_t a = 0; a < arena_ptrs.size(); ++a) {
        ACL_CHECK(aclrtMalloc(&arena_ptrs[a], sizeof(PpArena), ACL_MEM_MALLOC_HUGE_FIRST), "arena GM");
        std::printf("[arena] %u: dev=0x%016llx size=%zu\n", a, (unsigned long long)arena_ptrs[a], sizeof(PpArena));
    }

    // static, not automatic: the chase array alone is 16 MB and the default
    // 8 MB stack cannot hold it.
    static PpArena host_arena;
    std::memset(&host_arena, 0, sizeof(host_arena));

    // ---- Register the AICore half ----
    rtDevBinary_t bin{};
    bin.magic = RT_DEV_BINARY_MAGIC_ELF;
    bin.version = 0;
    bin.data = partner_bytes.data();
    bin.length = partner_bytes.size();
    void *aicore_handle = nullptr;
    RT_CHECK(rtRegisterAllKernel(&bin, &aicore_handle), "rtRegisterAllKernel(partner)");

    // ---- Which slots this part actually has ----
    // The slot space is 108 wide on every a5 part, but a part with fewer
    // clusters occupies it sparsely, and a disabled cluster leaves a hole that
    // no arithmetic on the core count predicts. A census launch reports the set
    // and cannot block: every block writes one line and returns. Scheduling a
    // slot that no block owns would spin that round to its budget instead.
    std::vector<uint32_t> occupied;
    {
        host_arena.ctl.go = 1;
        host_arena.ctl.mode = PP_MODE_CENSUS;
        host_arena.ctl.clock_iters = 1;
        ACL_CHECK(
            aclrtMemcpy(arena_ptrs[0], sizeof(PpArena), &host_arena, sizeof(PpArena), ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D census arena"
        );
        struct CensusArgs {
            uint64_t arena_ptr;
        } ca;
        ca.arena_ptr = reinterpret_cast<uint64_t>(arena_ptrs[0]);
        rtArgsEx_t census_args = {};
        census_args.args = &ca;
        census_args.argsSize = sizeof(ca);
        rtTaskCfgInfo_t census_cfg = {};
        census_cfg.schemMode = RT_SCHEM_MODE_BATCH;
        RT_CHECK(
            rtKernelLaunchWithHandleV2(aicore_handle, 0, block_dim, &census_args, nullptr, stream_aicore,
                                       &census_cfg),
            "rtKernelLaunchWithHandleV2(census)"
        );
        ACL_CHECK(aclrtSynchronizeStream(stream_aicore), "sync census");
        ACL_CHECK(
            aclrtMemcpy(&host_arena, sizeof(PpArena), arena_ptrs[0], sizeof(PpArena), ACL_MEMCPY_DEVICE_TO_HOST),
            "D2H census arena"
        );
        uint32_t occ_aic = 0, occ_die[2] = {0, 0};
        for (uint32_t s = 0; s < PP_MAX_SLOTS; ++s) {
            if (host_arena.attendance[s].attended != PP_ATTEND_MAGIC) continue;
            occupied.push_back(s);
            if (PP_SLOT_IS_AIC(s)) ++occ_aic;
            ++occ_die[PP_SLOT_DIE(s)];
        }
        std::memset(&host_arena, 0, sizeof(host_arena));
        if (occupied.empty()) {
            std::fprintf(stderr, "census found no AICore blocks; nothing can be scheduled\n");
            return 1;
        }
        std::printf("[census] %zu occupied slots: %u AIC, %zu AIV; die0 %u / die1 %u\n", occupied.size(), occ_aic,
                    occupied.size() - occ_aic, occ_die[0], occ_die[1]);
    }

    host_arena.ctl.go = 1;
    host_arena.ctl.mode = mode;
    host_arena.ctl.n_blocks_expected = block_dim;
    host_arena.ctl.core_slot[0] = slot_a;
    host_arena.ctl.core_slot[1] = slot_b;
    host_arena.ctl.cpu_id[0] = cpu_a;
    host_arena.ctl.cpu_id[1] = cpu_b;
    // Full matrix in one launch. Per-launch overhead is ~58 s against ~1.2 ms
    // of measurement, so 756 launches would take 12 h; 756 rounds inside one
    // launch take about a second, and share a clock state into the bargain.
    //
    // A schedule longer than PP_MAX_ROUNDS is split into equal chunks that each
    // fit the arena's pair_list, and one arena runs one chunk. The split is a
    // host-side loop, so the arena layout the device sees is unchanged and the
    // numbers stay comparable with a run whose schedule fit in a single piece.
    std::vector<uint32_t> c2c_all;
    uint32_t c2c_chunk_rounds = 0;
    {
        const char *cpus_env = std::getenv("PP_CPUS");
        uint32_t n = 0;
        if (cpus_env != nullptr) {
            std::string spec(cpus_env);
            size_t pos = 0;
            while (pos < spec.size() && n < PP_MAX_CPUS) {
                size_t comma = spec.find(',', pos);
                host_arena.ctl.cpu_list[n++] = std::atoi(spec.substr(pos, comma - pos).c_str());
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        if (n == 0) {
            host_arena.ctl.cpu_list[n++] = cpu_a;
        }
        // Sample AICore slots rather than enumerating all 108. The sample is
        // drawn from the full 0..107 space so cube and vector slots are mixed
        // in proportion, and the seed is printed so a run can be repeated
        // against exactly the same set.
        //
        // slot_list stays in shuffled order, which the device walks directly:
        // a slot's die and unit kind are therefore uncorrelated with its
        // position in the sweep, so an effect that tracks position cannot
        // present as a die or kind effect.
        const char *slots_env = std::getenv("PP_NSLOTS");
        const char *seed_env = std::getenv("PP_SEED");
        const char *reps_env = std::getenv("PP_REPS");
        uint32_t want = slots_env ? (uint32_t)std::atoi(slots_env) : (uint32_t)occupied.size();
        if (want > PP_MAX_SLOTLIST) want = PP_MAX_SLOTLIST;
        if (want > occupied.size()) want = (uint32_t)occupied.size();
        const unsigned seed = seed_env ? (unsigned)std::atoi(seed_env) : 20261111u;
        std::vector<uint32_t> pool(occupied);
        std::mt19937 rng(seed);
        std::shuffle(pool.begin(), pool.end(), rng);
        pool.resize(want);
        uint32_t n_aic = 0;
        for (uint32_t i = 0; i < want; ++i) {
            host_arena.slot_list[i] = pool[i];
            if (PP_SLOT_IS_AIC(pool[i])) ++n_aic;
        }
        std::printf("[sweep] sampled %u of %zu AICore slots (seed=%u): %u AIC, %u AIV\n", want, occupied.size(),
                    seed, n_aic, want - n_aic);
        std::printf("[sweep] slots:");
        for (uint32_t i = 0; i < want; ++i) std::printf(" %u", pool[i]);
        std::printf("\n");

        host_arena.ctl.n_cpus = n;
        host_arena.ctl.n_slots = want;
        host_arena.ctl.n_reps = reps_env ? (uint32_t)std::atoi(reps_env) : 1u;
        host_arena.ctl.n_rounds = host_arena.ctl.n_cpus * host_arena.ctl.n_slots * host_arena.ctl.n_reps;
        if (host_arena.ctl.n_rounds > PP_MAX_ROUNDS) {
            std::fprintf(stderr, "schedule %u rounds exceeds PP_MAX_ROUNDS=%u\n", host_arena.ctl.n_rounds,
                         PP_MAX_ROUNDS);
            return 1;
        }

        // Every (AICPU, slot) pair appears once per repeat, and the whole list
        // is shuffled as one. A round's AICPU, its AICore slot and its repeat
        // number are therefore all independent of its position in the sweep.
        std::vector<uint32_t> sched;
        sched.reserve(host_arena.ctl.n_rounds);
        for (uint32_t rep = 0; rep < host_arena.ctl.n_reps; ++rep) {
            for (uint32_t ci = 0; ci < n; ++ci) {
                for (uint32_t si = 0; si < want; ++si) sched.push_back((ci << 16) | si);
            }
        }
        std::shuffle(sched.begin(), sched.end(), rng);
        for (uint32_t i = 0; i < sched.size(); ++i) host_arena.pair_list[i] = sched[i];
        std::printf("[sweep] schedule: %zu rounds, pair order shuffled (seed=%u)\n", sched.size(), seed);

        if (mode == PP_MODE_CORE2CORE) {
            // Replace the sampled schedule with every slot the census found:
            // both endpoints are AICore here, and the question is which pairs
            // are slow, so no pair may be dropped. slot_list is in ascending
            // slot order and the pair order carries the shuffle instead, so a
            // round's die and cluster relation is independent of its position
            // in the sweep.
            const uint32_t n_sl = (uint32_t)occupied.size();
            for (uint32_t i = 0; i < n_sl; ++i) host_arena.slot_list[i] = occupied[i];
            host_arena.ctl.n_cpus = 0;
            host_arena.ctl.n_slots = n_sl;
            for (uint32_t rep = 0; rep < host_arena.ctl.n_reps; ++rep) {
                for (uint32_t i = 0; i < n_sl; ++i) {
                    for (uint32_t j = i + 1u; j < n_sl; ++j) c2c_all.push_back((i << 16) | j);
                }
            }
            std::shuffle(c2c_all.begin(), c2c_all.end(), rng);

            // Chunks are equal-sized so every arena runs the same number of
            // rounds for the same wall time, and a chunk is a slice of the
            // already-shuffled list, so it holds a random subset of pairs
            // rather than a die or cluster stratum.
            const uint32_t n_chunks = (uint32_t)((c2c_all.size() + PP_MAX_ROUNDS - 1u) / PP_MAX_ROUNDS);
            c2c_chunk_rounds = (uint32_t)((c2c_all.size() + n_chunks - 1u) / n_chunks);
            host_arena.ctl.n_rounds = c2c_chunk_rounds;
            std::printf("[c2c] %u AICore slots, %zu unordered pairs x %u reps = %zu rounds\n", n_sl,
                        (size_t)n_sl * (n_sl - 1u) / 2u, host_arena.ctl.n_reps, c2c_all.size());
            std::printf("[c2c] %u chunk(s) of %u rounds; arena a runs chunk a %% %u\n", n_chunks, c2c_chunk_rounds,
                        n_chunks);
        }
    }
    {
        // One random cycle through every line, so the walk visits all of them
        // before repeating and no step's address follows from the last.
        const char *ci = std::getenv("PP_CLOCK_ITERS");
        const char *cs = std::getenv("PP_CHASE_STEPS");
        host_arena.ctl.clock_iters = ci ? (uint32_t)std::atoi(ci) : 20000u;
        host_arena.ctl.chase_steps = cs ? (uint32_t)std::atoi(cs) : 256u;
        // The walked region is a prefix of the array, so its size is a host
        // knob. It matters: a region spanning many pages is spread over both
        // dies, and a walk across it averages the two distances instead of
        // showing either. A region inside one page has a single home.
        const char *cl = std::getenv("PP_CHASE_LINES_USED");
        uint32_t lines = cl ? (uint32_t)std::atoi(cl) : PP_CHASE_LINES;
        if (lines == 0u || lines > PP_CHASE_LINES) lines = PP_CHASE_LINES;
        std::vector<uint32_t> perm(lines);
        for (uint32_t i = 0; i < lines; ++i) perm[i] = i;
        std::mt19937 crng(0xC0FFEEu);
        std::shuffle(perm.begin(), perm.end(), crng);
        for (uint32_t i = 0; i < lines; ++i) {
            host_arena.chase[(size_t)perm[i] * 8u] = perm[(i + 1u) % lines];
        }
        std::printf("[probe] clock_iters=%u chase_steps=%u over %u lines (%zu KiB of the arena)\n",
                    host_arena.ctl.clock_iters, host_arena.ctl.chase_steps, lines,
                    (size_t)lines * PP_LINE / 1024);
    }
    host_arena.ctl.iters = iters;
    host_arena.ctl.warmup = warmup;
    host_arena.ctl.spin_budget = 100000000ULL;
    // get_sys_cnt is 1 ns/tick on a5, so this is ~4 s -- comfortably inside the
    // 45 s STARS op-execute timeout that would otherwise kill aicpu-sd.
    host_arena.ctl.aicore_run_budget_ticks = 30000ULL * 1000 * 1000;  // ~30 s, under the 45 s STARS kill
    host_arena.ctl.aicpu_deadline_ticks = 25000ULL * 1000 * 1000;    // ~25 s: AICPU gives up before the AICore
    host_arena.ctl.throttle_ticks = 20000;  // ~20 us: 107 parked blocks polling faster would outweigh the measurement

    // ---- AICPU half: dispatcher bootstrap, only when measuring ----
    DevBuf dev_dispatcher, dev_inner, dev_args;
    rtFuncHandle init_handle = nullptr, run_handle = nullptr;
    if (need_aicpu) {
        ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "dispatcher GM");
        ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "inner GM");
        ACL_CHECK(dev_args.Alloc(sizeof(PpDeviceArgs)), "DeviceArgs GM");
        ACL_CHECK(
            aclrtMemcpy(dev_dispatcher.ptr, dispatcher_bytes.size(), dispatcher_bytes.data(),
                        dispatcher_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D dispatcher"
        );
        ACL_CHECK(
            aclrtMemcpy(dev_inner.ptr, inner_bytes.size(), inner_bytes.data(), inner_bytes.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D inner"
        );
        char hostargs[sizeof(PpDeviceArgs)] = {};
        WriteU64(hostargs, kBootstrapDispBin, reinterpret_cast<uint64_t>(dev_dispatcher.ptr));
        WriteU64(hostargs, kBootstrapDispLen, dispatcher_bytes.size());
        WriteU64(hostargs, kBootstrapDevId, (uint64_t)device_id);
        WriteU64(hostargs, kBootstrapInnerBin, reinterpret_cast<uint64_t>(dev_inner.ptr));
        WriteU64(hostargs, kBootstrapInnerLen, inner_bytes.size());
        ACL_CHECK(
            aclrtMemcpy(dev_args.ptr, sizeof(PpDeviceArgs), hostargs, sizeof(PpDeviceArgs),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D DeviceArgs(bootstrap)"
        );

        struct Args {
            struct {
                uint64_t unused[5] = {0};
                uint64_t device_args_ptr = 0;
                uint64_t pad[20] = {0};
            } k_args;
            char kernel_name[32];
            char so_name[32];
            char op_name[32];
        } bargs = {};
        bargs.k_args.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
        std::strncpy(bargs.kernel_name, "DynTileFwkKernelServerInit", sizeof(bargs.kernel_name) - 1);
        std::strncpy(bargs.so_name, "libaicpu_extend_kernels.so", sizeof(bargs.so_name) - 1);
        rtAicpuArgsEx_t bootstrap_args = {};
        bootstrap_args.args = &bargs;
        bootstrap_args.argsSize = sizeof(bargs);
        bootstrap_args.kernelNameAddrOffset = offsetof(Args, kernel_name);
        bootstrap_args.soNameAddrOffset = offsetof(Args, so_name);
        RT_CHECK(
            rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1,
                                          &bootstrap_args, nullptr, stream_aicpu, 0),
            "rtAicpuKernelLaunchExWithArgs(bootstrap)"
        );
        ACL_CHECK(aclrtSynchronizeStream(stream_aicpu), "sync after bootstrap");

        const uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
        char b[64];
        std::snprintf(b, sizeof(b), "simpler_inner_%016lx_%d.so", fp, device_id);
        const std::string so_basename = b;
        std::printf("[bootstrap] driver SO uploaded as %s (fp=%016lx)\n", so_basename.c_str(), fp);

        char json_path_buf[160];
        std::snprintf(json_path_buf, sizeof(json_path_buf), "/tmp/pp_aicore_%016lx_%d.json", fp, getpid());
        const std::string json_path = json_path_buf;
        {
            std::ofstream f(json_path);
            if (!f.is_open()) return 1;
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
        char init_op[128], run_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(run_op, sizeof(run_op), "simpler_aicpu_run_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "rtsFuncGetByName init");
        RT_CHECK(rtsFuncGetByName(binary_handle, run_op, &run_handle), "rtsFuncGetByName run");
    }

    std::printf("[launch] mode=%u blockDim=%u slots=(%u,%u) cpus=(%d,%d) iters=%llu warmup=%llu\n", mode, block_dim,
                slot_a, slot_b, cpu_a, cpu_b, (unsigned long long)iters, (unsigned long long)warmup);

    for (uint32_t a = 0; a < arena_ptrs.size(); ++a) {
        void *const arena = arena_ptrs[a];
        // Each sweep starts from a zeroed arena and a round counter back at 0;
        // ctl, slot_list and pair_list are device-read-only, so the host copy
        // read back from the previous sweep still holds the schedule.
        host_arena.sched.round = 0;
        if (mode == PP_MODE_CORE2CORE && !c2c_all.empty()) {
            const uint32_t n_chunks = (uint32_t)((c2c_all.size() + c2c_chunk_rounds - 1u) / c2c_chunk_rounds);
            const uint32_t chunk = a % n_chunks;
            const size_t first = (size_t)chunk * c2c_chunk_rounds;
            const size_t len = std::min((size_t)c2c_chunk_rounds, c2c_all.size() - first);
            for (size_t i = 0; i < len; ++i) host_arena.pair_list[i] = c2c_all[first + i];
            host_arena.ctl.n_rounds = (uint32_t)len;
            std::printf("[c2c] arena %u runs chunk %u of %u: pairs %zu..%zu\n", a, chunk, n_chunks, first,
                        first + len - 1u);
        }
        ACL_CHECK(aclrtMemset(arena, sizeof(PpArena), 0, sizeof(PpArena)), "memset arena");
        ACL_CHECK(
            aclrtMemcpy(arena, offsetof(PpArena, attendance), &host_arena, offsetof(PpArena, attendance),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "H2D ctl + slot_list"
        );
        if (need_aicpu) {
            PpDeviceArgs run_args = {};
            run_args.arena_addr = reinterpret_cast<uint64_t>(arena);
            run_args.mode = mode;
            ACL_CHECK(
                aclrtMemcpy(dev_args.ptr, sizeof(run_args), &run_args, sizeof(run_args), ACL_MEMCPY_HOST_TO_DEVICE),
                "H2D DeviceArgs(run)"
            );
        }
        std::printf("\n=== arena %u of %zu : dev=0x%016llx ===\n", a, arena_ptrs.size(),
                    (unsigned long long)arena);

        struct AicoreArgs {
            uint64_t arena_ptr;
        } aa;
        aa.arena_ptr = reinterpret_cast<uint64_t>(arena);
        rtArgsEx_t rt_args = {};
        rt_args.args = &aa;
        rt_args.argsSize = sizeof(aa);
        rtTaskCfgInfo_t cfg_aicore = {};
        cfg_aicore.schemMode = RT_SCHEM_MODE_BATCH;
        RT_CHECK(
            rtKernelLaunchWithHandleV2(aicore_handle, 0, block_dim, &rt_args, nullptr, stream_aicore, &cfg_aicore),
            "rtKernelLaunchWithHandleV2(partner)"
        );

        if (need_aicpu) {
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
            RT_CHECK(rtsLaunchCpuKernel(init_handle, 1u, stream_aicpu, &cfg, &cpu_args), "rtsLaunchCpuKernel(init)");
            ACL_CHECK(aclrtSynchronizeStream(stream_aicpu), "sync init");
            RT_CHECK(rtsLaunchCpuKernel(run_handle, 16u, stream_aicpu, &cfg, &cpu_args), "rtsLaunchCpuKernel(run)");
        }

        ACL_CHECK(aclrtSynchronizeStream(stream_aicore), "sync aicore");
        if (need_aicpu) ACL_CHECK(aclrtSynchronizeStream(stream_aicpu), "sync aicpu");

        ACL_CHECK(
            aclrtMemcpy(&host_arena, sizeof(PpArena), arena, sizeof(PpArena), ACL_MEMCPY_DEVICE_TO_HOST),
            "D2H arena"
        );

        // ---- Report ----
        uint32_t attended = 0, aic_seen = 0, aiv_seen = 0, per_die[2] = {0, 0};
        uint64_t first_entry = UINT64_MAX, last_exit = 0;
        std::vector<uint32_t> present;
        for (uint32_t s = 0; s < PP_MAX_SLOTS; ++s) {
            const PpAttendance &e = host_arena.attendance[s];
            if (e.attended != PP_ATTEND_MAGIC) continue;
            present.push_back(s);
            ++attended;
            PP_SLOT_IS_AIC(s) ? ++aic_seen : ++aiv_seen;
            ++per_die[PP_SLOT_DIE(s)];
            const uint64_t en = e.t_entry, ex = e.t_exit;
            if (en < first_entry) first_entry = en;
            if (ex > last_exit) last_exit = ex;
        }
        std::printf("\n  attended: %u  (AIC %u, AIV %u)   die0 %u / die1 %u\n", attended, aic_seen, aiv_seen,
                    per_die[0], per_die[1]);
        const uint32_t expected = block_dim * 3u;
        if (attended != expected) {
            std::printf("  WARNING: expected %u attending slots (blockDim*3), saw %u\n", expected, attended);
        }
        // The occupied slot ids are the part's real topology, and they are what
        // PP_SLOT_DIE / PP_SLOT_IS_AIC / PP_SLOT_CLUSTER have to agree with. On a
        // part with fewer clusters the space stays 108 wide and thins out, so the
        // list is printed rather than summarised.
        if (mode == PP_MODE_CENSUS) {
            std::printf("  occupied slots (%zu):", present.size());
            for (size_t i = 0; i < present.size(); ++i) std::printf(" %u", present[i]);
            std::printf("\n");
        }
        if (attended > 0) {
            std::printf("  dispatch span: %llu ticks\n", (unsigned long long)(last_exit - first_entry));
        }

        if (mode == PP_MODE_PROBE) {
            const uint64_t via_dcci = host_arena.core_ball[0].token;
            const uint64_t via_stdev = host_arena.core_ball[1].token;
            std::printf("\n=== st_dev visibility probe ===\n");
            std::printf("  dcci + CACHELINE_OUT : 0x%016llx  %s\n", (unsigned long long)via_dcci,
                        via_dcci == PP_PROBE_MAGIC ? "VISIBLE" : "NOT VISIBLE");
            std::printf("  st_dev + dsb         : 0x%016llx  %s\n", (unsigned long long)via_stdev,
                        via_stdev == PP_PROBE_MAGIC ? "VISIBLE" : "NOT VISIBLE");
        }

        if (mode == PP_MODE_CORE2CORE) {
            const uint32_t nr = host_arena.ctl.n_rounds;
            uint32_t done = 0, failed = 0;
            std::printf("\n=== core2core sweep: %u rounds over %u slots ===\n", nr, host_arena.ctl.n_slots);
            std::printf("C2C,arena,slot_a,slot_b,die_a,die_b,kind_a,kind_b,rel,ns\n");
            for (uint32_t r = 0; r < nr && r < PP_MAX_ROUNDS; ++r) {
                const uint64_t ticks = host_arena.result_ticks[r];
                const uint32_t entry = host_arena.pair_list[r];
                const uint32_t sa = host_arena.slot_list[(entry >> 16) & 0xFFFFu];
                const uint32_t sb = host_arena.slot_list[entry & 0xFFFFu];
                if (ticks == 0) {
                    ++failed;
                    continue;
                }
                ++done;
                const char *rel = (PP_SLOT_DIE(sa) != PP_SLOT_DIE(sb))
                                      ? "chip"
                                      : (PP_SLOT_CLUSTER(sa) == PP_SLOT_CLUSTER(sb) ? "cluster" : "die");
                std::printf("C2C,%u,%u,%u,%u,%u,%s,%s,%s,%.2f\n", a, sa, sb, PP_SLOT_DIE(sa), PP_SLOT_DIE(sb),
                            SlotKind(sa), SlotKind(sb), rel, (double)ticks / (double)iters);
            }
            std::printf("\n  rounds completed: %u   failed: %u   of %u scheduled\n", done, failed, nr);
            uint32_t blocks_served = 0;
            for (uint32_t sl = 0; sl < PP_MAX_SLOTS; ++sl) {
                if (host_arena.attendance[sl].rounds_served > 0) ++blocks_served;
            }
            std::printf("  AICore blocks that served at least one round: %u\n", blocks_served);
            std::printf("PROBE,arena,slot,die,kind,clock_ticks,chase_ticks,chase_steps\n");
            for (uint32_t sl = 0; sl < PP_MAX_SLOTS; ++sl) {
                const PpCoreProbe &pr = host_arena.core_probe[sl];
                if (pr.attended != PP_ATTEND_MAGIC) continue;
                std::printf("PROBE,%u,%u,%u,%s,%llu,%llu,%u\n", a, sl, PP_SLOT_DIE(sl), SlotKind(sl),
                            (unsigned long long)pr.clock_ticks, (unsigned long long)pr.chase_ticks,
                            host_arena.ctl.chase_steps);
            }
        }

        if (mode == PP_MODE_MEASURE) {
            const uint32_t nr = host_arena.ctl.n_rounds;
            uint32_t done = 0, failed = 0;
            std::printf("\n=== sweep: %u rounds (%u cpus x %u slots) ===\n", nr, host_arena.ctl.n_cpus,
                        host_arena.ctl.n_slots);
            std::printf("CSV,arena,cpu,slot,die,kind,ns\n");
            for (uint32_t r = 0; r < nr && r < PP_MAX_ROUNDS; ++r) {
                const uint64_t ticks = host_arena.result_ticks[r];
                const int32_t c = host_arena.result_cpu[r];
                const uint32_t sl = host_arena.result_slot[r];
                if (ticks == 0 && host_arena.result_rc[r] == 0 && c == 0) continue;
                if (host_arena.result_rc[r] != 0) {
                    ++failed;
                    continue;
                }
                ++done;
                const double ns = (double)ticks / (double)iters;  // cntvct_el0 is 1 GHz on a5
                std::printf("CSV,%u,%d,%u,%u,%s,%.2f\n", a, c, sl, PP_SLOT_DIE(sl), SlotKind(sl), ns);
            }
            std::printf("\n  rounds completed: %u   failed: %u   of %u scheduled\n", done, failed, nr);
            std::printf("\n=== AICPU threads ===\n");
            for (uint32_t i = 0; i < PP_MAX_CPUS; ++i) {
                const PpCpuState &c = host_arena.cpu_state[i];
                if (c.attended != PP_ATTEND_MAGIC) continue;
                std::printf("  idx %u cpu%-3d rounds_served=%-6llu trips=%-8llu timer=%llu Hz\n", i, c.cpu_id,
                            (unsigned long long)c.elapsed_ticks, (unsigned long long)c.trips,
                            (unsigned long long)c.timer_hz);
            }
            uint32_t blocks_served = 0;
            for (uint32_t sl = 0; sl < PP_MAX_SLOTS; ++sl) {
                if (host_arena.attendance[sl].rounds_served > 0) ++blocks_served;
            }
            std::printf("  AICore blocks that served at least one round: %u\n", blocks_served);

            std::printf("PROBE,arena,slot,die,kind,clock_ticks,chase_ticks,chase_steps\n");
            for (uint32_t sl = 0; sl < PP_MAX_SLOTS; ++sl) {
                const PpCoreProbe &pr = host_arena.core_probe[sl];
                if (pr.attended != PP_ATTEND_MAGIC) continue;
                std::printf("PROBE,%u,%u,%u,%s,%llu,%llu,%u\n", a, sl, PP_SLOT_DIE(sl), SlotKind(sl),
                            (unsigned long long)pr.clock_ticks, (unsigned long long)pr.chase_ticks,
                            host_arena.ctl.chase_steps);
            }
            std::printf("CPUPROBE,arena,cpu,clock_ticks,chase_ticks,chase_steps\n");
            for (uint32_t i = 0; i < PP_MAX_CPUS; ++i) {
                const PpCpuState &c = host_arena.cpu_state[i];
                if (c.attended != PP_ATTEND_MAGIC) continue;
                std::printf("CPUPROBE,%u,%d,%llu,%llu,%u\n", a, c.cpu_id, (unsigned long long)c.clock_ticks,
                            (unsigned long long)c.chase_ticks, host_arena.ctl.chase_steps);
            }
        }
    }

    for (void *p : arena_ptrs) {
        if (p) (void)aclrtFree(p);
    }
    (void)aclrtDestroyStream(stream_aicore);
    if (need_aicpu) (void)aclrtDestroyStream(stream_aicpu);
    (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
    return 0;
}
