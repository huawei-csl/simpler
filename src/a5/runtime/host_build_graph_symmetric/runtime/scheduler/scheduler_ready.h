#pragma once

// host_build_graph_symmetric replaces only the scheduler. This header is
// host_build_graph's, forwarded so the runtime is complete under the
// src/{arch}/runtime/{name}/{runtime,common} layout the kernel compiler
// derives its include paths from.
#include "../../../host_build_graph/runtime/scheduler/scheduler_ready.h"
