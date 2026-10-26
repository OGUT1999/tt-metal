// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/program_fingerprint.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::program_fingerprint {

namespace {

std::string cores_to_string(const CoreRangeSet& cores) {
    std::vector<std::string> ranges;
    ranges.reserve(cores.ranges().size());
    for (const auto& range : cores.ranges()) {
        ranges.push_back(range.str());
    }
    return fmt::format("{}", fmt::join(ranges, ","));
}

std::string describe_kernel(const Kernel& kernel) {
    const auto& source = kernel.kernel_source();
    const std::string source_id = source.source_type_ == KernelSource::FILE_PATH
                                      ? source.source_
                                      : fmt::format("inline#{:x}", std::hash<std::string>{}(source.source_));

    std::vector<uint32_t> compile_time_args;
    kernel.process_compile_time_args([&](const std::vector<uint32_t>& values) { compile_time_args = values; });

    std::map<std::string, uint32_t> named_compile_time_args;
    kernel.process_named_compile_time_args([&](const std::unordered_map<std::string, uint32_t>& named_args) {
        named_compile_time_args.insert(named_args.begin(), named_args.end());
    });
    std::vector<std::string> named;
    named.reserve(named_compile_time_args.size());
    for (const auto& [name, value] : named_compile_time_args) {
        named.push_back(fmt::format("{}={}", name, value));
    }

    std::map<std::string, std::string> defines;
    kernel.process_defines([&](const std::string& define, const std::string& value) { defines[define] = value; });
    std::vector<std::string> define_strings;
    define_strings.reserve(defines.size());
    for (const auto& [define, value] : defines) {
        define_strings.push_back(fmt::format("{}={}", define, value));
    }

    return fmt::format(
        "kernel {} source={} cores={} processor_class={} compile_args=[{}] named_args=[{}] defines=[{}] opt={}/{} "
        "settings_hash={:x}",
        kernel.name(),
        source_id,
        cores_to_string(kernel.core_range_set()),
        static_cast<int>(kernel.get_kernel_processor_class()),
        fmt::join(compile_time_args, ","),
        fmt::join(named, ","),
        fmt::join(define_strings, ","),
        kernel.get_compiler_opt_level(),
        kernel.get_linker_opt_level(),
        kernel.compute_hash());
}

std::string describe_circular_buffer(const CircularBufferImpl& cb) {
    std::vector<uint8_t> indices(cb.buffer_indices().begin(), cb.buffer_indices().end());
    std::sort(indices.begin(), indices.end());
    std::vector<std::string> per_index;
    per_index.reserve(indices.size());
    for (const auto index : indices) {
        const auto& tile = cb.tile(index);
        per_index.push_back(fmt::format(
            "{}:page={},format={},tile={}",
            index,
            cb.page_size(index),
            static_cast<int>(cb.data_format(index)),
            tile.has_value() ? fmt::format("{}x{}", tile->get_height(), tile->get_width()) : "none"));
    }
    return fmt::format(
        "circular_buffer cores={} size={} globally_allocated={} global_cb={} indices=[{}]",
        cores_to_string(cb.core_ranges()),
        cb.size(),
        cb.globally_allocated(),
        cb.is_global_circular_buffer(),
        fmt::join(per_index, ";"));
}

}  // namespace

std::vector<std::string> describe(Program& program) {
    auto& impl = program.impl();
    std::vector<std::string> lines;

    const auto& hal = MetalContext::instance(impl.get_context_id()).hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        // Handles are assigned in creation order, so sorting by handle is stable across rebuilds.
        const auto& kernels = impl.get_kernels(index);
        std::vector<std::pair<KernelHandle, const Kernel*>> sorted;
        sorted.reserve(kernels.size());
        for (const auto& [handle, kernel] : kernels) {
            sorted.emplace_back(handle, kernel.get());
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [handle, kernel] : sorted) {
            lines.push_back(fmt::format("core_type {} handle {} {}", index, handle, describe_kernel(*kernel)));
        }
    }

    for (const auto& cb : impl.circular_buffers()) {
        lines.push_back(describe_circular_buffer(*cb));
    }

    for (const auto& semaphore : impl.semaphores()) {
        lines.push_back(fmt::format(
            "semaphore id={} cores={} core_type={} initial_value={}",
            semaphore.id(),
            cores_to_string(semaphore.core_range_set()),
            static_cast<int>(semaphore.core_type()),
            semaphore.initial_value()));
    }
    return lines;
}

std::vector<std::string> describe(distributed::MeshWorkload& workload) {
    std::vector<std::pair<distributed::MeshCoordinateRange, Program*>> programs;
    for (auto& [range, program] : workload.get_programs()) {
        programs.emplace_back(range, &program);
    }
    std::sort(programs.begin(), programs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::string> lines;
    for (const auto& [range, program] : programs) {
        std::ostringstream prefix;
        prefix << range;
        for (auto& line : describe(*program)) {
            lines.push_back(fmt::format("{} {}", prefix.str(), line));
        }
    }
    return lines;
}

}  // namespace tt::tt_metal::experimental::program_fingerprint
