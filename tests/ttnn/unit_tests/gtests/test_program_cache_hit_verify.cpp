// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <variant>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/cleanup.hpp>

#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::device_operation::test {
namespace {

// Bakes `baked_value` into a compile-time argument but leaves it out of its program hash: the bug
// verify_program_cache_hits exists to catch.
struct MisHashedOp {
    struct operation_attributes_t {
        uint32_t baked_value = 0;
    };
    struct tensor_args_t {
        Tensor input;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct Factory {
        struct shared_variables_t {};
        using cached_program_t = CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/, tensor_return_value_t&) {
            tt::tt_metal::Program program;
            tt::tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
                CoreCoord{0, 0},
                tt::tt_metal::DataMovementConfig{.compile_args = {attrs.baked_value}});
            return cached_program_t(std::move(program), shared_variables_t{});
        }

        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}
    };
    using program_factory_t = std::variant<Factory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return ttsl::hash::hash_objects_with_default_seed(tensor_args.input.tensor_spec());
    }
};

class ProgramCacheHitVerifyTest : public TTNNFixtureWithDevice {
protected:
    Tensor make_input() const {
        return create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape{1, 1, 32, 32},
                tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{})),
            device_);
    }

    static auto verify_hits(bool enabled) {
        const bool previous = ttnn::CONFIG.get<"verify_program_cache_hits">();
        ttnn::CONFIG.set<"verify_program_cache_hits">(enabled);
        return ttsl::make_cleanup([previous]() { ttnn::CONFIG.set<"verify_program_cache_hits">(previous); });
    }
};

TEST_F(ProgramCacheHitVerifyTest, MissingHashFieldIsSilentWhenOff) {
    const auto restore = verify_hits(false);
    const auto input = make_input();
    launch<MisHashedOp>({.baked_value = 1}, {.input = input});
    EXPECT_NO_THROW(launch<MisHashedOp>({.baked_value = 2}, {.input = input}));
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);
}

TEST_F(ProgramCacheHitVerifyTest, MissingHashFieldThrowsWhenOn) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    launch<MisHashedOp>({.baked_value = 1}, {.input = input});
    EXPECT_NO_THROW(launch<MisHashedOp>({.baked_value = 1}, {.input = input}));
    EXPECT_THROW(launch<MisHashedOp>({.baked_value = 2}, {.input = input}), std::exception);
}

TEST_F(ProgramCacheHitVerifyTest, HonestHitsPass) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    ttnn::add(input, input);
    EXPECT_NO_THROW(ttnn::add(input, input));
    EXPECT_NO_THROW(ttnn::multiply(input, input));
    EXPECT_NO_THROW(ttnn::multiply(input, input));
}

}  // namespace
}  // namespace ttnn::device_operation::test
