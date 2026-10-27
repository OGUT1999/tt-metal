# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.unpack_a import unpack_a_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.operand import BfdResource
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
)


def _uses_upk_to_dest_semaphores(config: GlobalConfig) -> bool:
    from helpers.llk_params import PerfRunType

    return not config.quasar_use_dvalid and config.perf_run_type in (
        None,
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )


def upk_to_dest_math_ack() -> str:
    return (
        "_llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::UNPACK_MATH);\n"
        "_llk_sync_get_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::UNPACK_MATH);\n"
    )


def _unp_sel(compute_unit: FpuNode) -> str:
    if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
        return "p_unpacr::UNP_B"
    return "p_unpacr::UNP_A"


def _num_tiles(compute_unit: FpuNode, block: BlockData):
    if compute_unit.reuse_dest != EltwiseBinaryReuseDestType.NONE:
        return 1
    return block.block_cols


class UnpackerA(Unpacker):
    granularity = InvocationGranularity.ROW
    golden_fn = staticmethod(unpack_a_golden)
    supports_dest_offset = False

    per_block_init = True

    def __init__(
        self, reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE
    ):
        if reuse_dest != EltwiseBinaryReuseDestType.NONE:
            self.granularity = InvocationGranularity.TILE

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_unary_operand.h",
            "llk_unpack_unary_operand_to_dest.h",
            "llk_math_common.h",
        ]

    def _perf_valid_args(
        self,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> Tuple[str, str, int]:
        if compute_unit.reuse_dest != EltwiseBinaryReuseDestType.NONE:
            num_faces = compute_unit.src_a.tile_shape.total_num_faces()
            return "true", "true", num_faces
        if config.dest_acc == DestAccumulation.Yes:
            return "true", "true", block.block_cols
        return "true", "false", block.block_cols

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            return ""
        set_a, set_b, count = self._perf_valid_args(config, compute_unit, block)
        return f"_perf_unpack_loop_set_valid<{set_a}, {set_b}>({count});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            return upk_to_dest_math_ack()
        clear_a, clear_b, count = self._perf_valid_args(config, compute_unit, block)
        return f"_perf_math_loop_clear_valid<{clear_a}, {clear_b}>({count});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        engine = (
            BfdResource.UNP1
            if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
            else BfdResource.UNP0
        )
        bfd_alloc = compute_unit.src_a.bfd_alloc_and_program(
            engine, result_name="bfd_id"
        )
        if compute_unit.unpack_to_dest.value:
            # Unpack owns the DEST bank tracking in this path; it is reset on every init.
            return (
                "{\n" + bfd_alloc + "_llk_unpack_dest_init_();\n"
                "_llk_unpack_unary_operand_to_dest_init_(bfd_id);\n"
                "}\n"
            )

        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        transpose_en = compute_unit.transpose_faces.cpp_enum_value
        unp_sel = _unp_sel(compute_unit)

        return (
            "{\n"
            + bfd_alloc
            + f"_llk_unpack_unary_operand_init_<{unp_sel}, {transpose_en}, {en_32bit_dest}, {reuse_dest}>"
            f"(bfd_id, {tensor_shape}, {_num_tiles(compute_unit, block)});\n"
            "}\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            # One DEST section per call, at DEST tile 0. 32-bit DEST is pinned for the SyncHalf bank flip.
            code = (
                "_llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK, semaphore::UNPACK_MATH);\n"
                f"_llk_unpack_unary_operand_to_dest_block_({block.tile_id_src_a}, 0, {_num_tiles(compute_unit, block)});\n"
                "_llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);\n"
            )
            if operation.dest_sync == DestSync.Half:
                code += "_llk_sync_advance_dest_section_<ckernel::to_underlying(ckernel::trisc::TriscID::Unpack), true, p_stall::UNPACK0>();\n"
            return code

        unp_sel = _unp_sel(compute_unit)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value

        return (
            f"_llk_unpack_unary_operand_<{unp_sel}, {reuse_dest}>"
            f"({block.tile_id_src_a}, {tensor_shape});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
