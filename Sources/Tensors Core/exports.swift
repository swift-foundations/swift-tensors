// ===----------------------------------------------------------------------===//
//
// This source file is part of the swift-tensors open source project
//
// Copyright (c) 2026 Coen ten Thije Boonkkamp and the swift-tensors project authors
// Licensed under Apache License v2.0
//
// See LICENSE for license information
//
// ===----------------------------------------------------------------------===//

// MARK: - External dependency re-exports per [MOD-001] / [MOD-002]
//
// Core is the dependency funnel. Variant targets depend only on Core; the
// consumer-facing types from L1 / L2 reach them via these re-exports. The
// L3 composition layer composes L1 `Tensor` + L2 `ISO_9899` libm + L1
// `Complex` + L1 `Numeric` per [PLAT-ARCH-008j] / [ECO-002] / [ECO-006].

@_exported public import Complex_Primitives
@_exported public import ISO_9899_Core
@_exported public import Numeric_Primitives
@_exported public import Numerics
@_exported public import Real_Primitives
@_exported public import Tensor_Primitives
@_exported public import Vector_Primitives
