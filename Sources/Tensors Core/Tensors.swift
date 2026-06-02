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

/// Namespace for L3 composed tensor operations.
///
/// `Tensors` is the L3 composition layer over the L1 `Tensor` namespace from
/// `swift-tensor-primitives`. Composition follows the institute's three-layer
/// pattern per [ECO-002] / [ECO-006]: L3 composes L1 primitives with L2 spec
/// implementations and other L1 packages; L3 does NOT introduce new value
/// types.
///
/// ## Composition Scope
///
/// At L3 this namespace contributes:
/// - Transcendental element-wise operations on real-valued tensors via
///   `swift-iso-9899` (libm) per [PLAT-ARCH-008j].
/// - Real-type reductions (mean, variance, stdev, norm) that compose the
///   intrinsic L1 reductions with libm-driven transcendentals.
/// - Batched matrix multiplication overloads.
///
/// ## Layered Design
///
/// L1 `swift-tensor-primitives` owns the irreducible `Tensor.Value` type and
/// its intrinsic operations (subscript, slice, transpose, reshape, broadcast,
/// basic arithmetic, simple reductions, naive matmul). This L3 layer extends
/// the same type with operations that require libm or compose multiple L1
/// packages. The L1 type is unchanged; consumers reach the unified surface
/// through a single `import Tensors`.
public enum Tensors {}
