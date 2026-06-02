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

public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & Numeric.Transcendental,
    Layout == Tensor.Layout.Order.Row
{
    /// Element-wise square root (`√xᵢ`).
    ///
    /// Routes per-element through `Element._sqrt(_:)` from
    /// `Numeric.Transcendental`, which binds the libm `sqrt`/`sqrtf` family
    /// (or the IEEE-754 square-root primitive on platforms with hardware
    /// support) at the L1/L2 boundary per [PLAT-ARCH-008j].
    ///
    /// - Returns: A row-major tensor with `√xᵢ` per element. For negative
    ///   `xᵢ`, the result is `nan` per IEEE-754.
    @inlinable
    public func sqrt() -> Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> {
        self.map { (x: Element) -> Element in Element._sqrt(x) }
    }
}
