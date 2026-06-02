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
    /// Element-wise natural logarithm (`ln xᵢ`).
    ///
    /// Routes per-element through `Element._log(_:)` from
    /// `Numeric.Transcendental`, which binds the libm `log`/`logf` family
    /// at the L1/L2 boundary per [PLAT-ARCH-008j].
    ///
    /// - Returns: A row-major tensor with `ln xᵢ` per element. For non-positive
    ///   `xᵢ`, the result is implementation-defined per ISO 9899:2018 §7.12.6.7
    ///   (typically `nan` for negative input, `-inf` for zero).
    @inlinable
    public func log() -> Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> {
        self.map { (x: Element) -> Element in Element._log(x) }
    }
}
