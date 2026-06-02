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
    /// Element-wise exponential (`e^xᵢ`).
    ///
    /// Routes per-element through `Element._exp(_:)` from
    /// `Numeric.Transcendental`, which binds the libm `exp`/`expf`/`exp2`
    /// family at the L1/L2 boundary per [PLAT-ARCH-008j]. The libm
    /// implementation lives in `swift-iso-9899` (L2) and is re-exposed
    /// through `Numeric.Transcendental` at L1 for the real types
    /// (`Double`, `Float`, `Float16` where available).
    ///
    /// - Returns: A row-major tensor of the same shape with `e^xᵢ` per
    ///   element.
    @inlinable
    public func exp() -> Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> {
        self.map { (x: Element) -> Element in Element._exp(x) }
    }
}
