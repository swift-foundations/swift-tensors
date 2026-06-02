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
    /// Element-wise hyperbolic cosine (`cosh xᵢ`).
    ///
    /// Routes per-element through `Element._cosh(_:)` from
    /// `Numeric.Transcendental`, which binds the libm `cosh`/`coshf` family
    /// at the L1/L2 boundary per [PLAT-ARCH-008j].
    @inlinable
    public func cosh() -> Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> {
        self.map { (x: Element) -> Element in Element._cosh(x) }
    }
}
