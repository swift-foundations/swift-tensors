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
    Element: Copyable & BinaryFloatingPoint & Numeric.Transcendental,
    Layout == Tensor.Layout.Order.Row
{
    /// The p-norm of all elements: `(Σ |xᵢ|^p)^(1/p)`.
    ///
    /// Common values:
    /// - `p = 1`: the Manhattan / 1-norm (`Σ |xᵢ|`).
    /// - `p = 2`: the Euclidean / 2-norm (`√(Σ xᵢ²)`).
    ///
    /// For `p → ∞`, the limit is `max |xᵢ|`; consumers needing the ∞-norm can
    /// compose `tensor.map { $0.magnitude }.maximum() ?? .zero` on the L1
    /// public surface.
    ///
    /// Composes the L1 `map(_:)` and `sum()` surface with `Element._pow(_:_:)`
    /// from `Numeric.Transcendental`, which binds the libm `pow`/`powf` family
    /// at the L1/L2 boundary per [PLAT-ARCH-008j].
    ///
    /// - Parameter p: The norm parameter. Must be a positive real value; the
    ///   result is implementation-defined per IEEE-754 for non-positive `p`.
    /// - Returns: The p-norm of every element.
    @inlinable
    public func norm(p: Element) -> Element {
        let raised: Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> =
            self.map { (x: Element) -> Element in
                let absolute = x.magnitude
                return Element._pow(absolute, p)
            }
        let total: Element = raised.sum()
        return Element._pow(total, 1 / p)
    }
}
