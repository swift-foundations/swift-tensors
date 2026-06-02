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
    Element: Copyable & BinaryFloatingPoint,
    Layout == Tensor.Layout.Order.Row
{
    /// Population variance of all elements.
    ///
    /// Computes `Σ(xᵢ − μ)² / n` where `μ` is the arithmetic mean. Uses the
    /// population (biased) formula; consumers wanting the sample (Bessel-
    /// corrected) variance can compose `variance() * n / (n − 1)`.
    ///
    /// Composes the L1 `mean()` and `map(_:)` surface: the mean is computed
    /// first, then a per-element `(x − μ)²` transform is reduced via L1's
    /// `sum()`. No internal storage is touched — the operation is entirely
    /// public-surface composition.
    ///
    /// - Returns: The variance of every element. For an empty tensor the
    ///   result is `nan` (mean is `nan`).
    @inlinable
    public func variance() -> Element {
        let m: Element = self.mean()
        let squaredDeviations: Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> =
            self.map { (x: Element) -> Element in
                let deviation = x - m
                return deviation * deviation
            }
        let total: Element = squaredDeviations.sum()
        let count = Element(Int(bitPattern: self.shape.count))
        return total / count
    }
}
