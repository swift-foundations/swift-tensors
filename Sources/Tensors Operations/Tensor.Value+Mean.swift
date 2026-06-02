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
    /// Mean of all elements (`Σxᵢ / n`).
    ///
    /// Composes the L1 sum-reduction (over `Element: AdditiveArithmetic`)
    /// with a division by the element count. The element count is taken from
    /// the tensor's shape via the L1 `Cardinal → Int` typed-bit-pattern bridge
    /// and converted to `Element` via `BinaryFloatingPoint`'s
    /// `BinaryInteger` initializer.
    ///
    /// - Returns: The arithmetic mean of every element. For an empty tensor
    ///   the result is `nan` (IEEE-754 division by zero).
    @inlinable
    public func mean() -> Element {
        let total: Element = self.sum()
        // Cardinal → Int via institute bit-pattern bridge; Int → Element via
        // stdlib BinaryInteger initializer. `shape.count` is non-negative,
        // so the bit-pattern reinterpretation cannot produce a negative Int
        // for any practically-representable tensor.
        let count = Element(Int(bitPattern: self.shape.count))
        return total / count
    }
}
