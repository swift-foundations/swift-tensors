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
    /// Population standard deviation of all elements.
    ///
    /// Computes `√variance()`. Composes the `variance()` reduction with
    /// `Element._sqrt(_:)` from `Numeric.Transcendental` per [PLAT-ARCH-008j].
    ///
    /// - Returns: The standard deviation of every element. For an empty
    ///   tensor the result is `nan` (variance is `nan`).
    @inlinable
    public func stdev() -> Element {
        Element._sqrt(self.variance())
    }
}
