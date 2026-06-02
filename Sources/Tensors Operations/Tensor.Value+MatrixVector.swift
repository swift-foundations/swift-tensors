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
    Element: Copyable & Swift.Numeric,
    Layout == Tensor.Layout.Order.Row,
    Rank == 2
{
    /// Matrix-vector product for a rank-2 matrix and a rank-1 vector.
    ///
    /// Computes `y[i] = Σⱼ A[i, j] · x[j]` for `A: (m, n)` and `x: (n)`.
    /// Output shape is `(m)`.
    ///
    /// Composes the L1 rank-2 `multiplied(by:)` kernel via column-promotion:
    /// the input vector is reshaped to a column matrix `(n, 1)`, the L1
    /// rank-2 matmul yields a column result `(m, 1)`, then a reshape produces
    /// the rank-1 output `(m)`. The intermediate copies are row-major-
    /// contiguous, so each reshape is structural metadata adjustment plus an
    /// element-preserving copy in the L1 implementation.
    ///
    /// - Parameter vector: The rank-1 vector of shape `(n)`.
    /// - Returns: A row-major rank-1 tensor of shape `(m)`.
    /// - Throws: `Tensor.Broadcast.Error.incompatibleShapes` when the matrix
    ///   inner dimension and vector length don't match.
    @inlinable
    public func multiplied(
        by vector: borrowing Tensor.Value<Element, 1, Layout>
    ) throws(Tensor.Broadcast.Error) -> Tensor.Value<Element, 1, Tensor.Layout.Order.Row> {
        let m = self.shape.dims[0]
        let n = self.shape.dims[1]
        let vectorLength = vector.shape.dims[0]

        if n != vectorLength {
            throw .incompatibleShapes(axis: .one, lhs: n, rhs: vectorLength)
        }

        // Reshape vector (n) → matrix (n, 1).
        var columnDims = InlineArray<2, Cardinal>(repeating: .zero)
        columnDims[0] = n
        columnDims[1] = .one
        let columnShape = Tensor.Shape<2>(columnDims)
        // Per-construction the element count is preserved (n · 1 = n); the
        // typed-throws reshape's error path is therefore unreachable. Use
        // a do/catch with explicit guard rather than `try!` per [IMPL-108].
        let xMatrix: Tensor.Value<Element, 2, Tensor.Layout.Order.Row>
        do throws(Tensor.Reshape.Error) {
            xMatrix = try vector.reshape(to: columnShape)
        } catch {
            // n · 1 always equals n; this branch is unreachable for any
            // construction respecting the matrix-inner-dim guard above.
            preconditionFailure("vector → column-matrix reshape preserves element count by construction")
        }

        // L1 rank-2 matmul: (m, n) · (n, 1) → (m, 1).
        let productMatrix: Tensor.Value<Element, 2, Tensor.Layout.Order.Row> =
            try self.multiplied(by: xMatrix)

        // Reshape result (m, 1) → vector (m).
        var resultDims = InlineArray<1, Cardinal>(repeating: .zero)
        resultDims[0] = m
        let resultShape = Tensor.Shape<1>(resultDims)
        do throws(Tensor.Reshape.Error) {
            return try productMatrix.reshape(to: resultShape)
        } catch {
            // m · 1 always equals m; unreachable for the same reason.
            preconditionFailure("(m, 1) → (m) reshape preserves element count by construction")
        }
    }
}
