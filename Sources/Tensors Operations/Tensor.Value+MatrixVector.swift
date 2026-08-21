public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & Swift.Numeric,
    Layout == Tensor.Layout.Order.Row,
    Rank == 2
{

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

        var columnDims = InlineArray<2, Cardinal>(repeating: .zero)
        columnDims[0] = n
        columnDims[1] = .one
        let columnShape = Tensor.Shape<2>(columnDims)

        let xMatrix: Tensor.Value<Element, 2, Tensor.Layout.Order.Row>
        do throws(Tensor.Reshape.Error) {
            xMatrix = try vector.reshape(to: columnShape)
        } catch {

            preconditionFailure(
                "vector → column-matrix reshape preserves element count by construction"
            )
        }

        let productMatrix: Tensor.Value<Element, 2, Tensor.Layout.Order.Row> =
            try self.multiplied(by: xMatrix)

        var resultDims = InlineArray<1, Cardinal>(repeating: .zero)
        resultDims[0] = m
        let resultShape = Tensor.Shape<1>(resultDims)
        do throws(Tensor.Reshape.Error) {
            return try productMatrix.reshape(to: resultShape)
        } catch {

            preconditionFailure("(m, 1) → (m) reshape preserves element count by construction")
        }
    }
}
