public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & Swift.Numeric,
    Layout == Tensor.Layout.Order.Row,
    Rank == 3
{

    @inlinable
    public func multiplied(
        batchedBy other: borrowing Tensor.Value<Element, 3, Layout>
    ) throws(Tensor.Broadcast.Error) -> Tensor.Value<Element, 3, Tensor.Layout.Order.Row> {
        let batchA = self.shape.dims[0]
        let m = self.shape.dims[1]
        let p = self.shape.dims[2]
        let batchB = other.shape.dims[0]
        let pOther = other.shape.dims[1]
        let n = other.shape.dims[2]

        if batchA != batchB {
            throw .incompatibleShapes(axis: .zero, lhs: batchA, rhs: batchB)
        }
        if p != pOther {
            throw .incompatibleShapes(axis: Cardinal(2 as UInt), lhs: p, rhs: pOther)
        }

        var resultDims = InlineArray<3, Cardinal>(repeating: .zero)
        resultDims[0] = batchA
        resultDims[1] = m
        resultDims[2] = n
        let resultShape = Tensor.Shape<3>(resultDims)
        let totalOutput = Int(bitPattern: resultShape.count)

        var elements: [Element] = []
        elements.reserveCapacity(totalOutput)

        let bInt = Int(bitPattern: batchA)
        let mInt = Int(bitPattern: m)
        let pInt = Int(bitPattern: p)
        let nInt = Int(bitPattern: n)

        (0..<bInt).forEach { b in
            (0..<mInt).forEach { i in
                (0..<nInt).forEach { k in
                    var accumulator = Element.zero
                    (0..<pInt).forEach { j in

                        var aPositions = InlineArray<3, Ordinal>(repeating: Ordinal(0))
                        aPositions[0] = Ordinal(UInt(bitPattern: b))
                        aPositions[1] = Ordinal(UInt(bitPattern: i))
                        aPositions[2] = Ordinal(UInt(bitPattern: j))
                        let aPosition = Tensor.Index.Position<3>(aPositions)

                        var bPositions = InlineArray<3, Ordinal>(repeating: Ordinal(0))
                        bPositions[0] = Ordinal(UInt(bitPattern: b))
                        bPositions[1] = Ordinal(UInt(bitPattern: j))
                        bPositions[2] = Ordinal(UInt(bitPattern: k))
                        let bPosition = Tensor.Index.Position<3>(bPositions)

                        let aValue: Element
                        let bValue: Element
                        do throws(Tensor.Index.Error) {
                            aValue = try self.element(at: aPosition)
                            bValue = try other.element(at: bPosition)
                        } catch {
                            preconditionFailure(
                                "batched matmul positions are in-bounds by construction"
                            )
                        }
                        accumulator += aValue * bValue
                    }
                    elements.append(accumulator)
                }
            }
        }

        do throws(Tensor.Shape<3>.Error) {
            return try Tensor.Value<Element, 3, Tensor.Layout.Order.Row>(
                shape: resultShape,
                elements: elements
            )
        } catch {

            preconditionFailure("batched matmul element count matches result shape by construction")
        }
    }
}
