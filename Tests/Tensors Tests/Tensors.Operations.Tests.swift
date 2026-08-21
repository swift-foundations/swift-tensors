import Tensors_Test_Support
import Testing

@Suite
struct `Tensors Operations Tests` {
    @Suite struct Unit {}
    @Suite struct `Edge Case` {}
    @Suite struct Integration {}
    @Suite(.serialized) struct Performance {}
}

private func rank1Tensor(
    _ elements: [Double]
) -> Tensor.Value<Double, 1, Tensor.Layout.Order.Row> {
    var dims = InlineArray<1, Cardinal>(repeating: .zero)
    dims[0] = Cardinal(UInt(elements.count))
    let shape = Tensor.Shape<1>(dims)
    let result: Tensor.Value<Double, 1, Tensor.Layout.Order.Row>
    do throws(Tensor.Shape<1>.Error) {
        result = try Tensor.Value<Double, 1, Tensor.Layout.Order.Row>(
            shape: shape,
            elements: elements
        )
    } catch {
        preconditionFailure("rank1Tensor: element count by construction")
    }
    return result
}

private func rank2Tensor(
    _ rows: Int,
    _ cols: Int,
    _ elements: [Double]
) -> Tensor.Value<Double, 2, Tensor.Layout.Order.Row> {
    var dims = InlineArray<2, Cardinal>(repeating: .zero)
    dims[0] = Cardinal(UInt(rows))
    dims[1] = Cardinal(UInt(cols))
    let shape = Tensor.Shape<2>(dims)
    let result: Tensor.Value<Double, 2, Tensor.Layout.Order.Row>
    do throws(Tensor.Shape<2>.Error) {
        result = try Tensor.Value<Double, 2, Tensor.Layout.Order.Row>(
            shape: shape,
            elements: elements
        )
    } catch {
        preconditionFailure("rank2Tensor: element count by construction")
    }
    return result
}

private func rank3Tensor(
    _ batches: Int,
    _ rows: Int,
    _ cols: Int,
    _ elements: [Double]
) -> Tensor.Value<Double, 3, Tensor.Layout.Order.Row> {
    var dims = InlineArray<3, Cardinal>(repeating: .zero)
    dims[0] = Cardinal(UInt(batches))
    dims[1] = Cardinal(UInt(rows))
    dims[2] = Cardinal(UInt(cols))
    let shape = Tensor.Shape<3>(dims)
    let result: Tensor.Value<Double, 3, Tensor.Layout.Order.Row>
    do throws(Tensor.Shape<3>.Error) {
        result = try Tensor.Value<Double, 3, Tensor.Layout.Order.Row>(
            shape: shape,
            elements: elements
        )
    } catch {
        preconditionFailure("rank3Tensor: element count by construction")
    }
    return result
}

private func readElement<Element: Copyable, Layout: Tensor.Layout.`Protocol`>(
    from tensor: borrowing Tensor.Value<Element, 2, Layout>,
    at i: Int,
    _ j: Int
) throws(Tensor.Index.Error) -> Element {
    var positions = InlineArray<2, Ordinal>(repeating: Ordinal(0))
    positions[0] = Ordinal(UInt(i))
    positions[1] = Ordinal(UInt(j))
    let position = Tensor.Index.Position<2>(positions)
    return try tensor.element(at: position)
}

private func readElement<Element: Copyable, Layout: Tensor.Layout.`Protocol`>(
    from tensor: borrowing Tensor.Value<Element, 1, Layout>,
    at i: Int
) throws(Tensor.Index.Error) -> Element {
    var positions = InlineArray<1, Ordinal>(repeating: Ordinal(0))
    positions[0] = Ordinal(UInt(i))
    let position = Tensor.Index.Position<1>(positions)
    return try tensor.element(at: position)
}

private func readElement<Element: Copyable, Layout: Tensor.Layout.`Protocol`>(
    from tensor: borrowing Tensor.Value<Element, 3, Layout>,
    at b: Int,
    _ i: Int,
    _ j: Int
) throws(Tensor.Index.Error) -> Element {
    var positions = InlineArray<3, Ordinal>(repeating: Ordinal(0))
    positions[0] = Ordinal(UInt(b))
    positions[1] = Ordinal(UInt(i))
    positions[2] = Ordinal(UInt(j))
    let position = Tensor.Index.Position<3>(positions)
    return try tensor.element(at: position)
}

private let elementWiseTolerance: Double = 1e-12

extension `Tensors Operations Tests`.Unit {
    @Test
    func `exp at zero equals one for every element`() throws(Tensor.Index.Error) {
        let zeros = rank1Tensor([0.0, 0.0, 0.0])
        let result = zeros.exp()
        #expect(try readElement(from: result, at: 0) == 1.0)
        #expect(try readElement(from: result, at: 1) == 1.0)
        #expect(try readElement(from: result, at: 2) == 1.0)
    }

    @Test
    func `log of e equals one`() throws(Tensor.Index.Error) {
        let e = 2.718281828459045
        let input = rank1Tensor([e])
        let result = input.log()
        let value = try readElement(from: result, at: 0)
        #expect((value - 1.0).magnitude < elementWiseTolerance)
    }

    @Test
    func `sqrt of four equals two`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([4.0, 9.0, 16.0])
        let result = input.sqrt()
        #expect(try readElement(from: result, at: 0) == 2.0)
        #expect(try readElement(from: result, at: 1) == 3.0)
        #expect(try readElement(from: result, at: 2) == 4.0)
    }

    @Test
    func `sin at zero equals zero`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([0.0])
        let result = input.sin()
        #expect(try readElement(from: result, at: 0) == 0.0)
    }

    @Test
    func `cos at zero equals one`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([0.0])
        let result = input.cos()
        #expect(try readElement(from: result, at: 0) == 1.0)
    }

    @Test
    func `tan at zero equals zero`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([0.0])
        let result = input.tan()
        #expect(try readElement(from: result, at: 0) == 0.0)
    }

    @Test
    func `sinh at zero equals zero`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([0.0])
        let result = input.sinh()
        #expect(try readElement(from: result, at: 0) == 0.0)
    }

    @Test
    func `cosh at zero equals one`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([0.0])
        let result = input.cosh()
        #expect(try readElement(from: result, at: 0) == 1.0)
    }

    @Test
    func `tanh at zero equals zero`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([0.0])
        let result = input.tanh()
        #expect(try readElement(from: result, at: 0) == 0.0)
    }
}

extension `Tensors Operations Tests`.Unit {
    @Test
    func `mean of one through four equals two point five`() {
        let input = rank1Tensor([1.0, 2.0, 3.0, 4.0])
        let result = input.mean()
        #expect(result == 2.5)
    }

    @Test
    func `variance of one through five equals two`() {

        let input = rank1Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        let result = input.variance()
        #expect((result - 2.0).magnitude < elementWiseTolerance)
    }

    @Test
    func `stdev of one through five equals sqrt of two`() {
        let input = rank1Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        let result = input.stdev()
        let expected = 1.4142135623730951
        #expect((result - expected).magnitude < elementWiseTolerance)
    }

    @Test
    func `two norm of three four equals five`() {

        let input = rank1Tensor([3.0, 4.0])
        let result = input.norm(p: 2.0)
        #expect((result - 5.0).magnitude < elementWiseTolerance)
    }

    @Test
    func `one norm sums absolute values`() {
        let input = rank1Tensor([-1.0, 2.0, -3.0, 4.0])
        let result = input.norm(p: 1.0)
        #expect((result - 10.0).magnitude < elementWiseTolerance)
    }

}

extension `Tensors Operations Tests`.Unit {
    @Test
    func `matrix-vector product computes y equal A x`() {

        let a = rank2Tensor(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let x = rank1Tensor([10.0, 20.0, 30.0])

        do throws(Tensor.Broadcast.Error) {
            let y = try a.multiplied(by: x)
            #expect(y.shape.dims[0] == Cardinal(2))
            do throws(Tensor.Index.Error) {
                #expect(try readElement(from: y, at: 0) == 140.0)
                #expect(try readElement(from: y, at: 1) == 320.0)
            } catch {
                #expect(Bool(false), "Unexpected index error: \(error)")
            }
        } catch {
            #expect(Bool(false), "Unexpected broadcast error: \(error)")
        }
    }

    @Test
    func `batched matmul of two 2x3 by two 3x2 produces two 2x2 results`() {

        let a = rank3Tensor(
            2,
            2,
            3,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ]
        )
        let b = rank3Tensor(
            2,
            3,
            2,
            [
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]
        )
        do throws(Tensor.Broadcast.Error) {
            let c = try a.multiplied(batchedBy: b)
            #expect(c.shape.dims[0] == Cardinal(2))
            #expect(c.shape.dims[1] == Cardinal(2))
            #expect(c.shape.dims[2] == Cardinal(2))
            do throws(Tensor.Index.Error) {
                #expect(try readElement(from: c, at: 0, 0, 0) == 58.0)
                #expect(try readElement(from: c, at: 0, 0, 1) == 64.0)
                #expect(try readElement(from: c, at: 0, 1, 0) == 139.0)
                #expect(try readElement(from: c, at: 0, 1, 1) == 154.0)
                #expect(try readElement(from: c, at: 1, 0, 0) == 58.0)
                #expect(try readElement(from: c, at: 1, 1, 1) == 154.0)
            } catch {
                #expect(Bool(false), "Unexpected index error: \(error)")
            }
        } catch {
            #expect(Bool(false), "Unexpected broadcast error: \(error)")
        }
    }
}

extension `Tensors Operations Tests`.`Edge Case` {
    @Test
    func `matrix-vector multiply with mismatched inner dim throws`() {
        let a = rank2Tensor(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let x = rank1Tensor([10.0, 20.0])
        do throws(Tensor.Broadcast.Error) {
            _ = try a.multiplied(by: x)
            #expect(Bool(false), "Expected throw")
        } catch let error {
            switch error {
            case .incompatibleShapes:
                break

            default:
                #expect(Bool(false), "Wrong error variant")
            }
        }
    }

    @Test
    func `batched matmul with mismatched batch sizes throws`() {
        let a = rank3Tensor(
            2,
            2,
            3,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ]
        )
        let b = rank3Tensor(3, 3, 2, Array(repeating: 1.0, count: 18))
        do throws(Tensor.Broadcast.Error) {
            _ = try a.multiplied(batchedBy: b)
            #expect(Bool(false), "Expected throw")
        } catch let error {
            switch error {
            case .incompatibleShapes(let axis, _, _):
                #expect(axis == .zero)

            default:
                #expect(Bool(false), "Wrong error variant")
            }
        }
    }

    @Test
    func `batched matmul with mismatched inner dim throws`() {
        let a = rank3Tensor(
            2,
            2,
            3,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ]
        )
        let b = rank3Tensor(2, 4, 2, Array(repeating: 1.0, count: 16))
        do throws(Tensor.Broadcast.Error) {
            _ = try a.multiplied(batchedBy: b)
            #expect(Bool(false), "Expected throw")
        } catch let error {
            switch error {
            case .incompatibleShapes(let axis, _, _):
                #expect(axis == Cardinal(2 as UInt))

            default:
                #expect(Bool(false), "Wrong error variant")
            }
        }
    }

}

extension `Tensors Operations Tests`.Integration {
    @Test
    func `exp then log returns approximately the input`() throws(Tensor.Index.Error) {
        let input = rank1Tensor([1.0, 2.0, 3.0])
        let roundTrip = input.exp().log()
        let v0 = try readElement(from: roundTrip, at: 0)
        let v1 = try readElement(from: roundTrip, at: 1)
        let v2 = try readElement(from: roundTrip, at: 2)
        #expect((v0 - 1.0).magnitude < elementWiseTolerance)
        #expect((v1 - 2.0).magnitude < elementWiseTolerance)
        #expect((v2 - 3.0).magnitude < elementWiseTolerance)
    }

    @Test
    func `sin squared plus cos squared equals one`() {

        let input = rank1Tensor([0.5, 1.5, 2.5])
        let sines = input.sin()
        let cosines = input.cos()
        do throws(Tensor.Broadcast.Error) {

            let sinSquared = try sines.multiplying(elementWise: sines)
            let cosSquared = try cosines.multiplying(elementWise: cosines)
            let one = try sinSquared.adding(cosSquared)
            do throws(Tensor.Index.Error) {
                let v0 = try readElement(from: one, at: 0)
                let v1 = try readElement(from: one, at: 1)
                let v2 = try readElement(from: one, at: 2)
                #expect((v0 - 1.0).magnitude < elementWiseTolerance)
                #expect((v1 - 1.0).magnitude < elementWiseTolerance)
                #expect((v2 - 1.0).magnitude < elementWiseTolerance)
            } catch {
                #expect(Bool(false), "Unexpected index error: \(error)")
            }
        } catch {
            #expect(Bool(false), "Unexpected broadcast error: \(error)")
        }
    }

    @Test
    func `stdev squared equals variance`() {
        let input = rank1Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        let v = input.variance()
        let s = input.stdev()
        #expect((s * s - v).magnitude < elementWiseTolerance)
    }
}
