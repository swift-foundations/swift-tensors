public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & BinaryFloatingPoint,
    Layout == Tensor.Layout.Order.Row
{

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
