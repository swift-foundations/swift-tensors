public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & BinaryFloatingPoint,
    Layout == Tensor.Layout.Order.Row
{

    @inlinable
    public func mean() -> Element {
        let total: Element = self.sum()

        let count = Element(Int(bitPattern: self.shape.count))
        return total / count
    }
}
