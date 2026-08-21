public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & BinaryFloatingPoint & Numeric.Transcendental,
    Layout == Tensor.Layout.Order.Row
{

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
