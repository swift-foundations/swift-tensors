public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & Numeric.Transcendental,
    Layout == Tensor.Layout.Order.Row
{

    @inlinable
    public func log() -> Tensor.Value<Element, Rank, Tensor.Layout.Order.Row> {
        self.map { (x: Element) -> Element in Element._log(x) }
    }
}
