public import Tensors_Core

extension Tensor.Value
where
    Element: Copyable & BinaryFloatingPoint & Numeric.Transcendental,
    Layout == Tensor.Layout.Order.Row
{

    @inlinable
    public func stdev() -> Element {
        Element._sqrt(self.variance())
    }
}
