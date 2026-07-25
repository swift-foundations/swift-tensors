# swift-tensors

![Development Status](https://img.shields.io/badge/status-active--development-blue.svg)

N-dimensional tensor types and operations for Swift.

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/swift-foundations/swift-tensors.git", branch: "main")
]
```

Add a product to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "Tensors", package: "swift-tensors")
    ]
)
```

## License

Apache 2.0. See [LICENSE](LICENSE.md).
