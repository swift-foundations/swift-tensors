// swift-tools-version: 6.3.1
// swift-tensors — composed tensor operations atop swift-tensor-primitives.

import PackageDescription

let package = Package(
    name: "swift-tensors",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .tvOS(.v26),
        .watchOS(.v26),
        .visionOS(.v26),
    ],
    products: [
        .library(name: "Tensors Operations", targets: ["Tensors Operations"]),
        .library(name: "Tensors", targets: ["Tensors"]),
        .library(name: "Tensors Test Support", targets: ["Tensors Test Support"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swift-primitives/swift-tensor-primitives.git", branch: "main"),
        .package(url: "https://github.com/swift-primitives/swift-complex-primitives.git", branch: "main"),
        .package(url: "https://github.com/swift-primitives/swift-numeric-primitives.git", branch: "main"),
        .package(url: "https://github.com/swift-primitives/swift-vector-primitives.git", branch: "main"),
        .package(url: "https://github.com/swift-iso/swift-iso-9899.git", branch: "main"),
        .package(url: "https://github.com/swift-foundations/swift-numerics.git", branch: "main"),
    ],
    targets: [

        // MARK: - Core (internal)
        .target(
            name: "Tensors Core",
            dependencies: [
                .product(name: "Tensor Primitives", package: "swift-tensor-primitives"),
                .product(name: "Complex Primitives", package: "swift-complex-primitives"),
                .product(name: "Numeric Primitives", package: "swift-numeric-primitives"),
                .product(name: "Real Primitives", package: "swift-numeric-primitives"),
                .product(name: "Vector Primitives", package: "swift-vector-primitives"),
                .product(name: "ISO 9899 Core", package: "swift-iso-9899"),
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),

        // MARK: - Operations
        .target(
            name: "Tensors Operations",
            dependencies: ["Tensors Core"]
        ),

        // MARK: - Umbrella
        .target(
            name: "Tensors",
            dependencies: [
                "Tensors Core",
                "Tensors Operations",
            ]
        ),

        // MARK: - Test Support
        .target(
            name: "Tensors Test Support",
            dependencies: [
                "Tensors",
                .product(name: "Tensor Primitives Test Support", package: "swift-tensor-primitives"),
            ],
            path: "Tests/Support"
        ),

        // MARK: - Tests
        .testTarget(
            name: "Tensors Tests",
            dependencies: [
                "Tensors",
                "Tensors Test Support",
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)

for target in package.targets where ![.system, .binary, .plugin, .macro].contains(target.type) {
    let ecosystem: [SwiftSetting] = [
        .strictMemorySafety(),
        .enableUpcomingFeature("ExistentialAny"),
        .enableUpcomingFeature("InternalImportsByDefault"),
        .enableUpcomingFeature("MemberImportVisibility"),
        .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
        .enableUpcomingFeature("InferIsolatedConformances"),
        .enableUpcomingFeature("LifetimeDependence"),
        .enableExperimentalFeature("LifetimeDependence"),
        .enableExperimentalFeature("Lifetimes"),
        .enableExperimentalFeature("SuppressedAssociatedTypes"),
    ]

    target.swiftSettings = (target.swiftSettings ?? []) + ecosystem
}
