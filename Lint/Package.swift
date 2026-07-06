// swift-tools-version: 6.3.1

// ===----------------------------------------------------------------------===//
//
// This source file is part of the swift-tensors open source project
//
// Copyright (c) 2026 Coen ten Thije Boonkkamp and the swift-tensors project authors
// Licensed under Apache License v2.0
//
// See LICENSE for license information
//
// ===----------------------------------------------------------------------===//

// Lint executable — consumer of the institute-tier rule bundle. New rules
// added to swift-institute-linter-rules / swift-linter-rules propagate here
// on the next dependency-resolution; this file does not need editing.

import PackageDescription

let package = Package(
    name: "Lint",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .executable(
            name: "Lint",
            targets: ["Lint"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/swift-foundations/swift-linter.git", branch: "main"),
        .package(url: "https://github.com/swift-foundations/swift-institute-linter-rules.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "Lint",
            dependencies: [
                .product(name: "Linter", package: "swift-linter"),
                .product(name: "Linter Institute Rules", package: "swift-institute-linter-rules"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)

for target in package.targets where ![.system, .binary, .plugin, .macro].contains(target.type) {
    let ecosystem: [SwiftSetting] = [
        .enableUpcomingFeature("ExistentialAny"),
        .enableUpcomingFeature("InternalImportsByDefault"),
        .enableUpcomingFeature("MemberImportVisibility"),
        .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    ]

    target.swiftSettings = (target.swiftSettings ?? []) + ecosystem
}
