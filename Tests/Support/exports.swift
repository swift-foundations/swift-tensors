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

// MARK: - SLI Spine per [MOD-024]
//
// Test Support extends the SLI re-export chain so downstream test fixtures see
// the literal conformances declared in upstream Standard Library Integration
// targets. The spine anchor is `Tensor Primitives Test Support` — the lowest
// in-scope dep that already chains the spine through Buffer, Tagged, etc.

@_exported public import Tensors
@_exported public import Tensor_Primitives_Test_Support
