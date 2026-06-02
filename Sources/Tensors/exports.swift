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

// MARK: - Umbrella per [MOD-005]
//
// The umbrella target's sole role is to re-export Core + variants so consumers
// can write a single `import Tensors` and reach the complete L3 surface.

@_exported public import Tensors_Core
@_exported public import Tensors_Operations
