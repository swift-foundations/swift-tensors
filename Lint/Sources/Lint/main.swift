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

// Lint executable — consumer of the institute-tier rule bundle. One bundle,
// one engine call. New rules added at the institute / universal tiers
// propagate here on the next dependency-resolution; this file does not need
// editing.

internal import Linter
internal import Linter_Institute_Rules

Lint.run(bundle: Lint.Rule.Bundle.institute)
