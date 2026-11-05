// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// firtool's (CIRCT's) module prefix annotation. The FIRRTL annotation file names it by its
// class, so this one must live in package sifive.enterprise.firrtl: firtool prefixes the
// target module (inclusive) and every module under it, the extracted memories too, with
// `prefix`.
package sifive.enterprise.firrtl

import firrtl.annotations.ModuleTarget
import firrtl.annotations.SingleTargetAnnotation

case class NestedPrefixModulesAnnotation(target: ModuleTarget, prefix: String, inclusive: Boolean)
    extends SingleTargetAnnotation[ModuleTarget] {
  def duplicate(n: ModuleTarget): NestedPrefixModulesAnnotation = this.copy(target = n)
}
