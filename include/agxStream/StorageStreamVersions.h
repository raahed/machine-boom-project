/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#pragma once

#include <agx/Integer.h>

namespace agxStream
{
  const agx::UInt16 STORAGESTREAM_VERSION_HINGE_SWING = 1; // Hinge based on Swing instead of Dot1.
  const agx::UInt16 STORAGESTREAM_VERSION_INDEXSET_STATE = 2; // Store indexset state from previous solve
  const agx::UInt16 STORAGESTREAM_VERSION_CONSTRAINT_ATTACHMENTS = 3; // Store attachment pairs for constraints
  const agx::UInt16 STORAGESTREAM_VERSION_AMOR_PARALLELIZATION = 4; // Merge split thresholds changes.
  const agx::UInt16 STORAGESTREAM_VERSION_ELECTRIC_MOTOR = 5; // Added electric motor
  const agx::UInt16 STORAGESTREAM_VERSION_COMBUSTION_ENGINE = 6; // Added combustion engine
  const agx::UInt16 STORAGESTREAM_VERSION_FRICTION_CONTROLLER = 7; // Elementary constraint friction controller.
  const agx::UInt16 STORAGESTREAM_VERSION_COMBUSTION_ENGINE_STARTER = 8; // Added combustion engine starter
  const agx::UInt16 STORAGESTREAM_VERSION_COMBUSTION_ENGINE_STABILITY_FIX = 9; // Change engine solver and fix stability problems
  const agx::UInt16 STORAGESTREAM_VERSION_NEW_TORQUE_CONVERTER = 10; // Replace old torque converter with new one
  const agx::UInt16 STORAGESTREAM_VERSION_FRICTION_CONTROLLER_STATIC_FORCE = 11; // Add "static" friction force to friction controller
  const agx::UInt16 STORAGESTREAM_VERSION_LINEARIZATION_ON_CONSTRAINT = 12; // Enable linearization flag moved from LockJoint to Constraint.
}

