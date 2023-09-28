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

#include <agx/agxPhysics_export.h>
#include <agx/Vec3.h>
#include <agx/Matrix3x3.h>
#include <agx/Matrix4x4.h>

#include <agxCollide/Quadric.h>

namespace agxCollide {

  class Shape;


  DOXYGEN_START_INTERNAL_BLOCK()
  /**
  Descriptor for one or more Quadric Surface(s) for a certain shape.

  A descriptor can only be created for some agxCollide::Shape subclasses and should
  be created via the static method createShapeDescriptor and not by filling in the members
  manually.
  */
  typedef struct AGXPHYSICS_EXPORT QuadricShapeDescriptor {

    // enum used for indexing radiuses/positions
    enum PlaneIndex : agx::UInt8 {
      TopPlane     = 0,
      BottomPlane  = 1,
      ClosestPlane = 2
    };


    /// Pointer to shape. Set to nullptr for invalid descriptors.
    agxCollide::Shape* ptr;       //

    bool  hollow;                 //

    agx::Real radiuses[2];        // Top and bottom radius. Same for cylinder, different for cone.

    agx::Real thickness;          // Valid if shape is hollow

    agx::Real positions[2];       // Position for top- and bottom endcap for outer Quad

    agx::Real innerTopPosition;   // Position for top endcap for inner Quad


    agxCollide::Quadric3D outerQ; // Quadric for outside of shape.
    agxCollide::Quadric3D innerQ; // Quadrid for inside of shape if hollow.


    // Constructor.
    QuadricShapeDescriptor();


    inline agx::Real getRadius(   PlaneIndex idx ) const { return radiuses[idx]; }
    inline agx::Real getPosition( PlaneIndex idx ) const { return positions[idx]; }

    inline agx::Real getHeight() const { return std::fabs( positions[0] - positions[1] ); }


    /**
    Create a descriptor for shape
    */
    static QuadricShapeDescriptor createShapeDescriptor( agxCollide::Shape* shapePtr );


  } QuadDesc;
  DOXYGEN_END_INTERNAL_BLOCK()

}

