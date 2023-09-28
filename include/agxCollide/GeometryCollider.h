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

#ifndef AGXCOLLIDE_GEOMETRYCOLLIDER_H
#define AGXCOLLIDE_GEOMETRYCOLLIDER_H

#include <agx/agxPhysics_export.h>
#include <agxCollide/ShapeCollider.h>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4290 )
#endif

namespace agxCollide
{
  class Geometry;

  /**
  The geometry collider is responsible for finding contacts between two geometries. Each geometry
  may be composed of multiple shapes, so the geometry collider tests the individual shape bounds
  against each other and use shape colliders to test overlapping primitives.

  This class is not used in the standard pipeline anymore, but can be used for manual collision detection.
  */
  class AGXPHYSICS_EXPORT GeometryCollider
  {
  public:
    /**
    Calculates two geometries against each other.
    Note that contact normals will point from the second to the first geometry.
    Contact reduction and explicit contact area computation are only done if enabled in the parameters.
    If the GeometryCollider is used to replicate the behavior of the default collision detection pipeline,
    it might be advisable to check if these parameters are available in the contact material for the two geometries.
    The material, hasInnerMaterial and solveImpact-properties of the LocalGeometryContact are not modified.
    \param geom1 The first geometry
    \param geom2 The second geometry
    \param result The resulting contact. Will be cleared before computation.
    \param useContactAreaApproach If true, will try to calculate the contact area and distribute it evenly
      over the points. If false, will distribute unit area over the contact points.
    \param contactReductionEnable Should contacts be reduced?
    \param contactReductionThreshold At which number of contact points should we start reducing?
    \param contactReductionBinResolution The bin resolution for contact reduction
      (check agxCollide::ContactReducer for details).
    */
    static void calculateContacts(
      Geometry* geom1,
      Geometry* geom2,
      LocalGeometryContact& result,
      const bool useContactAreaApproach = false,
      const agx::Bool& contactReductionEnable = false,
      const agx::UInt& contactReductionThreshold = 12,
      const agx::UInt& contactReductionBinResolution = 3);

    /**
    Calculates two geometries against each other.
    Note that contact normals will point from the second to the first geometry.
    Contact reduction and explicit contact area computation are only done if enabled in the parameters.
    If the GeometryCollider is used to replicate the behavior of the default collision detection pipeline,
    it might be advisable to check if these parameters are available in the contact material for the two geometries.
    The material, hasInnerMaterial and solveImpact-properties of the LocalGeometryContact are not modified.
    \param geom1 The first geometry
    \param geom2 The second geometry
    \retval result The resulting contact.
    \param useContactAreaApproach If true, will try to calculate the contact area and distribute it evenly
    over the points. If false, will distribute unit area over the contact points.
    \param contactReductionEnable Should contacts be reduced?
    \param contactReductionThreshold At which number of contact points should we start reducing?
    \param contactReductionBinResolution The bin resolution for contact reduction
    (check agxCollide::ContactReducer for details).
    */
    static LocalGeometryContact calculateContacts(
      Geometry* geom1,
      Geometry* geom2,
      const bool useContactAreaApproach = false,
      const agx::Bool& contactReductionEnable = false,
      const agx::UInt& contactReductionThreshold = 12,
      const agx::UInt& contactReductionBinResolution = 3);
  };
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* AGXCOLLIDE_GEOMETRYCOLLIDER_H */
