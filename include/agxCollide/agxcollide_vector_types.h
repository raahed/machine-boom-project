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

#ifndef AGXCOLLIDE_VECTOR_H
#define AGXCOLLIDE_VECTOR_H

#include <agx/Vector.h>
#include <agx/ref_ptr.h>
#include <agxCollide/GeometryPair.h>

namespace agxCollide {

  struct BroadPhasePair;
  class Convex;
  class Geometry;
  class Shape;
  class GeometryContact;
  class ParticleContact;
  class ParticleGeometryContact;

  // Types using vector
  // typedef agx::Vector<BroadPhasePair *>                   BroadPhasePairVector;
  typedef agx::VectorPOD<Geometry*>                          GeometryPtrVector;
  typedef agx::Vector<agxCollide::GeometryPair >          GeometryPairVector;
  typedef agx::Vector< agx::ref_ptr<Geometry> >           GeometryRefVector;
  typedef agx::Vector< agx::ref_ptr<Shape> >              ShapeRefVector;
  typedef agx::VectorPOD<GeometryContact *>                  GeometryContactPtrVector;
  typedef agx::Vector<GeometryContact>                    GeometryContactVector;

}
#endif
