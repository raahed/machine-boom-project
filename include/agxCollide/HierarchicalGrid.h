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

#ifndef AGXCOLLIDE_HIERARCHICALGRID_H
#define AGXCOLLIDE_HIERARCHICALGRID_H

#include <agx/agx_vector_types.h>
#include <agx/Component.h>
#include <agx/SymmetricPair.h>
#include <agxData/Entity.h>
#include <agx/Task.h>


namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(HierarchicalGrid);
  class AGXPHYSICS_EXPORT HierarchicalGrid : public agx::Component
  {
  public:
    HierarchicalGrid();

    typedef agx::SymmetricPair<agx::UInt32> GridOverlap;
    typedef agx::VectorPOD<GridOverlap> GridOverlapVector;

    void findOverlaps(const agx::Vec3Vector& positionVector, const agx::RealVector& radiusVector, GridOverlapVector& result);

  protected:
    virtual ~HierarchicalGrid();

  private:
    agx::TaskRef m_findOverlapsTask;
  };

}


#endif /* AGXCOLLIDE_HIERARCHICALGRID_H */
