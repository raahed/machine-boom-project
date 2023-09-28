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


#include <agx/Referenced.h>
#include <agx/Vec3.h>
#include <agxWire/Node.h>
#include <agxWire/ShapeContactNode.h>
#include <agxWire/WireUtils.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxSDK
{
  class SimulationProxy;
}

namespace agxCollide
{
  class Shape;
  class GeometryContactPtr;
  class Geometry;
}

namespace agxWire
{
  class Wire;
  class WireGeometryController;

  typedef agx::Vector<agxCollide::GeometryContact> GeometryContactVector;
  typedef agx::HashVector< Node*, GeometryContactVector > NodePtrGeometryContactVectorTable;
  typedef agx::Vector< std::pair< ShapeContactNode*, agx::Real> > ShapeContactRealPairVector;
  typedef agx::HashVector<ShapeContactNodeRef, ShapeContactNodeRef> NodeMergePairHashVector;

  typedef agx::HashSet<agxCollide::Shape*> ShapeSet;
  typedef agx::HashTable<ShapeContactNode*, ShapeSet > NodeShapeSetVector;

  class AGXPHYSICS_EXPORT WireContactController : public agx::Referenced
  {
    public:
      WireContactController( Wire* wire );

      virtual agx::Bool handleGeometryContact(agxCollide::GeometryContact* geometryContact, agxCollide::Geometry* otherGeometry) = 0;

      agx::Vec3 calculateContactNodeShapeEdgeOffset(ContactNode* cln, agxCollide::Shape* shape, const agx::Real& wireRadius) const;

      virtual void preCollideUpdate() = 0;
      virtual void preUpdate() = 0;
      virtual void postUpdate() = 0;

      static Node* changeToContact(Node* node, Wire* wire, agxCollide::Shape* shape, const agx::Vec3& cPoint,
                                   const agx::Vec3& cNormal, const agx::Real& cDepth, const agx::Bool removeNode);

      const agxSDK::SimulationProxy* getSimulationProxy() const;
      agxWire::WireGeometryController* getGeometryController();
      const agxWire::WireGeometryController* getGeometryController() const;
      agxWire::WireDistanceCompositeConstraint* getWireDistanceCompositeConstraint();
      const agxWire::WireDistanceCompositeConstraint* getWireDistanceCompositeConstraint() const;

    protected:

      void clearTempData();

      virtual ~WireContactController();


    protected:
      Wire* m_wire;
  };

  typedef agx::ref_ptr< WireContactController > WireContactControllerRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
