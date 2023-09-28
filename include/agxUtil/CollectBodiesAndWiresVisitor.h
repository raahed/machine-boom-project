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

#ifndef AGXUTIL_COLLECT_BODIES_AND_WIRES_VISITOR_H
#define AGXUTIL_COLLECT_BODIES_AND_WIRES_VISITOR_H

#include <agx/agx.h>
#include <agxWire/Wire.h>

namespace agxUtil {

  /**
  This visitor will visit each rigid body and wire in an assembly (or collection) and extract them to a vector
  with all bodies.
  Excluded are bodies which has isPowerlineBody() == true
  Bodies are stored as raw pointers. Do not store and use the result between time steps as bodies will come and go.
  Also, make sure the wire is added to the simulation (initialized).
  */
  class AGXPHYSICS_EXPORT CollectBodiesAndWiresVisitor : public agxSDK::AssemblyVisitor
  {
  public:

    using WirePtrSetVector = agx::SetVector<agxWire::Wire*>;
    using RigidBodyPtrSetVector = agx::SetVector<agx::RigidBody*>;
    using WireNodeRefVector = agx::Vector<agxWire::NodeRef>;

    /// Constructor
    CollectBodiesAndWiresVisitor() {}

    /// Destructor
    virtual ~CollectBodiesAndWiresVisitor() {}

    /// \return the vector of all collected bodies
    RigidBodyPtrSetVector& getBodies() { return m_bodies; }
    /// \return the vector of all collected bodies
    const RigidBodyPtrSetVector& getBodies() const { return m_bodies; }

    /// \return the vector of all collected wires
    WirePtrSetVector& getWires() { return m_wires; }
    /// \return the vector of all collected wires
    const WirePtrSetVector& getWires() const { return m_wires; }

    /// \return vector with wire nodes fixed to other (non-wire-specific) objects
    const WireNodeRefVector& getUserWireNodes() const { return m_userWireNodes; }

  protected:

    using agxSDK::AssemblyVisitor::visit;

    virtual void visit(agx::RigidBody* body)
    {
      // Ignore powerline bodies
      if (!body->isPowerlineBody())
        m_bodies.insert(body);
    }

    virtual void visit(agxSDK::Assembly* assembly)
    {
      assembly->traverse(this);
    }

    // Visit "wires"
    virtual void visit(agxSDK::EventListener* listener)
    {
      if ( listener == nullptr )
        return;

      auto wire = listener->asSafe<agxWire::Wire>();
      if (wire == nullptr || m_wires.contains( wire ) || wire->getConstraint() == nullptr)
        return;

      const auto constraint = wire->getConstraint();
      for (auto it = constraint->getBeginIterator(); it != constraint->getEndIterator(); ++it) {
        auto rb = (*it)->getRigidBody();
        if ( agxWire::Wire::isLumpedNode( rb ) )
          m_bodies.insert( rb );
        else
          m_userWireNodes.push_back( *it );
      }

      m_wires.insert(wire);
    }

    virtual void visit(const agx::RigidBody*body) const
    {
      const_cast<CollectBodiesAndWiresVisitor *>(this)->visit(body);
    }

    virtual void visit(const agxSDK::Assembly* assembly)const
    {
      const_cast<CollectBodiesAndWiresVisitor *>(this)->visit(assembly);
    }

    virtual void visit(const agxSDK::EventListener*listener) const
    {
      const_cast<CollectBodiesAndWiresVisitor *>(this)->visit(listener);
    }

    RigidBodyPtrSetVector m_bodies;
    WirePtrSetVector m_wires;
    WireNodeRefVector m_userWireNodes;
  };
}

#endif

