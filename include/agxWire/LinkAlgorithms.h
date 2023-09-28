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

#include <agxWire/Link.h>

#include <agxSDK/ContactEventListener.h>

#include <agx/LockJoint.h>

namespace agxWire
{
  AGX_DECLARE_POINTER_TYPES( LinkObjectStabilizingAlgorithm );
  /**
  (Optional) Link algorithm that tries to stabilize the link when the link is in contact
  with an object. If there's high enough tension in the wire, this algorithm will create
  extra constraints between the link and the contacting object.

  It's not recommended to have this enabled for all objects so an execute filter has to
  be given to define which objects this feature is enabled for.
  */
  class AGXPHYSICS_EXPORT LinkObjectStabilizingAlgorithm : public agxWire::Link::Algorithm
  {
    public:
      class LinkExecuteFilter;
      class GeometryPropertyBoolFilter;
      class RigidBodyPropertyBoolFilter;

      typedef agx::ref_ptr< GeometryPropertyBoolFilter > GeometryPropertyBoolFilterRef;
      typedef agx::ref_ptr< RigidBodyPropertyBoolFilter > RigidBodyPropertyBoolFilterRef;

    public:
      /**
      Construct given filter to filter out which objects should have this feature.
      \param filter - user implemented or pre-defined execute filter
      \param tensionThreshold - tension threshold defining the moment the stabilizing constraints should be applied (default: 1.0E4)
      */
      LinkObjectStabilizingAlgorithm( agxWire::LinkObjectStabilizingAlgorithm::LinkExecuteFilter* filter, agx::Real tensionThreshold = agx::Real( 1E4 ) );

      /**
      Assign new tension threshold defining the moment the stabilizing constraints should be applied.
      \param tensionThreshold - new tension threshold (default: 1.0E4)
      */
      void setTensionThreshold( agx::Real tensionThreshold );

      /**
      \return the current tension threshold
      */
      agx::Real getTensionThreshold() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::LinkObjectStabilizingAlgorithm );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Protected default constructor, used during restore.
      */
      LinkObjectStabilizingAlgorithm();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~LinkObjectStabilizingAlgorithm();

    protected:
      /**
      Adding contact listener if link switched from disabled do enabled, otherwise removing the contact listener.
      */
      virtual void onLinkEnable( agx::Bool enable, agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy ) override;

      /**
      Checks stability and adding stabilizing constraints if needed.
      */
      virtual void pre( agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy ) override;

      /**
      Removing the stabilizing constraints.
      */
      virtual void post( agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy ) override;

    protected:
      typedef agx::Vector< agx::LockJointRef > ConstraintContainer;

    protected:
      agxSDK::ContactEventListenerRef m_contactListener;
      ConstraintContainer m_constraints;
      agx::Real m_tensionThreshold;
  };

  /**
  Base filter for selecting objects to the LinkObjectStabilizingAlgorithm.
  For user implemented filters there's three things you have to do:
    1. Implement the matchOther( const agxCollide::Geometry* ) which should
       return true if this filter gave a match to that geometry, otherwise false.
    2. Implement store for serialization.
    3. Implement restore for de-serialization.
  */
  class AGXPHYSICS_EXPORT LinkObjectStabilizingAlgorithm::LinkExecuteFilter : public agxSDK::ExecuteFilter, public agxStream::Serializable
  {
    public:
      /**
      Construct given link this algorithm belongs to. It needs the link
      because it's the only way to know if this filter has been invalidated.
      \param link - link this execute filter is defined for
      */
      LinkExecuteFilter( const agxWire::Link* link )
        : m_link( link ) {}

      /**
      \return the link this execute filter is defined for
      */
      const agxWire::Link* getLink() const { return m_link; }

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      /**
      Store this object to stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore this object from stream.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxWire::LinkObjectStabilizingAlgorithm::LinkExecuteFilter );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Called when this base filter has determined which geometry that's not part
      of the link - the \p otherGeometry.
      \param otherGeometry - the geometry the link is currently overlapping
      \return true if it's a match, otherwise false
      */
      virtual bool matchOther( const agxCollide::Geometry* otherGeometry ) const = 0;

      using agxSDK::ExecuteFilter::match;

      /**
      Base match method which determines which geometry belongs to the link and who's not.
      */
      virtual bool match( const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 ) const override
      {
        // If the link has been deleted our contact event listener shouldn't still be in the simulation!

        // ^^ But if the link has been removed we will send a match to the internal contact event
        // listener so that the listener can remove itself.
        if ( m_link == nullptr )
          return true;

        agx::Bool firstIsLink = m_link != nullptr && geometry1->getRigidBody() == m_link->getRigidBody();
        agx::Bool secondIsLink = m_link != nullptr && geometry2->getRigidBody() == m_link->getRigidBody();
        return firstIsLink ? matchOther( geometry2 ) : secondIsLink ? matchOther( geometry1 ) : false;
      }

    private:
      agxWire::LinkConstObserver m_link;
  };

  /**
  Execute filter for the LinkObjectStabilizingAlgorithm which trying to match
  a 'property-bool' in the other geometry which the link interacts with.

  Example:
  link->add( new LinkObjectStabilizingAlgorithm( new LinkObjectStabilizingAlgorithm::GeometryPropertyBoolFilter( "LinkStabilize", link ) ) );

  // We want the link to be 'extra' stable while in contact with this geometry:
  geometry->getPropertyContainer()->addPropertyBool( "LinkStabilize", true );
  */
  class AGXPHYSICS_EXPORT LinkObjectStabilizingAlgorithm::GeometryPropertyBoolFilter : public LinkObjectStabilizingAlgorithm::LinkExecuteFilter
  {
    public:
      /**
      Construct matching filter given property-bool name and the link.
      \param propertyName - property-bool for geometry for positive match (geometry->getPropertyContainer()->addPropertyBool( propertyName, true ))
      \param link - link this filter belongs to, in order to filter out which of the two geometries in a geometry contact is a link geometry
      */
      GeometryPropertyBoolFilter( const agx::String& propertyName, const agxWire::Link* link )
        : LinkExecuteFilter( link ), m_propertyName( propertyName ) {}

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::LinkObjectStabilizingAlgorithm::GeometryPropertyBoolFilter );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Protected default constructor, used during restore.
      */
      GeometryPropertyBoolFilter();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~GeometryPropertyBoolFilter();

      /**
      Match function.
      \return true if match, otherwise false
      */
      virtual bool matchOther( const agxCollide::Geometry* otherGeometry ) const override
      {
        return matchT( otherGeometry );
      }

      /**
      Match function for either agxCollide::Geometry or agx::RigidBody.
      \return true if match, otherwise false
      */
      template< typename T >
      agx::Bool matchT( const T* obj ) const
      {
        agx::Bool propertyValue = false;
        return obj != nullptr &&
               obj->hasPropertyContainer() &&
               obj->getPropertyContainer()->getPropertyBool( m_propertyName, propertyValue ) &&
               propertyValue;
      }

    protected:
      agx::String m_propertyName;
  };

  /**
  Execute filter for the LinkObjectStabilizingAlgorithm which trying to match
  a 'property-bool' in the other rigid body which the link interacts with.

  Example:
  link->add( new LinkObjectStabilizingAlgorithm( new LinkObjectStabilizingAlgorithm::RigidBodyPropertyBoolFilter( "LinkStabilize", link ) ) );

  // We want the link to be 'extra' stable while in contact with this body:
  rb->getPropertyContainer()->addPropertyBool( "LinkStabilize", true );
  */
  class AGXPHYSICS_EXPORT LinkObjectStabilizingAlgorithm::RigidBodyPropertyBoolFilter : public LinkObjectStabilizingAlgorithm::GeometryPropertyBoolFilter
  {
    public:
      /**
      Construct matching filter given property-bool name and the link.
      \param propertyName - property-bool for rigid body for positive match (rb->getPropertyContainer()->addPropertyBool( propertyName, true ))
      \param link - link this filter belongs to, in order to filter out which of the two geometries in a geometry contact is a link geometry
      */
      RigidBodyPropertyBoolFilter( const agx::String& propertyName, const agxWire::Link* link )
        : GeometryPropertyBoolFilter( propertyName, link ) {}

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::LinkObjectStabilizingAlgorithm::RigidBodyPropertyBoolFilter );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Protected default constructor, used during restore.
      */
      RigidBodyPropertyBoolFilter();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~RigidBodyPropertyBoolFilter();

      /**
      Match function.
      \return true if match, otherwise false
      */
      virtual bool matchOther( const agxCollide::Geometry* otherGeometry ) const override
      {
        return matchT( otherGeometry->getRigidBody() );
      }
  };
}

