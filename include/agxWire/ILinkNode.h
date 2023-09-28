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

#include <agxWire/Node.h>

namespace agxWire
{
  AGX_DECLARE_POINTER_TYPES( ILinkNode );

  /**
  Interface class for nodes connected to links.
  */
  class AGXPHYSICS_EXPORT ILinkNode : public agxWire::ConnectingNode
  {
    public:
      /**
      Object holding parameters, such as bend and twist stiffness,
      related to wire to link connections.
      */
      class AGXPHYSICS_EXPORT ConnectionProperties
      {
        public:
          /**
          Default constructor, with default bend and twist stiffness disabled.
          */
          ConnectionProperties();

          /**
          Construct given bend- and twist stiffness.
          */
          ConnectionProperties( agx::Real bendStiffness, agx::Real twistStiffness );

          /**
          Assign bend stiffness of the link to wire connection. The bend stiffness is
          interpreted as Young's modulus of the wire. I.e., the final stiffness
          is dependent on the wire radius and segment lengths. Default: 0.
          \param bendStiffness - new bend stiffness for this connection
          */
          void setBendStiffness( agx::Real bendStiffness );

          /**
          \return the currently used bend stiffness of this connection (default: 0)
          */
          agx::Real getBendStiffness() const;

          /**
          Assign twist stiffness of the link to wire connection. The twist stiffness is
          interpreted as Young's modulus of the wire. I.e., the final stiffness
          is dependent on the wire radius. Currently, the segment length is assumed
          to be 1 meter and default value of the stiffness is 0.
          \param twistStiffness - new twist stiffness for this connection
          */
          void setTwistStiffness( agx::Real twistStiffness );

          /**
          \return the currently used twist stiffness of this connection (default: 0)
          */
          agx::Real getTwistStiffness() const;

          DOXYGEN_START_INTERNAL_BLOCK()
          agx::Real getBendDamping() const;
          void store( agxStream::OutputArchive& out ) const;
          void restore( agxStream::InputArchive& in );
          DOXYGEN_END_INTERNAL_BLOCK()

        private:
          agx::Real m_bendStiffness;
          agx::Real m_bendDamping;
          agx::Real m_twistStiffness;
      };

    public:
      /**
      \return the connection properties for the connection between the wire and the link
      */
      virtual agxWire::ILinkNode::ConnectionProperties* getConnectionProperties() const = 0;

      /**
      \return other link connection if present, otherwise null
      */
      virtual agxWire::ILinkNode* getLinkConnection() const = 0;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxWire::ILinkNode );

    protected:
      /**
      Default protected constructor used during restore of this object.
      */
      ILinkNode();

      /**
      Implementation constructor.
      */
      ILinkNode( agx::RigidBody* rb, const agx::Vec3& relTranslate );
  };
}

