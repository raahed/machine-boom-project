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

#ifndef AGXMODEL_DEFORMABLE1DNODE_H
#define AGXMODEL_DEFORMABLE1DNODE_H

#include <agxModel/Deformable1DAttachment.h>
#include <agxModel/Deformable1DSegment.h>
#include <agxModel/Tree.h>

namespace agxModel
{
  class Deformable1DGeometryFactory;

  AGX_DECLARE_POINTER_TYPES(Deformable1DNode);

  typedef agx::Vector< Deformable1DNodeRef > Deformable1DNodeRefContainer;


  /**
  Special node implementation of segments in e.g., a Deformable1D object.
  */
  class AGXMODEL_EXPORT Deformable1DNode : public agxModel::Tree::Branch
  {
    public:
      /**
      Structure containing relevant data for elasto-plastic deformation.
      */
      struct AGXMODEL_EXPORT DeformationData
      {
        agx::Vec3 translationalForce; /**< Current translational force in the constraint. */
        agx::Vec3 rotationalForce;    /**< Current rotational force in the constraint. */
        agx::Quat relativeRotation;   /**< Current relative rotation. */

        void store( agxStream::OutputArchive& out ) const;
        void restore( agxStream::InputArchive& in );
        void storeLightData( agxStream::StorageStream& str ) const;
        void restoreLightData( agxStream::StorageStream& str );
      };

    public:
      /**
      Construct given full transform of the rigid body in this node.
      \param transform - full transform of the rigid body in this node
      */
      Deformable1DNode( const agx::AffineMatrix4x4& transform );

      /**
      Construct given full transform of the rigid body in this node and an external
      attachment.
      \param transform - full transform of the rigid body in this node
      \param attachment - external attachment to this node
      */
      Deformable1DNode( const agx::AffineMatrix4x4& transform, agxModel::NodeAttachment* attachment );

      /**
      Position of this node is defined to be at the start of this node given this
      node has an extent.
      \return position of this node
      */
      agx::Vec3 getPosition() const;

      /**
      \return the length of this node
      */
      agx::Real getLength() const;

      /**
      Add an external attachment to this node.
      \param attachment - attachment to add
      \return true if added, otherwise false
      */
      agx::Bool add( agxModel::NodeAttachmentRef attachment );

      /**
      Finds attachment given index \p index.
      \param index - index of attachment
      \return external attachment given \p index if present, if index is out of bounds null
      */
      agxModel::NodeAttachment* getAttachment( agx::UInt index ) const;

      /**
      Find attachment given name \p name.
      \param name - name of the attachment (naming by attachment->setName( "name" ))
      \return attachment with name \p name, null if not present
      */
      agxModel::NodeAttachment* getAttachment( const agx::String& name ) const;

      /**
      \return the number of attachments this node has
      */
      agx::UInt getNumAttachments() const;

      /**
      \param i - index of child
      \return child number \p i (null if out of bound)
      */
      agxModel::Deformable1DNode* getChild( agx::UInt i = 0 ) const;

      /**
      \return the number of children to this node
      */
      agx::UInt getNumChildren() const;

      /**
      \return true if number of children > 0, otherwise false
      */
      agx::Bool hasChildren() const;

      /**
      \return parent node of this node
      */
      agxModel::Deformable1DNode* getParent() const;

      /**
      \note If this is the end node this segment will only have an end position, not an end node.
      \return the segment between this node and the next
      */
      agxModel::Segment* getSegment();

      /**
      \note If this is the end node this segment will only have an end position, not an end node.
      \return the segment between this node and the next
      */
      const agxModel::Segment* getSegment() const;

      /**
      Updates deformation data (current constraint forces and relative rotations) given current state.
      \return the updated deformation data
      */
      const agxModel::Deformable1DNode::DeformationData& updateDeformationData();

      /**
      \return the deformation data from last call to updateDeformationData
      */
      const agxModel::Deformable1DNode::DeformationData& getDeformationData() const;

      /**
      Calculates the moment of area given a geometry factory containing values such as width and height.
      If \p geometryFactory is null width and height will be estimated as one (1).
      \param geometryFactory - geometry factory holding values such as width and height
      \return moment of area about all three rotational degrees of freedom
      */
      agx::Vec3 calculateAreaMomentOfInertia( const agxModel::Deformable1DGeometryFactory* geometryFactory ) const;

    public:
      /**
      Internal method that calculates/estimate the length of \p geometry.
      \param geometry - geometry to estimate length
      */
      virtual agx::Real getLength( const agxCollide::Geometry* geometry ) const override;

      /**
      Internal method that splits a geometry in two.
      */
      virtual agxCollide::Shape* splitShape( agxCollide::Shape* shapeInBranch, agx::Real shapePosition, agxModel::Tree::Branch* newBranch, agx::Real& newBelowLength, agx::Real& newAboveLength ) override;

      /**
      Internal method that creates the constraint between this node and its parent.
      */
      virtual agx::ConstraintRef createConstraintWithParent() const override;

      /**
      \return clone of this node
      */
      virtual agxModel::Tree::Branch* clone() override;

      /**
      Remove notification from an assembly (typically the agxModel::Deformable1D object) and the simulation.
      \note Any rigid body, constraint, geometry, etc has to be removed from both the assembly and the simulation.
      */
      virtual void removeNotification( agxSDK::Assembly* assembly, agxSDK::Simulation* simulation );


      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::Deformable1DNode);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;


    protected:
      /**
      A node has to have a transform, hidden default constructor.
      */
      Deformable1DNode();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DNode();

      friend class Deformable1DRoute;
      /**
      Assign the length of this node.
      \param length - new length of this node
      */
      void setLength( agx::Real length );

      /**
      \param[out] translational - translational force applied by the constraint
      \param[out] rotational - rotational force applied by the constraint
      \return true if the values have been set
      */
      agx::Bool getCurrentConstraintForce( agx::Vec3& translational, agx::Vec3& rotational ) const;

    protected:
      typedef agx::Vector< NodeAttachmentRef > AttachmentContainer;

    protected:
      agx::Real m_length;
      AttachmentContainer m_attachments;
      DeformationData m_deformationData;
      Segment m_segment;
  };
}

#endif // AGXMODEL_DEFORMABLE1DNODE_H
