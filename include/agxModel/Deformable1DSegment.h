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

#ifndef AGXMODEL_DEFORMABLE1DSEGMENT_H
#define AGXMODEL_DEFORMABLE1DSEGMENT_H

#include <agxModel/export.h>

#include <agx/AffineMatrix4x4.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( Deformable1DNode );

  /**
  A Deformable1D segment is the segment between two nodes, or if this
  is the last segment, only one node but still has a start- and an
  end position defined. Every node contains a segment.

     node0     segment0      node1     segment1      node2
       o=======================o=======================o=========...
    begin0                 end0|begin1             end1|begin2

  \note Best usage of this class is to always access it through a node,
        i.e, to always write node->getSegment()->get* since this class
        holds weak reference to a node and calls to this class is not
        valid if Segment::isValid() returns false.
  */
  class AGXMODEL_EXPORT Segment
  {
    public:
      /**
      Construct given node at the begin of this segment.
      \param beginNode - node at begin of this segment
      */
      Segment( agxModel::Deformable1DNode* beginNode );

      /**
      This object may only be hold buy a node.
      */
      ~Segment();

      /**
      \return true if valid and begin node has not been deleted
      */
      agx::Bool isValid() const;

      /**
      \return true if this is an end segment (i.e., end node doesn't exist)
      */
      agx::Bool isEnd() const;

      /**
      \return begin position of this segment
      */
      agx::Vec3 getPositionBegin() const;

      /**
      \return end position of this segment
      */
      agx::Vec3 getPositionEnd() const;

      /**
      \return begin transform of this segment (getTransformBegin().getTranslate() == getPositionBegin(), getTransformBegin().getRotate() == getRotation())
      */
      agx::AffineMatrix4x4 getTransformBegin() const;

      /**
      \return end transform of this segment (getTransformEnd().getTranslate() == getPositionEnd(), getTransformEnd().getRotate() == getRotation())
      */
      agx::AffineMatrix4x4 getTransformEnd() const;

      /**
      \return transform of this segment
      */
      agx::Quat getRotation() const;

    protected:
      Segment();

    protected:
      agxModel::Deformable1DNodeObserver m_node;
  };
}

#endif // AGXMODEL_DEFORMABLE1DSEGMENT_H
