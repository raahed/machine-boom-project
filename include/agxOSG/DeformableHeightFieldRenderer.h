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


#ifndef AGXOSG_DEFORMABLE_HEIGHTFIELD_RENDERER_H
#define AGXOSG_DEFORMABLE_HEIGHTFIELD_RENDERER_H

#include <agxOSG/GeometryNode.h>
#include <agxSDK/StepEventListener.h>
#include <agxCollide/HeightField.h>
#include <agxModel/HeightFieldDeformer.h>
#include <agxOSG/utils.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxOSG
{
  class Group;
  class Node;
  class GeometryNode;
}

namespace osg
{
  class HeightField;
  class Geometry;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxOSG
{
  /// An osg-renderer for a deformable height field.
  class AGXOSG_EXPORT DeformableHeightFieldRenderer : public agxSDK::StepEventListener
  {
  public:

    /**
    * Constructor for DeformableHeightFieldRenderer. Will generate a GeometryNode for the height field.
    * \param heightFieldDeformer: A height field deformer.
    * \param root: A root to which the GeometryNode for the height field should be attached.
    */
    DeformableHeightFieldRenderer( agxModel::HeightFieldDeformer* heightFieldDeformer, osg::Group* root );

    /// Overloading inherited method from StepEventListener.
    virtual void last(const agx::TimeStamp& );

    /// Returns the height field node. Might be nullptr if not valid.
    agxOSG::GeometryNode* getHeightFieldNode();

    /// Force wire frame rendering of deformable height field
    bool forceWireFrameRendering();

    /// Set the height field render able as dirty to trigger rendering update
    void setAsDirty();

    /// Is renderer valid?
    bool isValid() const;

  protected:
    virtual ~DeformableHeightFieldRenderer( );

  private:
    osg::observer_ptr<osg::Group> m_root;
    agxModel::HeightFieldDeformerRef m_heightFieldDeformer;
    osg::observer_ptr<agxOSG::GeometryNode> m_heightFieldNode;
    osg::observer_ptr<osg::HeightField> m_renderHeightField;
    bool m_valid;
  };



  // Implementations

  AGX_FORCE_INLINE DeformableHeightFieldRenderer::~DeformableHeightFieldRenderer( )
  {
  }


  AGX_FORCE_INLINE agxOSG::GeometryNode* DeformableHeightFieldRenderer::getHeightFieldNode()
  {
    return m_heightFieldNode.get();
  }


  AGX_FORCE_INLINE bool DeformableHeightFieldRenderer::isValid() const
  {
    return m_valid;
  }

}

#endif
