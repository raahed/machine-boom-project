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

#ifndef AGXOSG_TREE_H
#define AGXOSG_TREE_H

#include <agxOSG/SplineRendererOld.h>

#include <agxModel/Tree.h>

namespace agxOSG
{
  class AGXOSG_EXPORT TreeRenderer : public agxSDK::StepEventListener
  {
    public:
      TreeRenderer( agxModel::Tree* tree, osg::Group* root );

      /**
      Assign color to this tree. Default color is white.
      \param color - new color
      */
      void setColor( const agx::Vec4& color );

      /**
      \return color of the tree
      */
      agx::Vec4 getColor() const;

    protected:
      virtual ~TreeRenderer() {}

      virtual void addNotification();
      virtual void post( const agx::TimeStamp& );

    protected:
      osg::ref_ptr< osg::Group > m_node;
      agxModel::TreeRef          m_tree;
      CylinderGeometryShaderRef  m_cylinderGeometry;
  };

  typedef agx::ref_ptr< TreeRenderer > TreeRendererRef;

  class AGXOSG_EXPORT Tree : public agxModel::Tree
  {
    public:
      Tree( osg::Group* root );

      /**
      Assign color to this tree. Default color is white.
      \param color - new color
      */
      void setColor( const agx::Vec4& color ) { m_renderer->setColor( color ); }

      /**
      \return color of the tree
      */
      const agx::Vec4 getColor() const { return m_renderer->getColor(); }

    protected:
      virtual ~Tree() {}

      /**
      Use agxModel::Tree instead if you intend to use this constructor.
      */
      Tree() : m_renderer( nullptr ), m_graphicsRoot( nullptr ) {}

      virtual Tree* clone();

      virtual void addNotification( agxSDK::Simulation* simulation );
      using agxModel::Tree::addNotification;

    protected:
      TreeRendererRef m_renderer;
      osg::observer_ptr< osg::Group > m_graphicsRoot;
  };

  typedef agx::ref_ptr< Tree > TreeRef;
}

#endif
