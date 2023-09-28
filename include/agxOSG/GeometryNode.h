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

#include <agxOSG/export.h>

#include <agxCollide/Geometry.h>
#include <agxCollide/Space.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/MatrixTransform>
#include <osg/observer_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{

  class Node;

  /**
  A node that can be associated with a collision geometry so that the transformation of this
  node is updated from the transformation of the Geometry.

  If setAutoRemove() is set to true, this node will be removed from the scenegraph
  whenever the associated Geometry is deallocated.
  */
  class AGXOSG_EXPORT GeometryNode : public osg::MatrixTransform
  {
  public:

    enum RenderFlag
    {
      RENDER_DATA=0x1, /**< Specify the RenderData data of a GeometryNode */
      GEOMETRY_DATA=0x2 /**< Specify the renderable data generated directly from the AGX shapes */
    };

    /// Default constructor
    GeometryNode();

    /**
    Constructor that associates this node's transformation with the transformation of the specified
    geometry.
    */
    GeometryNode( agxCollide::Geometry *geometry );

    /**
    Associate this node's transformation with the transformation of the specified
    geometry.
    */
    void setGeometry( agxCollide::Geometry* geometry );

    /**
    Get the associated geometry.
    */
    agxCollide::Geometry *getGeometry();

    /**
    \return true if this node should be removed, when the destructor of the associated
    geometry is called.
    */
    bool getAutoRemove() const { return m_autoRemove; }

    /**
    Specify whether this node should be removed when the destructor of the associated
    geometry is called.
    */
    void setAutoRemove( bool flag ) { m_autoRemove = flag; }

    virtual osg::Object* cloneType() const { return new GeometryNode (); }
    virtual osg::Object* clone(const osg::CopyOp& copyop) const { return new GeometryNode (*this,copyop); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const GeometryNode *>(obj)!=nullptr; }
    virtual const char* className() const { return "MatrixTransform"; }
    virtual const char* libraryName() const { return "osg"; }

    virtual void traverse(osg::NodeVisitor& nv);

    /*! Copy constructor using CopyOp to manage deep vs shallow copy.*/
    GeometryNode( const GeometryNode& node, const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY );

    void removeNode(agxCollide::Space* space, agxCollide::Geometry *geometry);
    void removeFromParents();

    void selectRenderChild(RenderFlag flag);
    RenderFlag getSelectedRenderChild() const;

    /**
    Set a child to this node so it can be toggled with the setRenderFlag
    \param flag - Specifies which kind of render data this is: RENDER_DATA (as stored in Geometry as RenderData)
    or RENDER_GEOMETRY (parsed from the AGX Shape).
    \param node - The node to add
    */
    void setRenderChild(RenderFlag flag, osg::Node *node);


    /**
    \param flag - Specifies which child that should be returned-
    \return a pointer to the specified child. Null if the specified child does not exist!
    */
    osg::Node *getRenderChild(RenderFlag flag);

    /**
    \param flag - Specifies which child that should be returned-
    \return a pointer to the specified child. Null if the specified child does not exist!
    */
    const osg::Node *getRenderChild(RenderFlag flag) const;

  public:
    agxCollide::Geometry::ShapeChangeEvent::CallbackType shapeCallback;

  protected:
    // agxCollide::Space::Event::CallbackType m_removeGeometryCallback;
    bool m_autoRemove;

    agx::observer_ptr<agxCollide::Geometry> m_geometry;

    virtual ~GeometryNode();

    osg::ref_ptr<osg::Group> m_geometryParent;
    osg::ref_ptr<osg::Group> m_renderParent;
  };


} // Namespace agxOSG

