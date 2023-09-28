/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/
#pragma once


#include <agxOSG/export.h>
#include <agxSDK/GuiEventListener.h>
#include <agxCollide/Trimesh.h>
#include <agxOSG/GeometryNode.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/observer_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.


namespace osg
{
  class Node;
  class Geode;
  class Group;
}

namespace agxOSG
{

  /**
  Updates a renderable representation of a deformable triangle mesh
  */
  class AGXOSG_EXPORT DeformableMeshUpdater : public agxSDK::GuiEventListener
  {
  public:
    /**
    Constructor.
    During update, the renderable mesh might be reconstructed from scratch if the number of vertices in the physical mesh changes.

    \param trimesh - The mesh that will be deformed
    \param visualRoot - A root node that contains one agxOSG::GeometryNode with ONE triangle mesh representation
    */
    DeformableMeshUpdater(agxCollide::Trimesh* trimesh, osg::Group* visualRoot);

    /**
    Update the visual mesh from the collision mesh
    \param force - Even if the trimesh has not been changed since last update, do the update anyway
    */
    void update(bool force = false);

    /**
    Force the visual mesh to be updated every frame, even though the collision mesh has not been modified.
    \param flag - If true, the visual mesh will be updated every frame no matter what.
    */
    void setForceUpdate(bool flag);

    bool getForceUpdate() const;

  protected:
    virtual ~DeformableMeshUpdater();

  private:

    bool prepareGeometry(osg::Node* node);
    bool prepareGeometry(osg::Group* group);
    bool prepareGeometry(osg::Geode* geode);

    void addNotification() override;

    bool  prepareGeometry();
    void update(float x, float y) override;

  private:

    agxCollide::TrimeshObserver m_trimesh;
    osg::observer_ptr<osg::Group> m_visualRoot;
    agx::UInt32 m_lastModificationCount;
    bool m_forceUpdate;


    osg::observer_ptr<agxOSG::GeometryNode> m_geometryNode;

    osg::ref_ptr< osg::Drawable > m_drawable;
    osg::ref_ptr< osg::Geometry > m_osgGeometry;
    osg::ref_ptr< osg::Vec3Array > m_osgVertices;
    osg::ref_ptr< osg::Vec3Array >  m_osgNormals;
    osg::ref_ptr< osg::Vec2Array > m_osgTexCoords;
  };
}
