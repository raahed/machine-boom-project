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
#include <agx/Math.h>
#include <agx/Referenced.h>
#include <agx/RigidBody.h>
#include <agxOSG/GeometryNode.h>


namespace agxOSG
{
  typedef agx::HashTable<agx::RigidBody*, osg::Group*> BodyToGroupNodeTable;
  typedef agx::HashTable<agx::String, osg::ref_ptr<osg::Group>> BodyToRenderGroupTable;

  AGX_DECLARE_POINTER_TYPES(RigidBodyRenderCache);
  /**
  * RigidBodyRenderCache is used to cache generated visuals and lookup mappings of
  * RigidBodies cloned from Templates in agx::RigidBodyEmitter and other related
  * granular applications. It is used to optimize rendering by basic instantiation
  * of similar template bodies and also for fast lookup to circumvent the complexity
  * of agxOSG::findGeometryNode.
  */
  class AGXOSG_EXPORT RigidBodyRenderCache : public agx::Referenced
  {
  public:
    /**
    * Default constructor.
    * \param onlyAddEmittedBodies - True if only emitted RigidBodies
                                    should be added, false otherwise.
    */
    RigidBodyRenderCache(bool onlyAddEmittedBodies=true);

    /**
    Add the osg::Group created from a RigidBody to the cache.
    \param rootNode - the rootNode that is used to build the cache.
    */
    void addToCache(osg::Group* rootNode);

    /**
    Get the current mapping in the cache between a RigidBody to a osg::Group.
    \return the current osg::Group mapped from the RigidBody.
    */
    osg::Group* getNode(agx::RigidBody* body);

    /**
    \return true if only emitted bodies should be added, false otherwise.
    */
    bool getOnlyAddEmittedBodies() const;

    /**
    \param enable - true if only emitted bodies should be added, false otherwise.
    */
    void setOnlyAddEmittedBodies( bool enable );

    /**
    * Add template body to visual cache that is used during playback in qtViewer.
    * \param body - The specified template body to add to node cache.
    * \return the visual osg::group node generated from the body.
    */
    osg::Group* addTemplateBodyToVisualCache( agx::RigidBody* body );

    /**
    * Get cached template body visual from the node cache if it exists.
    * \return the cached visual node for the template body if it exists,
              otherwise nullptr.
    */
    osg::Group* getCachedTemplateBodyVisual( agx::RigidBody* body );

    /**
    * Gets cached template body visual from the node cache if it exists,
    * otherwise a new visual node will be created in the cache and returned.
    * If the specified body is not a generated from a Template, the function
    * returns nullptr.
    * \param body - the specified body that should be used to get or create
    *               the visual representation from the node cache.
    * \return the cached visual node for the template body.
    */
    osg::Group* getOrCreateCachedTemplateBodyNode( agx::RigidBody* body );

    /**
    * Checks wherever the specifed body is generated/cloned from the temlpate
    * body via RigidBodyEmitter och creation via the Momentum API.
    * \note - the function checks if a string property called "TemplateUUID"
    *         eixsts in the body's property container.
    */
    bool bodyIsTemplateEmitted( agx::RigidBody* body );

    /**
    * Clear the internal cache.
    */
    void clear();

  protected:
    virtual ~RigidBodyRenderCache();
    void traverseAndInsert( osg::Group* rootNode );
    bool shouldAddBody(agx::RigidBody* body) const;

  private:
    BodyToGroupNodeTable   m_bodiesToGroupNodes;
    BodyToRenderGroupTable m_bodiesToRenderGroups;
    bool                   m_onlyAddEmittedBodies;
  };
}
