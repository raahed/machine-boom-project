/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXOSG_RIGID_BODY_SYSTEM_DRAWABLE_H
#define AGXOSG_RIGID_BODY_SYSTEM_DRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agxOSG/Node.h>

#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agx/RigidBodyEmitter.h>


namespace agxOSG
{
  class AGXOSG_EXPORT RigidBodySystemDrawable : public osg::Group
  {
  public:
    typedef agx::HashTable<agxCollide::Geometry*, GeometryNode*> GeometryToNodeCache;

    RigidBodySystemDrawable( agx::RigidBodyEmitter *emitter);
    // RigidBodySystemDrawable( const RigidBodySystemDrawable&c, const osg::CopyOp& copyOp );

    virtual osg::Object* cloneType() const { return new RigidBodySystemDrawable ( m_emitter.get() ); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new RigidBodySystemDrawable (*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const RigidBodySystemDrawable *>(obj)!=nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "RigidBodySystemDrawable"; }

    void createRigidBodyInstance( agx::RigidBody* body );

    static osg::Group* copyRigidBodyVisualChildren( agx::RigidBody* newBody, osg::Group* visualTemplateGroup );

  protected:
    virtual ~RigidBodySystemDrawable();
    void instanceCreated( agx::RigidBodyEmitter* emitter, agx::RigidBody* body, agx::RigidBodyEmitter::DistributionModel* model );
    void addToBodyNodeCache( agx::RigidBody* body, osg::Group* group );
    osg::Group* getCachedBodyNode( agx::RigidBody* body );

    void initBodyCache( agx::RigidBodyEmitter* emitter );
    bool bodyCacheIsEmpty();

  private:
    mutable agx::RigidBodyEmitterObserver m_emitter;
    agx::RigidBodyEmitter::Event::CallbackType m_createInstanceCallback;
    agx::HashTable<agx::RigidBody*, osg::ref_ptr<osg::Group>> m_cachedBodyNodes;
  };

}


#endif /* AGXOSG_RIGID_BODY_SYSTEM_DRAWABLE_H */
