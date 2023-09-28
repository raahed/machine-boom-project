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

#ifndef AGXOSG_GRANULARMERGEDBODYDRAWABLE_H
#define AGXOSG_GRANULARMERGEDBODYDRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agxOSG/Node.h>
#include <agxOSG/GeometryNode.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agxSDK/Simulation.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(GranularMergedBodyDrawable);
  class AGXOSG_EXPORT GranularMergedBodyDrawable : public osg::Drawable
  {

  public:
    GranularMergedBodyDrawable(agxSDK::Simulation *simulation, osg::Group * root = nullptr);

    virtual osg::Object* cloneType() const { return new GranularMergedBodyDrawable(m_simulation, m_root); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new GranularMergedBodyDrawable(*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const GranularMergedBodyDrawable *>(obj) != nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "GranularMergedBodyDrawable"; }


    /** Compute the bounding box around Drawables's geometry.*/
# if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
    virtual osg::BoundingSphere computeBound() const;
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    /// The actual draw method for the particle system
    virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

    void setEnable(bool enable);

    bool getEnable() const;

    agxSDK::Simulation * getSimulation();

    void updateMergedBodyDrawable();

    void addRigidBodyCallback(agxSDK::Simulation *, agx::RigidBody *body);

    void removeRigidBodyCallback(agxSDK::Simulation *, agx::RigidBody *body);

  protected:
    virtual ~GranularMergedBodyDrawable();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:

    agxSDK::Simulation::RigidBodyEvent::CallbackType m_addRigidBodyCallback;
    agxSDK::Simulation::RigidBodyEvent::CallbackType m_removeRigidBodyCallback;

    mutable agxSDK::SimulationObserver  m_simulation;
    osg::Group * m_root;
    agx::TaskGroupRef m_renderTask;

    bool m_enable;

    agx::HashTable<agx::RigidBody*, osg::Group *> m_createdBodies;
  };

  AGX_FORCE_INLINE agxSDK::Simulation * GranularMergedBodyDrawable::getSimulation() { return m_simulation; }

  AGX_FORCE_INLINE bool GranularMergedBodyDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE void GranularMergedBodyDrawable::setEnable(bool enable) { m_enable = enable; }

}

#endif /* AGXOSG_PARTICLE_SYSTEMDRAWABLE_H */
