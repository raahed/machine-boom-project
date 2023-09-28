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

#ifndef AGXOSG_RIGIDBODY_TRAJECTORIES_DRAWABLE_H
#define AGXOSG_RIGIDBODY_TRAJECTORIES_DRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agxOSG/ScalarColorMap.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agxSDK/Simulation.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(RigidBodyTrajectoryDrawable);
  class AGXOSG_EXPORT RigidBodyTrajectoryDrawable : public osg::Drawable
  {
    typedef std::deque<agx::Vec3> PositionBuffer;
    typedef std::deque<agx::Vec4f> ColorsBuffer;
    typedef std::pair<PositionBuffer, ColorsBuffer> TrajectoryData;
    typedef agx::HashTable<agx::Index, TrajectoryData> TrajectoryTable;

  public:
    RigidBodyTrajectoryDrawable(agxSDK::Simulation *simulation);

    virtual osg::Object* cloneType() const { return new RigidBodyTrajectoryDrawable(m_simulation); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new RigidBodyTrajectoryDrawable(*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const RigidBodyTrajectoryDrawable *>(obj) != nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "RigidBodyTrajectoryDrawable"; }

    /** Compute the bounding box around Drawables's geometry.*/
# if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
    virtual osg::BoundingSphere computeBound() const;
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    /// The actual draw method for the particle system
    virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

    /// Update the trajectory structures
    void updateTrajectories();

    /// Set the number of positions used in the trajectories
    void setNumTrajectoryPositions(agx::UInt num);

    /// Get the number of positions used in the trajectories
    agx::UInt getNumTrajectoryPositions() const;

    /// Set default color of trajectories
    void setDefaultTrajectoryColor(const agx::Vec4f& color);

    /// Reset trajectories
    void resetTrajectories();

    /// Remove the current stored trajectory from the drawable for a particle with given id
    void removeTrajectoryFromId(agx::Index particleId);

    void setEnable( bool enable, bool doResample = true );

    bool getEnable() const;

    void setOnlyDrawRigidBodyEmitterBodies( bool enable );

    bool getOnlyDrawRigidBodyEmitterBodies() const;

    // Amount of particles that should sampled in the tracing. 0 None ; 1 All
    void setSampling(agx::Real sampling, bool doResample = true);

    agx::Real getSampling() const;

    agxSDK::Simulation * getSimulation();

    void resample();

    void addRigidBodyCallback(agxSDK::Simulation *, agx::RigidBody *body);

    void removeRigidBodyCallback(agxSDK::Simulation *, agx::RigidBody *body);

    /**
    Set the scalar color map that will be used color the rigid body trajectories with
    respect to speed.
    \param colorMap - the scalar map that will be used to color the body trajectories
    \note - setting a color map overrides the current set trajectory color via
            `setDefaultTrajectoryColor`.
    */
    void setColorMap(ScalarColorMap* colorMap);

    /**
    \return the scalar color map that is used to color the rigid body trajectories with
            respect to speed.
    */
    ScalarColorMap* getColorMap() const;

  protected:
    virtual ~RigidBodyTrajectoryDrawable();

    void updateKernelBuffers();

    void handleCreationEvent(agx::Index index);

    void handleDestructionEvent(agx::Index index);

    void sample(agx::Index index);

    void popOldPositionsInDeadTrajectories();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:

    agxSDK::Simulation::RigidBodyEvent::CallbackType m_addRigidBodyCallback;
    agxSDK::Simulation::RigidBodyEvent::CallbackType m_removeRigidBodyCallback;

    mutable agxSDK::SimulationObserver  m_simulation;
    agx::TaskGroupRef                   m_renderTask;

    agx::Vector<TrajectoryData>  m_destroyedTrajectories;
    TrajectoryTable              m_activeTrajectories;
    agx::Vec4f                   m_trajectoryColor;

    agxData::BufferRef m_verticesBuffer;
    agxData::BufferRef m_colorBuffer;
    agxData::BufferRef m_countBuffer;
    agxData::BufferRef m_firstsBuffer;
    agx::UInt m_maxPosCounts;
    bool      m_enable;
    bool      m_onlyDrawRigidBodyEmitterBodies;
    agx::Real m_sampling;
    ScalarColorMapRef m_colorMap;
  };

  AGX_FORCE_INLINE agxSDK::Simulation * RigidBodyTrajectoryDrawable::getSimulation() { return m_simulation; }

  AGX_FORCE_INLINE bool RigidBodyTrajectoryDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE agx::Real RigidBodyTrajectoryDrawable::getSampling() const { return m_sampling; }

  AGX_FORCE_INLINE agx::UInt RigidBodyTrajectoryDrawable::getNumTrajectoryPositions() const { return m_maxPosCounts; }

  AGX_FORCE_INLINE void RigidBodyTrajectoryDrawable::setDefaultTrajectoryColor(const agx::Vec4f& color) { m_trajectoryColor = color; }
}

#endif /* AGXOSG_PARTICLE_SYSTEMDRAWABLE_H */
