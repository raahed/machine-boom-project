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

#ifndef AGXOSG_FRAME_TRAJECTORY_DRAWABLE_H
#define AGXOSG_FRAME_TRAJECTORY_DRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agxSDK/Simulation.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(FrameTrajectoryDrawable);
  class AGXOSG_EXPORT FrameTrajectoryDrawable : public osg::Drawable
  {
    typedef std::deque<agx::Vec3> PositionBuffer;
    typedef std::deque<agx::Vec4f> ColorsBuffer;
    typedef std::pair<PositionBuffer, ColorsBuffer> TrajectoryData;

    typedef agx::HashTable<agx::Index, TrajectoryData> TrajectoryTable;

  public:
    FrameTrajectoryDrawable(agxSDK::Simulation *simulation);

    virtual osg::Object* cloneType() const { return new FrameTrajectoryDrawable(m_simulation); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new FrameTrajectoryDrawable(*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const FrameTrajectoryDrawable *>(obj) != nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "FrameTrajectoryDrawable"; }


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

    /// Add a trajectory to be drawn given a frame
    bool addTrajectory(agx::Frame* frame);

    /// Add a trajectory to be drawn given a frame and a color
    bool addTrajectory(agx::Frame* frame, agx::Vec4f color);

    /// Remove the current stored trajectory from the drawable for a particle with given id
    bool removeTrajectory(agx::Frame* frame);

    /// Enable the drawable
    void setEnable(bool enable);

    /// Return true of the drawable is enabled
    bool getEnable() const;

    /// Get the simulation related to the drawable
    agxSDK::Simulation * getSimulation();

  protected:
    virtual ~FrameTrajectoryDrawable();

    void updateKernelBuffers();

    void popOldPositionsInDeadTrajectories();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:

    mutable agxSDK::SimulationObserver  m_simulation;
    agx::TaskGroupRef m_renderTask;

    agx::Vector<TrajectoryData>         m_destroyedTrajectories;
    agx::Vector<std::tuple<agx::FrameRef, agx::Vec4f, TrajectoryData>> m_activeTrajectories;
    agx::Vec4f                          m_defaultTrajectoryColor;

    agxData::BufferRef m_verticesBuffer;
    agxData::BufferRef m_colorBuffer;
    agxData::BufferRef m_countBuffer;
    agxData::BufferRef m_firstsBuffer;
    agx::UInt m_maxPosCounts;
    bool m_enable;

  };

  AGX_FORCE_INLINE agxSDK::Simulation * FrameTrajectoryDrawable::getSimulation() { return m_simulation; }

  AGX_FORCE_INLINE bool FrameTrajectoryDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE void FrameTrajectoryDrawable::setEnable(bool enable) { m_enable = enable; }

  AGX_FORCE_INLINE agx::UInt FrameTrajectoryDrawable::getNumTrajectoryPositions() const { return m_maxPosCounts; }

  AGX_FORCE_INLINE void FrameTrajectoryDrawable::setDefaultTrajectoryColor(const agx::Vec4f& color) { m_defaultTrajectoryColor = color; }
}

#endif /* AGXOSG_PARTICLE_SYSTEMDRAWABLE_H */
