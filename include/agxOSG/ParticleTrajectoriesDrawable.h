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

#ifndef AGXOSG_PARTICLE_TRAJECTORIES_DRAWABLE_H
#define AGXOSG_PARTICLE_TRAJECTORIES_DRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agx/ParticleSystem.h>


namespace agxOSG
{
  class AGXOSG_EXPORT ParticleTrajectoriesDrawable : public osg::Drawable
  {
    //typedef agx::Vec3Vector PositionBuffer;
    //typedef agx::Vec4fVector ColorsBuffer;
    typedef std::deque<agx::Vec3> PositionBuffer;
    typedef std::deque<agx::Vec4f> ColorsBuffer;
    typedef std::pair<PositionBuffer, ColorsBuffer> ParticleTrajectoryData;

    typedef agx::HashTable<agx::Index, ParticleTrajectoryData> TrajectoryTable;

    class ParticleStorageListener : public agxData::EntityStorage::EventListener
    {

    public:
      ParticleStorageListener(ParticleTrajectoriesDrawable* psdraw);

      virtual void destroyCallback(agxData::EntityStorage* storage) override;
      virtual void createInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
      virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange range) override;
      virtual void destroyInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
      virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
      virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
      virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange instance) override;
      virtual void permuteCallback(agxData::EntityStorage* storage, agxData::Array< agx::Index > permutation) override;

      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    protected:
      ParticleTrajectoriesDrawable * m_psDrawable;
    };

  public:
    ParticleTrajectoriesDrawable(agx::ParticleSystem *particleSystem);
    // ParticleSystemDrawable( const ParticleSystemDrawable&c, const osg::CopyOp& copyOp );

    virtual osg::Object* cloneType() const { return new ParticleTrajectoriesDrawable(m_particleSystem.get()); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new ParticleTrajectoriesDrawable(*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const ParticleTrajectoriesDrawable *>(obj) != nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "ParticleTrajectoriesDrawable"; }


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
    void updateParticleTrajectories();

    /// Set the number of positions used in the trajectories
    void setNumTrajectoryPositions(agx::UInt num);

    /// Get the number of positions used in the trajectories
    agx::UInt getNumTrajectoryPositions() const;

    /// Reset trajectories
    void resetTrajectories();

    /// Pops latest positions in all trajectories. Useful when stepping backwards in a journal.
    void popLatestPositions();

    /// Remove the current stored trajectory from the drawable for a particle with given id
    void removeParticleTrajectoryFromId(agx::Index particleId);

    void createParticleTrajectoryFromId( agx::Index particleId );

    bool particleIsRendered( agx::Physics::ParticlePtr& ptr );

    void setEnable(bool enable);

    bool getEnable() const;

    // Amount of particles that should sampled in the tracing. 0 None ; 1 All
    void setSampling(agx::Real sampling, bool doResample=true);

    agx::Real getSampling() const;

    agx::ParticleSystem * getParticleSystem();

    void resample();

  protected:
    virtual ~ParticleTrajectoriesDrawable();

    void updateKernelBuffers();

    void handleCreationEvent(agx::Index index);

    void handleDestructionEvent(agx::Index index);

    void sample(agx::Index index);

    void popOldPositionsInDeadTrajectories();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    mutable agx::ParticleSystemObserver m_particleSystem;
    agx::TaskGroupRef m_renderTask;

    agx::Vector<ParticleTrajectoryData> m_destroyedParticlesTrajectories;
    TrajectoryTable                     m_activeParticleTrajectories;

    agxData::BufferRef m_verticesBuffer;
    agxData::BufferRef m_colorBuffer;
    agxData::BufferRef m_countBuffer;
    agxData::BufferRef m_firstsBuffer;
    agx::UInt m_maxPosCounts;
    bool m_enable;
    agx::Real m_sampling;

    ParticleStorageListener m_particleStorageListener;
  };

  AGX_FORCE_INLINE agx::ParticleSystem * ParticleTrajectoriesDrawable::getParticleSystem() { return m_particleSystem; }

  AGX_FORCE_INLINE bool ParticleTrajectoriesDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE agx::Real ParticleTrajectoriesDrawable::getSampling() const { return m_sampling; }

  AGX_FORCE_INLINE agx::UInt ParticleTrajectoriesDrawable::getNumTrajectoryPositions() const { return m_maxPosCounts; }
}

#endif /* AGXOSG_PARTICLE_SYSTEMDRAWABLE_H */
