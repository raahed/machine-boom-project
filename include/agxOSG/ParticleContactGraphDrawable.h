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

#ifndef AGXOSG_PARTICLE_CONTACTGRAPH_DRAWABLE_H
#define AGXOSG_PARTICLE_CONTACTGRAPH_DRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agx/ParticleSystem.h>

namespace agxOSG
{
  class AGXOSG_EXPORT ParticleContactGraphDrawable : public osg::Drawable
  {

  public:
    ParticleContactGraphDrawable(agx::ParticleSystem *particleSystem);
    // ParticleSystemDrawable( const ParticleSystemDrawable&c, const osg::CopyOp& copyOp );

    virtual osg::Object* cloneType() const { return new ParticleContactGraphDrawable(m_particleSystem.get()); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new ParticleContactGraphDrawable(*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const ParticleContactGraphDrawable *>(obj) != nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "ParticleContactGraphDrawable"; }


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

    void updateKernelBuffers();

    void setMinForce(agx::Real minForce);

    void setMaxForce(agx::Real maxForce);

    agx::Real getMinForce() const;

    agx::Real getMaxForce() const;

    agx::ParticleSystem * getParticleSystem();

    void resample();

  protected:
    virtual ~ParticleContactGraphDrawable();

    void initColors();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    mutable agx::ParticleSystemObserver m_particleSystem;
    agx::TaskGroupRef m_renderTask;

    agxData::BufferRef m_verticesBuffer;
    agxData::BufferRef m_colorBuffer;
    agxData::BufferRef m_countBuffer;
    agxData::BufferRef m_firstsBuffer;
    int m_maxPosCounts;

    osg::ref_ptr<osgSim::ColorRange> m_colorRange;
    agx::Real m_minForce;
    agx::Real m_maxForce;
    bool m_enable;
    agx::Real m_sampling;
  };

  AGX_FORCE_INLINE agx::ParticleSystem * ParticleContactGraphDrawable::getParticleSystem() { return m_particleSystem; }

  AGX_FORCE_INLINE bool ParticleContactGraphDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE void ParticleContactGraphDrawable::setEnable(bool enable) { m_enable = enable; }

  AGX_FORCE_INLINE void ParticleContactGraphDrawable::setMinForce(agx::Real minForce) { m_minForce = minForce; initColors(); }

  AGX_FORCE_INLINE void ParticleContactGraphDrawable::setMaxForce(agx::Real maxForce) { m_maxForce = maxForce; initColors(); }

  AGX_FORCE_INLINE agx::Real ParticleContactGraphDrawable::getMinForce() const { return m_minForce; }

  AGX_FORCE_INLINE agx::Real ParticleContactGraphDrawable::getMaxForce() const { return m_maxForce; }
}


#endif /* AGXOSG_PARTICLE_SYSTEMDRAWABLE_H */
