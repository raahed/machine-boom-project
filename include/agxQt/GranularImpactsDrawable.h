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

#ifndef AGXOSG_GRANULARIMPACTDRAWABLE_H
#define AGXOSG_GRANULARIMPACTDRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agxSDK/Simulation.h>

namespace agxQt
{
  AGX_DECLARE_POINTER_TYPES(GranularImpactsDrawable);
  class AGXQT_EXPORT GranularImpactsDrawable : public osg::Drawable
  {
  public:
    GranularImpactsDrawable(agx::ParticleSystem* particleSystem);

    virtual osg::Object* cloneType() const { return new GranularImpactsDrawable(m_particleSystem); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new GranularImpactsDrawable(*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const GranularImpactsDrawable *>(obj) != nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "GranularImpactsDrawable"; }


    /** Compute the bounding box around Drawables's geometry.*/
# if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
    virtual osg::BoundingSphere computeBound() const;
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    /// The actual draw method for the merged bodies
    virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

    void setEnable(bool enable);

    bool getEnable() const;

    void updateDrawable();

    bool hasParticleSystem() const;

    agx::Real getRenderPointRadius() const;

    void setRenderPointRadius(agx::Real radius);

    void setRenderParticleGeometryImpacts(bool enable);

    bool getRenderParticleGeometryImpacts() const;

    void setRenderParticleParticleImpacts(bool enable);

    bool getRenderParticleParticleImpacts() const;

    void setImpactRenderingEnergyColoringRange(const agx::RangeReal& range);

    agx::RangeReal getImpactRenderingEnergyColoringRange() const;

    void setImpactRenderingCutoffEnergy(agx::Real cutoffEnergy);

    agx::Real getImpactRenderingCutoffEnergy() const;

    void setUseParticleColorForImpactColoring(bool enable);

    void setImpactRenderingStartTime(agx::Real cutoffTime);

    agx::Real getImpactRenderingStartTime() const;

  protected:
    virtual ~GranularImpactsDrawable();

    void initColors();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    mutable agxSDK::SimulationObserver  m_simulation;
    mutable agx::ParticleSystemObserver m_particleSystem;

    agx::TaskGroupRef                   m_renderTask;
    mutable bool                        m_firstFrame;
    bool                                m_enable;
    bool                                m_hasGranularLicense;

    bool                                m_renderParticleParticle;
    bool                                m_renderParticleGeometry;

    bool                                m_useParticleColorForImpactColoring;

    agx::Real                           m_contactRadius;

    osg::ref_ptr<osgSim::ColorRange>    m_colorRange;
    agx::Real                           m_maxEnergy;
    agx::Real                           m_minEnergy;
    agx::Real                           m_cutoffEnergy;
    agx::Real                           m_renderingStartTime;

    agxData::BufferRef                  m_verticesBuffer;
    agxData::BufferRef                  m_colorBuffer;
    agxData::BufferRef                  m_radiusBuffer;
  };

  AGX_FORCE_INLINE bool GranularImpactsDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE void GranularImpactsDrawable::setEnable(bool enable) { m_enable = enable; }

  AGX_FORCE_INLINE bool agxQt::GranularImpactsDrawable::hasParticleSystem() const { return m_particleSystem != nullptr; }

  AGX_FORCE_INLINE agx::Real agxQt::GranularImpactsDrawable::getRenderPointRadius() const { return m_contactRadius; }

  AGX_FORCE_INLINE void agxQt::GranularImpactsDrawable::setRenderPointRadius(agx::Real radius) { m_contactRadius = radius; }

  AGX_FORCE_INLINE void agxQt::GranularImpactsDrawable::setImpactRenderingStartTime(agx::Real startTime){ m_renderingStartTime = startTime; }

  AGX_FORCE_INLINE agx::Real agxQt::GranularImpactsDrawable::getImpactRenderingStartTime() const { return m_renderingStartTime; }
}

#endif /* AGXOSG_GRANULARIMPACTDRAWABLE_H */
