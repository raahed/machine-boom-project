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

#ifndef AGXOSG_PARTICLE_SYSTEMDRAWABLE_H
#define AGXOSG_PARTICLE_SYSTEMDRAWABLE_H

#include <agx/config/AGX_USE_OPENGL_INSTANCING.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agx/ParticleSystem.h>

#ifndef DEFAULT_PARTICLE_RENDERING_MODE
  #if defined(__APPLE__)
    // The shader for quad sprites does not compile on macOS, since OpenGL 2.1 is used.
    // It should be updated to OpenGL 3.x, but that requires setting up the context a bit differently.
    #define DEFAULT_PARTICLE_RENDERING_MODE agxOSG::ParticleSystemDrawable::SPRITES
  #else
    #define DEFAULT_PARTICLE_RENDERING_MODE agxOSG::ParticleSystemDrawable::QUAD_SPRITES
  #endif
#endif

namespace agxOSG
{
  class AGXOSG_EXPORT ParticleSystemDrawable : public osg::Drawable
  {
  public:
    enum ParticleRenderingMode {
      SPRITES,
      QUAD_SPRITES,
      TRIANGLES
    #if AGX_USE_OPENGL_INSTANCING()
      , INSTANCES
    #endif
    };

    enum ParticleShaderMode
    {
      SIMPLE_SPRITES,
      ROTATION_SPRITES,
      ALPHA_SPRITES,
    };

  public:

    ParticleSystemDrawable( agx::ParticleSystem *particleSystem, ParticleRenderingMode renderMode = DEFAULT_PARTICLE_RENDERING_MODE, ParticleShaderMode particleShaderMode = ROTATION_SPRITES);
    // ParticleSystemDrawable( const ParticleSystemDrawable&c, const osg::CopyOp& copyOp );


    ParticleShaderMode getParticleShaderMode() const;
    void setParticleShaderMode(ParticleShaderMode flag);


    virtual osg::Object* cloneType() const { return new ParticleSystemDrawable ( m_particleSystem.get() ); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new ParticleSystemDrawable (*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const ParticleSystemDrawable *>(obj)!=nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "ParticleSystemDrawable"; }


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

    agx::ParticleSystem* getParticleSystem();

    /// Set the minimum allowed ParticleRenderingMode for the particles
    static void setAllowedRenderMode(ParticleRenderingMode mode);

    /// Get the minimum allowed ParticleRenderingMode for the particles
    static ParticleRenderingMode getAllowedRenderMode();

  protected:
    virtual ~ParticleSystemDrawable();

    mutable agx::ParticleSystemObserver m_particleSystem;
    agx::TaskGroupRef                   m_renderTask;
    ParticleShaderMode                  m_particleShaderMode;
    ParticleRenderingMode               m_renderMode;
    bool                                m_enable;
  private:
    static ParticleRenderingMode s_allowedRenderMode;
  };

  AGX_FORCE_INLINE bool ParticleSystemDrawable::getEnable() const { return m_enable; }

  AGX_FORCE_INLINE void ParticleSystemDrawable::setEnable(bool enable) { m_enable = enable; }
}

#endif /* AGXOSG_PARTICLE_SYSTEMDRAWABLE_H */
