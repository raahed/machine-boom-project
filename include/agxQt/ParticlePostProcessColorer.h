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

#ifndef AGXQT_PARTICLEPOSTPROCESSCOLORER_H
#define AGXQT_PARTICLEPOSTPROCESSCOLORER_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/ref_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <functional>
#include <agxQt/export.h>
#include <agx/Referenced.h>
#include <agx/Bound.h>
#include <agx/ParticleSystem.h>
#include <agx/AnalysisBox.h>
#include <agxOSG/ScalarColorMap.h>

namespace agx
{
  class ParticleSystem;
}

namespace agxQt
{
  // Forward declaration
  class ParticleColorer;

  /**
  * ParticleColorMode
  */
  typedef std::function<ParticleColorer*()> ColorerConstructor;
  struct AGXQT_EXPORT ParticleColorMode
  {
    agx::Name          m_id;
    ColorerConstructor m_constructor;
    agx::Name          m_unit;

    bool operator==(const ParticleColorMode& rhs) { return m_id == rhs.m_id; }
  };
  typedef agx::Vector<ParticleColorMode> ParticleColorModeVector;

  /**
  * ParticleColorer
  */
  AGX_DECLARE_POINTER_TYPES(ParticleColorer);
  class AGXQT_EXPORT ParticleColorer : public agx::Referenced
  {
  public:
    typedef std::function<agx::Real(agx::ParticleSystem*, agx::Physics::ParticlePtr)> GetParticleRealFunc;
    static const char* const COLORMODE_NONE_NAME;

  public:
    ParticleColorer(const agx::Name& name);

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) = 0;

    virtual void initColorRange(agx::Real minScalar, agx::Real maxScalar);

    agxOSG::ScalarColorMap* getColorMap() const;

    agx::Name getName() const;

  protected:
    void applyColorFromMap( agx::Vec4f& targetColorData, agx::Real scalar, bool modifyAlpha );

  protected:
    virtual ~ParticleColorer() {};

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxOSG::ScalarColorMapRef    m_colorMap;
    agx::Name                    m_name;
  };

  AGX_FORCE_INLINE agxOSG::ScalarColorMap* ParticleColorer::getColorMap() const { return m_colorMap; }

  AGX_FORCE_INLINE agx::Name ParticleColorer::getName() const { return m_name; }

  AGX_FORCE_INLINE ParticleColorMode AGXQT_EXPORT createNoneParticleColorMode()
  {
    return ParticleColorMode{ ParticleColorer::COLORMODE_NONE_NAME, []{ return nullptr; }, "" };
  }

  /**
  * Basic class for particle post processor colorer. Contains different types of coloring algorithms
  */
  AGX_DECLARE_POINTER_TYPES(ParticleColorerManager);
  class AGXQT_EXPORT ParticleColorerManager : public agx::Referenced
  {
  public:
    typedef agx::HashTable<agx::Name, ParticleColorMode> ParticleColorModeTable;
  public:
    ParticleColorerManager();

    void setMaxScalar(agx::Real maxScalar);

    void setMinScalar(agx::Real minScalar);

    agx::Real getMaxScalar() const;

    agx::Real getMinScalar() const;

    void setAnalysisBox(const agx::AnalysisBox* analysisBox);

    void setParticleColorModeFromName(const agx::String& name);

    bool particleColorModeExists(const agx::String& name);

    void addCustomColorMode(const agx::String& name, ParticleColorer::GetParticleRealFunc getRealFunc, const agx::String& unitString);

    ParticleColorMode getActiveParticleColorMode() const;

    ParticleColorModeVector getAvailableParticleColorModes();

    void colorParticles(agx::ParticleSystem * system);

    agxOSG::ScalarColorMap* getColorMap() const;

    void setEnableModifyScalarFromAlpha(bool enable);

    void setLimitInBound(bool limitInBound);

    bool getLimitInBound() const;

  protected:
    void setParticleColorMode(ParticleColorMode mode);

    void initParticleModes();

    virtual ~ParticleColorerManager() {};

    void initColorRange();

    bool shouldLimitColorToBound() const;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real                                m_maxScalar;
    agx::Real                                m_minScalar;
    const agx::AnalysisBox*                  m_analysisBox;
    bool                                     m_limitInBound;
    bool                                     m_modifyAlphaFromScalar;
    ParticleColorModeTable                   m_particleColorModeTable;
    ParticleColorerRef                       m_activeParticleColorer;
  };

  AGX_FORCE_INLINE agx::Real ParticleColorerManager::getMaxScalar() const { return m_maxScalar; }

  AGX_FORCE_INLINE agx::Real ParticleColorerManager::getMinScalar() const { return m_minScalar; }

  AGX_FORCE_INLINE void ParticleColorerManager::setMaxScalar(agx::Real maxScalar) { m_maxScalar = maxScalar; initColorRange(); }

  AGX_FORCE_INLINE void ParticleColorerManager::setMinScalar(agx::Real minScalar) { m_minScalar = minScalar; initColorRange(); }

  AGX_FORCE_INLINE void ParticleColorerManager::setAnalysisBox(const agx::AnalysisBox* analysisBox) { m_analysisBox = analysisBox; initColorRange(); }

  AGX_FORCE_INLINE void ParticleColorerManager::setLimitInBound(bool limitInBound) { m_limitInBound = limitInBound; }

  AGX_FORCE_INLINE bool ParticleColorerManager::getLimitInBound() const { return m_limitInBound; }

  AGX_FORCE_INLINE void ParticleColorerManager::setEnableModifyScalarFromAlpha(bool enable) { m_modifyAlphaFromScalar = enable; }

  /**
  * Implements the particle post processor colorer for velocities
  */
  class AGXQT_EXPORT ParticlePostPorcessColorerVelocity : public ParticleColorer
  {
  public:
    ParticlePostPorcessColorerVelocity();

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessColorerVelocity() {};
  };

  /**
  * Implements the particle post processor colorer for contact force
  */
  class AGXQT_EXPORT ParticlePostPorcessColorerContactForce : public ParticleColorer
  {
  public:
    ParticlePostPorcessColorerContactForce();

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessColorerContactForce() {};
  };

  /**
  * Implements the particle post processor colorer for contact force
  */
  class AGXQT_EXPORT ParticlePostPorcessColorerHeight : public ParticleColorer
  {
  public:
    ParticlePostPorcessColorerHeight(const agx::Name& name, const agx::Vec3& heightAxis = agx::Vec3::Z_AXIS());

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessColorerHeight(){};
    agx::Vec3 m_heightAxis;
  };

  /**
  * Implements the particle post processor colorer for contact force
  */
  class AGXQT_EXPORT ParticlePostPorcessColorerRadius : public ParticleColorer
  {
  public:
    ParticlePostPorcessColorerRadius();

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessColorerRadius(){};
  };

  /**
  * Implements the particle post processor colorer for velocities
  */
  class AGXQT_EXPORT ParticlePostPorcessColorerDisplacement : public ParticleColorer
  {
  public:
    ParticlePostPorcessColorerDisplacement();

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessColorerDisplacement() {};

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    bool m_shouldResetPostions;
    agx::HashTable<agx::Index, agx::Vec3> m_particleIdToPosition;
  };

  /**
  * Implements the particle post processor colorer for velocities
  */
  class AGXQT_EXPORT ParticlePostPorcessKineticEnergy : public ParticleColorer
  {
  public:
    ParticlePostPorcessKineticEnergy();

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessKineticEnergy() {};

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
  };

  /**
  * Implements the particle post processor colorer for velocities
  */
  class AGXQT_EXPORT ParticlePostPorcessColorerLambda : public ParticleColorer
  {
  public:
    ParticlePostPorcessColorerLambda(const agx::Name& name, ParticleColorer::GetParticleRealFunc getRealFunc);

    virtual void colorParticles(agx::ParticleSystem * system, bool limitBound, const agx::Bound3& bound, bool modifyAlpha) override;

  protected:
    virtual ~ParticlePostPorcessColorerLambda() {};

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    GetParticleRealFunc m_getRealFunc;
  };
}

#endif