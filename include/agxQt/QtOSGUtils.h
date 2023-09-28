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

#ifndef AGXQT_QTOSGUTILS_H
#define AGXQT_QTOSGUTILS_H

#include <agx/config/AGX_USE_KEYSHOT.h>
#include <agxQt/export.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <agxQt/ui_RenderController.h>
#include <QWidget>
#include <osgViewer/ViewerEventHandlers>
#include <osg/Group>
#include <osgViewer/Renderer>
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/ParticlePostProcessColorer.h>
#include <agxQt/GranularImpactsDrawable.h>
#include <agxQt/ParticleFilters.h>
#include <agxQt/CameraViewSet.h>
#include <agx/Referenced.h>
#include <agxOSG/ParticleSystemDrawable.h>
#include <agxOSG/ParticleTrajectoriesDrawable.h>
#include <agxOSG/RigidBodyTrajectoryDrawable.h>
#include <agxOSG/ParticleContactGraphDrawable.h>
#include <agxOSG/ClipPlane.h>
#include <agx/Physics/GranularBodySystem.h>
#include <agxSDK/StepEventListener.h>

namespace agxOSG
{
  class SceneDecorator;
  class Node;
  class Group;
}

namespace agxQt
{
  class QtAgxWrapper;

  AGX_DECLARE_POINTER_TYPES(ClipPlaneController);
  /**
 * Class that is used to manipulate the clip plane in the scene.
 */
  class ClipPlaneController : public agx::Referenced
  {
  public:
    const agx::Vec3 BASE_NORMAL = agx::Vec3(0, 1, 0);

  public:
    /// Constructor
    ClipPlaneController(agxOSG::ClipPlane* clipPlane);

    void setPosition( agx::Vec3 position );

    void setRotation(const agx::Quat& quat );

    agx::Vec3 getPosition() const;

    agx::Quat getRotation() const;

    bool getEnable() const;

    void setEnable(bool enable);

    bool getEnableRendering() const;

    void setEnableRendering(bool enable);

    agx::Real getIncrement() const;

    void setIncrement(agx::Real increment);

    /// Increments the Near plane of the scene in the coupled agxViewer
    void incrementClipPlane();

    /// Decrements the Near plane of the scene in the coupled agxViewer
    void decrementClipPlane();

  protected:
    void updateClipPlaneFromIncrement( agx::Real increment );

    /// Destructor
    virtual ~ClipPlaneController();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxOSG::ClipPlane*  m_clipPlane;
    agx::Quat           m_rotation;
    agx::Real           m_increment;
    bool                m_enableRendering;
  };

  AGX_DECLARE_POINTER_TYPES(TrajectoryRenderer);

  //////////////////////////////////////////////////////////////////////////
  // ParticleTrajectoryRenderer
  //////////////////////////////////////////////////////////////////////////
  class TrajectoryRenderer : public agx::Referenced
  {
    typedef agx::Vector < agx::Vec3 > Vec3List;

  public:
    TrajectoryRenderer( agxSDK::Simulation* simulation,
                        osg::Group* root,
                        agx::ParticleSystem* ps );

    void setMaxPositionPoints( agx::UInt posPoints );

    agx::UInt getMaxPositionPoints();

    void setEnable( bool enable );

    bool getEnable() const;

    void setSampling( agx::Real enable );

    agx::Real getSampling() const;

    void reSample();

    void resetTrajectoires();

    void setColorMap(agxOSG::ScalarColorMap* colorMap);

    agxOSG::ScalarColorMap* getColorMap();

  protected:
    virtual ~TrajectoryRenderer();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
    agx::ParticleSystem*               m_particleSystem;
    osg::observer_ptr<osg::Group>      m_root;
    osg::ref_ptr<osg::Geode>           m_trajectoryGeode;
    agx::UInt                          m_numTrajectoryPositions;
    agx::Real                          m_sampling;
    bool                               m_enable;

    osg::ref_ptr < agxOSG::ParticleTrajectoriesDrawable > m_particleTrajectoryDrawable;
    osg::ref_ptr < agxOSG::RigidBodyTrajectoryDrawable >  m_RBTrajectorydrawable;
  };

  AGX_DECLARE_POINTER_TYPES( ParticleContactGraphRenderer );


  //////////////////////////////////////////////////////////////////////////
  // ParticleContactGraphRenderer
  //////////////////////////////////////////////////////////////////////////
  class ParticleContactGraphRenderer : public agx::Referenced
  {
  public:
    ParticleContactGraphRenderer(osg::Group * root, agx::ParticleSystem * ps);

    void setEnable(bool enable);

    bool getEnable() const;

    agx::Real getMinNetworkForce() const;

    void setMinNetworkForce(agx::Real minForce);

    agx::Real getMaxNetworkForce() const;

    void setMaxNetworkForce(agx::Real minForce);

  protected:
    virtual ~ParticleContactGraphRenderer();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
    osg::observer_ptr<osg::Group>         m_root;
    agx::ParticleSystem*                  m_particleSystem;
    agxOSG::ParticleContactGraphDrawable* m_graphDrawable;
  };

  AGX_FORCE_INLINE void ParticleContactGraphRenderer::setEnable(bool enable) { m_graphDrawable->setEnable(enable); }

  AGX_FORCE_INLINE bool ParticleContactGraphRenderer::getEnable() const { return m_graphDrawable->getEnable(); }

  AGX_FORCE_INLINE void ParticleContactGraphRenderer::setMaxNetworkForce(agx::Real maxForce) { m_graphDrawable->setMaxForce(maxForce); }

  AGX_FORCE_INLINE void ParticleContactGraphRenderer::setMinNetworkForce(agx::Real minForce) { m_graphDrawable->setMinForce(minForce); }

  AGX_FORCE_INLINE agx::Real ParticleContactGraphRenderer::getMinNetworkForce() const { return m_graphDrawable->getMinForce(); }

  AGX_FORCE_INLINE agx::Real ParticleContactGraphRenderer::getMaxNetworkForce() const { return m_graphDrawable->getMaxForce(); }

  /**
  * This utility class is used for post processing of the particle system drawable representation, such as alpha rendering and velocity color.
  */
  class ParticleSystemRenderUtility : public agxSDK::StepEventListener
  {
  public:
    ParticleSystemRenderUtility(
      agxQt::QtAgxWrapper* qtagxViewer,
      osg::Group* root,
      agxOSG::ParticleSystemDrawable* psdraw,
      agx::ParticleSystem* system);

    void setParticleShaderMode(agxOSG::ParticleSystemDrawable::ParticleShaderMode mode);

    agxOSG::ParticleSystemDrawable::ParticleShaderMode getParticleShaderMode() const;

    void setAlphaValue(agx::Real alpha);

    agx::Real getAlphaValue() const;

    virtual void pre(const agx::TimeStamp& t);

    virtual void post(const agx::TimeStamp& t);

    void setParticleColorModeFromName(const agx::String& mode);

    ParticleColorMode getActiveParticleColorMode() const;

    ParticleColorModeVector getAvailableColorModes() const;

    ParticleRenderFilterManager* getParticleRenderFilterManager();

    void setMaxColorScalar(agx::Real maxVel);

    void setMinColorScalar(agx::Real minVel);

    agx::Real getMaxColorScalar() const;

    agx::Real getMinColorScalar() const;

    void setAnalysisBox(const agx::AnalysisBox* box);

    void setLimitColoringToAnalysisBound(bool enable);

    bool getLimitColoringToAnalysisBound() const;

    void setShowColorLegend(bool enable, int width, int height);

    void setShowColorLegend(bool enable);

    void updateColorLegend(int width, int height);

    void refreshColorLegend();

    bool getShowColorLegend() const;

    void setEnableParticleRendering(bool enable);

    bool getEnableParticleRendering() const;

    void preRenderParticles( agx::ParticleSystem * system );

    void preRenderBodies();

    ParticleContactGraphRendererRef getParticleContactGraphRenderer();

    GranularImpactsDrawable* getGranularImpactsDrawable();

    void addCustomColorers();

    bool hasOpenColorLegend() const;

    void resetParticleFilteredState();

    void setEnableRenderOnNSSBodies(const agx::RigidBodyRefVector& bodies, bool enable);

    void setEnableRenderingOnTemplateBodies(const agx::RigidBodyRefVector& bodies, bool enable);

    agxOSG::ParticleSystemDrawable* getParticleSystemDrawable();

  protected:

    virtual ~ParticleSystemRenderUtility();

    void preRender();

    void resetAlphaOnColors();

    void setAlphaOnParticles(agx::Real alpha);

    void removeColorLegend();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxQt::QtAgxWrapper*                    m_qtAgxWrapper;
    osg::ref_ptr<agxOSG::ParticleSystemDrawable> m_psDrawable;
    agx::ParticleSystem*                    m_system;
    agx::Real                               m_alpha;

    // color renderer
    osg::ref_ptr<osg::Group>                 m_hudGroup;
    osg::observer_ptr<osg::Group>            m_root;

    ParticleContactGraphRendererRef  m_particleContactGraphRenderer;
    ParticleColorerManagerRef        m_particleColorerManager;
    ParticleRenderFilterManagerRef   m_particleRenderFilterManager;

    // Impacts drawable
    osg::ref_ptr<agxQt::GranularImpactsDrawable> m_impactsDrawable;
    bool                                         m_enableParticleRendering;
  };

  AGX_DECLARE_POINTER_TYPES(ParticleSystemRenderUtility);

  /* Implementation */

  AGX_FORCE_INLINE agx::Real ParticleSystemRenderUtility::getAlphaValue() const { return m_alpha; }

  AGX_FORCE_INLINE ParticleColorMode ParticleSystemRenderUtility::getActiveParticleColorMode() const { return m_particleColorerManager->getActiveParticleColorMode(); }

  AGX_FORCE_INLINE ParticleColorModeVector ParticleSystemRenderUtility::getAvailableColorModes() const { return m_particleColorerManager->getAvailableParticleColorModes(); }

  AGX_FORCE_INLINE void ParticleSystemRenderUtility::setLimitColoringToAnalysisBound(bool enable) { m_particleColorerManager->setLimitInBound(enable); }

  AGX_FORCE_INLINE bool ParticleSystemRenderUtility::getLimitColoringToAnalysisBound() const { return m_particleColorerManager->getLimitInBound(); }

  AGX_FORCE_INLINE agx::Real ParticleSystemRenderUtility::getMinColorScalar() const
  {
    if (m_particleColorerManager)
      return m_particleColorerManager->getMinScalar();
    else
      return 0;
  }

  AGX_FORCE_INLINE agx::Real ParticleSystemRenderUtility::getMaxColorScalar() const
  {
    if (m_particleColorerManager)
      return m_particleColorerManager->getMaxScalar();
    else
      return 0;
  }

  AGX_FORCE_INLINE bool ParticleSystemRenderUtility::getEnableParticleRendering() const { return m_enableParticleRendering; }

  AGX_FORCE_INLINE ParticleContactGraphRendererRef ParticleSystemRenderUtility::getParticleContactGraphRenderer() { return m_particleContactGraphRenderer; }

  AGX_FORCE_INLINE GranularImpactsDrawable* ParticleSystemRenderUtility::getGranularImpactsDrawable() { return m_impactsDrawable; }

  osg::Geode * createCube(const agx::Vec3& position, const agx::Vec3& boxhalfVec3);
  osg::Geode * createCubeLines(const agx::Vec3& position, const agx::Vec3& boxhalfVec3);
  osg::Geode * createPlaneLines(const agx::Vec3& position, const agx::Vec3& normal, agx::Real size);

#if AGX_USE_KEYSHOT()
  /**
  * Dumps particle information to a keyshot file for rendering later
  */
  class KeyShotParticleDumper
  {
    KeyShotParticleDumper();
  public:

    static void dumpParticlesInSimulationToKeyShotFile(agxSDK::Simulation * simulation, agx::String filename);

    static void exportCurrentJournalFrameToBip(int frame,
      const agx::String& journalFilename,
      const agx::String& sessionName,
      const agx::String& bipFile,
      agxQt::CameraDescription* camera,
      agxQt::ParticleSystemRenderUtility* renderUtil);

  protected:
    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  };
#endif
}

#endif
