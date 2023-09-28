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

#ifndef AGXQT_QTPARTICLERENDERWIDGET_H
#define AGXQT_QTPARTICLERENDERWIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <QProcess>
#include <agxQt/ui_ParticleRenderSettingsWidget.h>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/export.h>
#include <agxQt/QtOSGUtils.h>
#include <agx/Referenced.h>
#include <agx/Bound.h>
#include <agxQt/ParticleFilters.h>

#define DEFAULT_VIDEO_FPS 30
#define DEFAULT_CAPTURE_FPS 30

#define QUALITY_INTERVAL 20
#define DEFAULT_QUALITY_TICK 17

namespace agxQt
{
  class QtAgxWrapper;

  class AGXQT_EXPORT ParticleRenderWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit ParticleRenderWidget(QWidget *parent = 0);
    virtual ~ParticleRenderWidget();

    /// Set initial data to the inputs in the widget
    void setInitialData( agxQt::QtAgxWrapper* wrapper );

    void setActiveColorModeFromName(const agx::String& mode);

    void setActiveParticleFilterFromName(const agx::String& name);

    agx::String getSelectedParticleFilterModeName() const { return m_activeParticleFilterMode.m_id; }

  protected:
    void synchComboBox(const ParticleColorModeVector& modes);

    void synchComboBox(const ParticleRenderFilterModeVector& modes);

  signals:
    void signalSynchSettingWithAgxInstance();
    void signalRenderSettingsUpdated();

  protected:
    void connectSignals();
    void synchLabels();
    bool hasParticleUtil();
    bool hasValidColoring();

    protected slots:
    void slotMinVelEditFinished();
    void slotMaxVelEditFinished();
    void slotSliderChanged(int val);

    void slotMinNetworkForceEditFinished();
    void slotMaxNetworkForceEditFinished();
    void slotImpactEnergyEditFinished();

    void slotColorSelectionChanged(int);
    void slotPositionSliderChanged(int);
    void slotSamplingEditFinished();
    void slotParticleFilterModeChanged(int);
    void slotFilterThresholdScalarEditFinished();

    void slotParticleContactNetworkCheckBox(bool enable);
    void slotParticleTrajectoriesCheckBox(bool enable);
    void slotLimitToAnalysisBoxCheckBox(bool enable);
    void slotParticleRenderingCheckBox(bool enable);
    void slotImpactGeometryParticleCheckBox(bool enable);
    void slotImpactParticleParticleCheckBox(bool enable);
    void slotShowColorLegendCheckBox(bool enable);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    ParticleRenderFilterModeVector                        m_particleParticleFilterModes;
    ParticleFilterMode                                    m_activeParticleFilterMode;
    Ui::ParticleRenderWidget*                             m_ui;
    agx::observer_ptr<agxQt::ParticleSystemRenderUtility> m_particlRenderUtil;
    agx::observer_ptr<agxQt::TrajectoryRenderer>          m_trajectoryRenderer;
    agx::Vector<QCheckBox*>                               m_checkBoxes;
    ParticleColorMode                                     m_activeColorMode;
    ParticleColorModeVector                               m_colorModes;
  };
}

#endif // QT_VIDEOSETTINGS_WIDGET