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

#ifndef AGXQT_RENDERCONTROLLER_H
#define AGXQT_RENDERCONTROLLER_H
#include <agxQt/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <agxQt/ui_RenderController.h>
#include <QWidget>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxQt
{
  class AGXQT_EXPORT RenderControllerWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit RenderControllerWidget(QWidget *parent = 0);

    QCheckBox* getDebugCheckBox();
    QCheckBox* getOSGCheckBox();

  protected:
    virtual ~RenderControllerWidget();

  private:
    Ui::RenderController * m_ui;
  };
}



#endif // NOTEPAD_H