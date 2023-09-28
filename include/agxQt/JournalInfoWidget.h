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

#ifndef AGXQT_JOURNALINFOWIDGET_H
#define AGXQT_JOURNALINFOWIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <agxQt/ui_Controller.h>
#include <agxQt/ui_JournalInfoWidget.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/export.h>

namespace agx
{
  class Journal;
}

namespace agxQt
{
  /**
  * JournalInfoWidget for displaying journal information in qtViewer
  */
  class AGXQT_EXPORT JournalInfoWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit JournalInfoWidget(QWidget *parent = 0);
    virtual ~JournalInfoWidget();

    void setVisibleGUI(bool enable);

  public slots:
    void slotUpdateJournal();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::JournalInfoWidget  * m_ui;
    agx::Journal           * m_journal;
  };
}

#endif