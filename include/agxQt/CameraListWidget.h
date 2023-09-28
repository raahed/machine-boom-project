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

#ifndef AGXQT_CAMERALISTWIDGET_H
#define AGXQT_CAMERALISTWIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <agxQt/ui_CameraList.h>
#include <QCloseEvent>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/TimeStamp.h>
#include <agx/agx.h>
#include <agxQt/export.h>
#include <iostream>
#include <agx/Vector.h>
#include <agx/String.h>
#include <agxQt/CameraViewSet.h>

namespace agxQt
{
  /**
  * Model used to access data from the camera view set (Model-view Design pattern)
  */
  class CameraViewModel : public QAbstractListModel
  {
    Q_OBJECT

  public:
    CameraViewModel(CameraDescriptionSet * cameraSet, QObject *parent = 0)
      : QAbstractListModel(parent), m_cameraSet(cameraSet) {}

    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant data(const QModelIndex &index, int role) const;

    QVariant headerData(int section, Qt::Orientation orientation,
      int role = Qt::DisplayRole) const;

    Qt::ItemFlags flags(const QModelIndex &index) const;

    bool setData(const QModelIndex &index, const QVariant &value,
      int role = Qt::EditRole);

    CameraDescriptionSet * getCameraDescriptionSet();

  private:
    CameraDescriptionSet * m_cameraSet;
  };

  /**
  * Class used for showing and manipulating data regarding loaded camera views
  */
  class AGXQT_EXPORT CameraListWidget : public QWidget
  {
    Q_OBJECT

  public:

    explicit CameraListWidget(QWidget *parent = 0);

    virtual ~CameraListWidget();

    /// Setup the view widget from a CameraDescriptionSet
    void setupFromCameraDescriptionSet(CameraDescriptionSet * cameraSet);

public slots:
    void slotUpdateSelectedViewInList();

signals:
    void signalSetActiveView();
    void signalStoreCamera();
    void signalShowCameraList(bool);

    protected slots:
      void slotDeleteButtonPressed();
      void slotSetViewButtonPressed();
      void slotStoreViewButtonPressed();

  protected:
    virtual void closeEvent( QCloseEvent * event );

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::CameraViewList * m_ui;
    CameraViewModel * m_model;
  };
}

#endif
