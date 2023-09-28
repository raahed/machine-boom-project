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

#ifndef SIMULATIONSTRUCTUREWIDGET_H
  #define SIMULATIONSTRUCTUREWIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <QList>
#include <QVariant>
#include <QPixmap>
#include <agxQt/ui_SimulationStructureWidget.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/TimeStamp.h>
#include <agx/agx.h>
#include <agxSDK/Simulation.h>
#include <agxQt/export.h>
#include <iostream>
#include <memory>

namespace agxQt
{

  class TreeModel;
  class TreeItem;

  /**
  * Widget for displaying simulation structure
  */
  class AGXQT_EXPORT SimulationStructureWidget : public QWidget
  {
    Q_OBJECT

  public:

  public:
    explicit SimulationStructureWidget(QWidget *parent = 0);
    virtual ~SimulationStructureWidget();

    void setSimulation(agxSDK::Simulation * simulation);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::SimulationStructure *           m_ui;
    agxSDK::Simulation *                m_activeSimulation;
    std::unique_ptr<TreeModel>          m_activeModel;
  };

  /**
  * TreeItem
  */
  class TreeItem
  {
  public:
    explicit TreeItem(const QList<QVariant> &data, const QPixmap& image, TreeItem *parentItem = 0);
    ~TreeItem();

    void appendChild(TreeItem *child);

    TreeItem *child(int row);
    int childCount() const;
    int columnCount() const;
    QVariant data(int column) const;
    int row() const;
    TreeItem *parentItem();
    QPixmap getImage();

  private:
    QList<TreeItem*> m_childItems;
    QList<QVariant>  m_itemData;
    TreeItem *       m_parentItem;
    QPixmap          m_image;
  };

  /**
  * TreeModel
  */
  class TreeModel : public QAbstractItemModel
  {
    Q_OBJECT

  public:
    explicit TreeModel(agxSDK::Simulation* simulation, QObject *parent = 0);
    ~TreeModel();

    QVariant data(const QModelIndex &index, int role) const Q_DECL_OVERRIDE;
    Qt::ItemFlags flags(const QModelIndex &index) const Q_DECL_OVERRIDE;
    QVariant headerData(int section, Qt::Orientation orientation,
      int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;
    QModelIndex index(int row, int column,
      const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
    QModelIndex parent(const QModelIndex &index) const Q_DECL_OVERRIDE;
    int rowCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
    int columnCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;

  private:
    void setupModelData(agxSDK::Simulation * simulation, TreeItem *parent);

    TreeItem* rootItem;
  };
}


#endif