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

#ifndef AGXQT_OSGRENDERER_H
#define AGXQT_OSGRENDERER_H

#include <agxQt/export.h>
#include <agx/Math.h>

#include <iostream>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgGA/GUIEventAdapter>

#include <QtOpenGL/QGLWidget>
#include <QtDesigner/QDesignerExportWidget>
#include <QDesignerCustomWidgetInterface>
#include <QtDesigner>
#include <QtGui>

#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QString>
#include <QPushButton>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agxQt
{
  //! [0] //! [1]
  //QDESIGNER_WIDGET_EXPORT

  class AGXQT_EXPORT OsgRenderer : public QGLWidget
  {
    Q_OBJECT
      //! [0]

  public:
    OsgRenderer(QWidget *parent = 0);
    ~OsgRenderer();

    void setEnableResize(bool state);

    public slots:
        void slotForceGLUpdate();
        void slotEnableRecordingNotifier(bool enable);
    //signals:
    //    void updated(QTime currentTime);


    virtual QSize minimumSizeHint() const;
    virtual QSize sizeHint() const;

    osgViewer::GraphicsWindow* getGraphicsWindow();
    osgViewer::Viewer* getViewer();

    void toggleFullScreen();

  protected:
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int width, int height);

    virtual void keyPressEvent( QKeyEvent* event );
    virtual void keyReleaseEvent( QKeyEvent* event );

    virtual void mousePressEvent( QMouseEvent* event );
    virtual void mouseReleaseEvent( QMouseEvent* event );
    virtual void mouseMoveEvent( QMouseEvent* event );

    void handlePainterCalls();
    void drawText(QPainter * pianter);

  private:
    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_gw;
    osg::ref_ptr<osgViewer::Viewer> m_viewer;
    osg::ref_ptr<osg::Node> m_root;

    QTimer m_glupdateTimer;
    bool m_drawRecordNotifier;

    //! [2]
  };
  //! [1] //! [2]


  class QtKeyboardMap
  {

  public:
    QtKeyboardMap()
    {
      mKeyMap[Qt::Key_Escape     ] = osgGA::GUIEventAdapter::KEY_Escape;
      mKeyMap[Qt::Key_Delete   ] = osgGA::GUIEventAdapter::KEY_Delete;
      mKeyMap[Qt::Key_Home       ] = osgGA::GUIEventAdapter::KEY_Home;
      mKeyMap[Qt::Key_Enter      ] = osgGA::GUIEventAdapter::KEY_KP_Enter;
      mKeyMap[Qt::Key_End        ] = osgGA::GUIEventAdapter::KEY_End;
      mKeyMap[Qt::Key_Return     ] = osgGA::GUIEventAdapter::KEY_Return;
      mKeyMap[Qt::Key_PageUp     ] = osgGA::GUIEventAdapter::KEY_Page_Up;
      mKeyMap[Qt::Key_PageDown   ] = osgGA::GUIEventAdapter::KEY_Page_Down;
      mKeyMap[Qt::Key_Left       ] = osgGA::GUIEventAdapter::KEY_Left;
      mKeyMap[Qt::Key_Right      ] = osgGA::GUIEventAdapter::KEY_Right;
      mKeyMap[Qt::Key_Up         ] = osgGA::GUIEventAdapter::KEY_Up;
      mKeyMap[Qt::Key_Down       ] = osgGA::GUIEventAdapter::KEY_Down;
      mKeyMap[Qt::Key_Backspace  ] = osgGA::GUIEventAdapter::KEY_BackSpace;
      mKeyMap[Qt::Key_Tab        ] = osgGA::GUIEventAdapter::KEY_Tab;
      mKeyMap[Qt::Key_Space      ] = osgGA::GUIEventAdapter::KEY_Space;
      mKeyMap[Qt::Key_Delete     ] = osgGA::GUIEventAdapter::KEY_Delete;
      mKeyMap[Qt::Key_Alt      ] = osgGA::GUIEventAdapter::KEY_Alt_L;
      mKeyMap[Qt::Key_Shift    ] = osgGA::GUIEventAdapter::KEY_Shift_L;
      mKeyMap[Qt::Key_Control  ] = osgGA::GUIEventAdapter::KEY_Control_L;
      mKeyMap[Qt::Key_Meta     ] = osgGA::GUIEventAdapter::KEY_Meta_L;

      mKeyMap[Qt::Key_F1             ] = osgGA::GUIEventAdapter::KEY_F1;
      mKeyMap[Qt::Key_F2             ] = osgGA::GUIEventAdapter::KEY_F2;
      mKeyMap[Qt::Key_F3             ] = osgGA::GUIEventAdapter::KEY_F3;
      mKeyMap[Qt::Key_F4             ] = osgGA::GUIEventAdapter::KEY_F4;
      mKeyMap[Qt::Key_F5             ] = osgGA::GUIEventAdapter::KEY_F5;
      mKeyMap[Qt::Key_F6             ] = osgGA::GUIEventAdapter::KEY_F6;
      mKeyMap[Qt::Key_F7             ] = osgGA::GUIEventAdapter::KEY_F7;
      mKeyMap[Qt::Key_F8             ] = osgGA::GUIEventAdapter::KEY_F8;
      mKeyMap[Qt::Key_F9             ] = osgGA::GUIEventAdapter::KEY_F9;
      mKeyMap[Qt::Key_F10            ] = osgGA::GUIEventAdapter::KEY_F10;
      mKeyMap[Qt::Key_F11            ] = osgGA::GUIEventAdapter::KEY_F11;
      mKeyMap[Qt::Key_F12            ] = osgGA::GUIEventAdapter::KEY_F12;
      mKeyMap[Qt::Key_F13            ] = osgGA::GUIEventAdapter::KEY_F13;
      mKeyMap[Qt::Key_F14            ] = osgGA::GUIEventAdapter::KEY_F14;
      mKeyMap[Qt::Key_F15            ] = osgGA::GUIEventAdapter::KEY_F15;
      mKeyMap[Qt::Key_F16            ] = osgGA::GUIEventAdapter::KEY_F16;
      mKeyMap[Qt::Key_F17            ] = osgGA::GUIEventAdapter::KEY_F17;
      mKeyMap[Qt::Key_F18            ] = osgGA::GUIEventAdapter::KEY_F18;
      mKeyMap[Qt::Key_F19            ] = osgGA::GUIEventAdapter::KEY_F19;
      mKeyMap[Qt::Key_F20            ] = osgGA::GUIEventAdapter::KEY_F20;

      mKeyMap[Qt::Key_hyphen         ] = '-';
      mKeyMap[Qt::Key_Equal         ] = '=';

      mKeyMap[Qt::Key_division      ] = osgGA::GUIEventAdapter::KEY_KP_Divide;
      mKeyMap[Qt::Key_multiply      ] = osgGA::GUIEventAdapter::KEY_KP_Multiply;
      mKeyMap[Qt::Key_Minus         ] = '-';
      mKeyMap[Qt::Key_Plus          ] = '+';
      //mKeyMap[Qt::Key_H              ] = osgGA::GUIEventAdapter::KEY_KP_Home;
      //mKeyMap[Qt::Key_                    ] = osgGA::GUIEventAdapter::KEY_KP_Up;
      //mKeyMap[92                    ] = osgGA::GUIEventAdapter::KEY_KP_Page_Up;
      //mKeyMap[86                    ] = osgGA::GUIEventAdapter::KEY_KP_Left;
      //mKeyMap[87                    ] = osgGA::GUIEventAdapter::KEY_KP_Begin;
      //mKeyMap[88                    ] = osgGA::GUIEventAdapter::KEY_KP_Right;
      //mKeyMap[83                    ] = osgGA::GUIEventAdapter::KEY_KP_End;
      //mKeyMap[84                    ] = osgGA::GUIEventAdapter::KEY_KP_Down;
      //mKeyMap[85                    ] = osgGA::GUIEventAdapter::KEY_KP_Page_Down;
      mKeyMap[Qt::Key_Insert        ] = osgGA::GUIEventAdapter::KEY_KP_Insert;
      //mKeyMap[Qt::Key_Delete        ] = osgGA::GUIEventAdapter::KEY_KP_Delete;
    }

    ~QtKeyboardMap()
    {
    }

    int remapKey(QKeyEvent* event)
    {
      KeyMap::iterator itr = mKeyMap.find(event->key());
      if (itr == mKeyMap.end())
      {
        return int(*(event->text().toLatin1().data()));
      }
      else
        return itr->second;
    }

  private:
    typedef std::map<unsigned int, int> KeyMap;
    KeyMap mKeyMap;
  };

  static QtKeyboardMap s_QtKeyboardMap;
}


#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif
