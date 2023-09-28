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


#ifndef AGXOSG_SCCAMERA_MANIPULATOR
#define AGXOSG_SCCAMERA_MANIPULATOR

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgGA/StandardManipulator>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) //  warning C4251: non dll-interface class
#endif

namespace agxOSG {


  /** SCCameraManipulator is base class for camera control based on focal center,
      distance from the center, and orientation of distance vector to the eye.
      This is the base class for trackball style manipulators.*/
  class AGXOSG_EXPORT SCCameraManipulator : public osgGA::StandardManipulator
  {

    typedef osgGA::StandardManipulator inherited;

      public:

        SCCameraManipulator(int flags = DEFAULT_SETTINGS);
        SCCameraManipulator(const SCCameraManipulator& om,
                            const osg::CopyOp& copyOp = osg::CopyOp::SHALLOW_COPY );

        META_Object(agxOSG, SCCameraManipulator);

          virtual void setByMatrix( const osg::Matrixd& matrix );
          virtual void setByInverseMatrix( const osg::Matrixd& matrix );
          virtual osg::Matrixd getMatrix() const;
          virtual osg::Matrixd getInverseMatrix() const;

          virtual void setTransformation( const osg::Vec3d& eye, const osg::Quat& rotation );
          virtual void setTransformation( const osg::Vec3d& eye, const osg::Vec3d& center, const osg::Vec3d& up );
          virtual void getTransformation( osg::Vec3d& eye, osg::Quat& rotation ) const;
          virtual void getTransformation( osg::Vec3d& eye, osg::Vec3d& center, osg::Vec3d& up ) const;

          virtual void setCenter( const osg::Vec3d& center );
          const osg::Vec3d& getCenter() const;
          virtual void setRotation( const osg::Quat& rotation );
          const osg::Quat& getRotation() const;
          virtual void setDistance( double distance );
          double getDistance() const;

          virtual void setTrackballSize( const double& size );
          inline double getTrackballSize() const;
          virtual void setWheelZoomFactor( double wheelZoomFactor );
          inline double getWheelZoomFactor() const;

          virtual void setMinimumDistance( const double& minimumDistance, bool relativeToModelSize = false );
          double getMinimumDistance( bool *relativeToModelSize = nullptr ) const;

          virtual osgUtil::SceneView::FusionDistanceMode getFusionDistanceMode() const;
          virtual float getFusionDistanceValue() const;

      protected:

        virtual bool handleMousePush(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us);

        virtual bool handleMouseWheel( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

          virtual bool performMovementLeftMouseButton( const double eventTimeDelta, const double dx, const double dy );
          virtual bool performMovementMiddleMouseButton( const double eventTimeDelta, const double dx, const double dy );
          virtual bool performMovementRightMouseButton( const double eventTimeDelta, const double dx, const double dy );
          virtual bool performMouseDeltaMovement( const float dx, const float dy );
          virtual void applyAnimationStep( const double currentProgress, const double prevProgress );

          bool handleKeyDown(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us);
          bool handleKeyUp(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us);
          bool setCenterByMousePointerIntersection(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us);


          virtual void rotateTrackball( const float px0, const float py0,
                                        const float px1, const float py1, const float scale );

          bool rotate(const double eventTimeDelta, const double dx, const double dy);
          virtual void rotateWithFixedVertical( const float dx, const float dy );
          virtual void rotateWithFixedVertical( const float dx, const float dy, const osg::Vec3f& up );
          virtual void panModel( const float dx, const float dy, const float dz = 0.f );
          virtual void zoomModel( const float dy, bool pushForwardIfNeeded = true );
          void trackball( osg::Vec3d& axis, float& angle, float p1x, float p1y, float p2x, float p2y );
          float tb_project_to_sphere( float r, float x, float y );
          virtual bool startAnimationByMousePointerIntersection( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

          osg::Vec3d _center;
          osg::Quat  _rotation;
          double     _distance;

          double _trackballSize;
          double _wheelZoomFactor;

          double _minimumDistance;
          static int _minimumDistanceFlagIndex;
          osg::Vec3 _rotationCenter;

          int m_modKeyMask;

          class OrbitAnimationData : public AnimationData {
          public:
              osg::Vec3d _movement;
              void start( const osg::Vec3d& movement, const double startTime );
          };
          virtual void allocAnimationData() { _animationData = new OrbitAnimationData(); }
  };


  //
  //  inline functions
  //

  /** Get the size of the trackball relative to the model size. */
  inline double SCCameraManipulator::getTrackballSize() const  { return _trackballSize; }
  /** Get the mouse wheel zoom factor.*/
  inline double SCCameraManipulator::getWheelZoomFactor() const  { return _wheelZoomFactor; }

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif



#endif
