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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2010 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
 *
 * FirstPersonManipulator code Copyright (C) 2010 PCJohn (Jan Peciva)
 * while some pieces of code were taken from OSG.
 * Thanks to company Cadwork (www.cadwork.ch) and
 * Brno University of Technology (www.fit.vutbr.cz) for open-sourcing this work.
*/

#ifndef AGXOSG_FIRST_PERSON_MANIPULATOR
#define AGXOSG_FIRST_PERSON_MANIPULATOR

#include <osg/Version>

#if defined(OSG_VERSION_GREATER_OR_EQUAL)
# if OSG_VERSION_GREATER_OR_EQUAL(2,9,11)


#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgGA/StandardManipulator>
#include <osgGA/TrackballManipulator>
#include <osgViewer/Viewer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/HashSet.h>
#include <agx/Timer.h>
#include <agx/Vec2.h>

namespace agxOSG {


/** FirstPersonManipulator is base class for camera control based on position
    and orientation of camera, like walk, drive, and flight manipulators.
    At the moment, all movement keys are hard coded to
    'h' left
    'k' right
    'u' forward
    'j' backward
    'o' up
    'l' down
    Camera rotation when keeping left mouse button pressed.
    This should be changed in the future to allow customizable keyboard control.
    */
class AGXOSG_EXPORT FirstPersonManipulator : public osgGA::StandardManipulator
{
        typedef StandardManipulator inherited;

    public:

        FirstPersonManipulator( osgViewer::GraphicsWindow* window, int flags = DEFAULT_SETTINGS );
        FirstPersonManipulator( const FirstPersonManipulator& fpm,
                                const osg::CopyOp& copyOp = osg::CopyOp::SHALLOW_COPY );

        META_Object( osgGA, FirstPersonManipulator );

        virtual void setByMatrix( const osg::Matrixd& matrix );
        virtual void setByInverseMatrix( const osg::Matrixd& matrix );
        virtual osg::Matrixd getMatrix() const;
        virtual osg::Matrixd getInverseMatrix() const;

        virtual void setTransformation( const osg::Vec3d& eye, const osg::Quat& rotation );
        virtual void setTransformation( const osg::Vec3d& eye, const osg::Vec3d& center, const osg::Vec3d& up );
        virtual void getTransformation( osg::Vec3d& eye, osg::Quat& rotation ) const;
        virtual void getTransformation( osg::Vec3d& eye, osg::Vec3d& center, osg::Vec3d& up ) const;

        virtual void setVelocity( const double& velocity );
        inline double getVelocity() const;
        void setRotationSpeed( const double& rotationSpeed );
        inline double getRotationSpeed() const;
        virtual void setAcceleration( const double& acceleration, bool relativeToModelSize = false );
        double getAcceleration( bool *relativeToModelSize = nullptr ) const;
        virtual void setMaxVelocity( const double& maxVelocity, bool relativeToModelSize = false );
        double getMaxVelocity( bool *relativeToModelSize = nullptr ) const;

        virtual void setMovement( const double& wheelMovement, bool relativeToModelSize = false );
        double getMovement( bool *relativeToModelSize = nullptr ) const;

        virtual void home( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );
        virtual void home( double );

        virtual void init( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

    protected:
        FirstPersonManipulator();

        virtual bool handleFrame( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );
        virtual bool handleKeyDown( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );
        virtual bool handleKeyUp( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

        virtual bool handleMouseWheel( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

        virtual bool handleMousePush( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );
        virtual bool handleMouseRelease( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

        //virtual bool performMovementLeftMouseButton( const double eventTimeDelta, const double dx, const double dy );
        virtual bool performMouseDeltaMovement( const float dx, const float dy );
        virtual void applyAnimationStep( const double currentProgress, const double prevProgress );
        virtual bool startAnimationByMousePointerIntersection( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us );

        void moveForward( const double distance );
        void moveForward( const osg::Quat& rotation, const double distance );
        void moveRight( const double distance );
        void moveUp( const double distance );

        void handleContinuousEvents(agx::Real dt);

        osg::Vec3d m_eye;
        osg::Quat  m_rotation;
        double m_velocity;
        double m_rotationSpeed;
        double m_acceleration;
        static int m_accelerationFlagIndex;
        double _maxVelocity;
        static int m_maxVelocityFlagIndex;
        double m_movement;
        static int m_movementFlagIndex;
        agx::Timer m_eventTimer;
        agx::HashSet<int> m_continuousKeys;
        bool m_mouseDragged;
        bool m_mousedDraggedFirstFrame;
        agx::Vec2f m_mouseStartPos;
        agx::Vec2f m_mouseDelta;
        osg::observer_ptr<osgViewer::GraphicsWindow> m_window;


        class FirstPersonAnimationData : public AnimationData {
        public:
            osg::Quat m_startRot;
            osg::Quat m_targetRot;
            void start( const osg::Quat& startRotation, const osg::Quat& targetRotation, const double startTime );
        };
        virtual void allocAnimationData() { _animationData = new FirstPersonAnimationData(); }
};


//
//  inline methods
//

/// Returns velocity.
double FirstPersonManipulator::getVelocity() const  { return m_velocity; }


/// Returns the rotation speed.
double FirstPersonManipulator::getRotationSpeed() const  { return m_rotationSpeed; }


}

#endif // Version of OSG > 2.9.11
#endif // Version > 2.9.11
#endif /* OSGGA_FIRST_PERSON_MANIPULATOR */

