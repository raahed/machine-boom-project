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

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2006 Robert Osfield
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
*/




//\todo We must check if we can use OSG camera code, otherwise we must rewrite...



#ifndef AGXGL_CAMERA_H
#define AGXGL_CAMERA_H

#include <agx/agxPhysics_export.h>
#include <agx/Object.h>
#include <agx/Vec3.h>
#include <agx/Clock.h>
#include <agx/Matrix4x4.h>
#include <agx/AffineMatrix4x4.h>

namespace agxGL
{
  class AGXPHYSICS_EXPORT Camera : public agx::Component
  {
  public:
    static agx::Model *ClassModel();

    static Camera *load(agx::TiXmlElement *eCamera, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eCamera) override;

    static Camera *getActiveCamera();

  public:
    enum Projection
    {
      PERSPECTIVE,
      ORTHOGONAL
    };

    Camera(const agx::Name& name = "camera");

    void setViewPort(unsigned width, unsigned height);


    /** Set the projection matrix. Can be thought of as setting the lens of a camera. */
    void setProjectionMatrix(const agx::Matrix4x4& matrix);

    /** Set to an orthographic projection. See OpenGL glOrtho for documentation further details.*/
    void setProjectionMatrixAsOrtho(double left, double right,
                                    double bottom, double top,
                                    double zNear, double zFar);

    /** Set to a perspective projection. See OpenGL glFrustum documentation for further details.*/
    void setProjectionMatrixAsFrustum(double left, double right,
                                      double bottom, double top,
                                      double zNear, double zFar);


    /** Create a symmetrical perspective projection, See OpenGL gluPerspective documentation for further details.
      * Aspect ratio is defined as width/height.*/
    void setProjectionMatrixAsPerspective(double fovy, double aspectRatio,
                                          double zNear, double zFar);

    /** Get the projection matrix.*/
    inline const agx::Matrix4x4& getProjectionMatrix() const;

    /** Get the orthographic settings of the orthographic projection matrix.
      * Returns false if matrix is not an orthographic matrix, where parameter values are undefined.*/
    bool getProjectionMatrixAsOrtho(double& left, double& right,
                                    double& bottom, double& top,
                                    double& zNear, double& zFar) const;

    /** Get the frustum setting of a perspective projection matrix.
      * Returns false if matrix is not a perspective matrix, where parameter values are undefined.*/
    bool getProjectionMatrixAsFrustum(double& left, double& right,
                                      double& bottom, double& top,
                                      double& zNear, double& zFar) const;

    /** Get the frustum setting of a symmetric perspective projection matrix.
      * Returns false if matrix is not a perspective matrix, where parameter values are undefined.
      * Note, if matrix is not a symmetric perspective matrix then the shear will be lost.
      * Asymmetric matrices occur when stereo, power walls, caves and reality center display are used.
      * In these configurations one should use the 'getProjectionMatrixAsFrustum' method instead.*/
    bool getProjectionMatrixAsPerspective(double& fovy,double& aspectRatio,
                                          double& zNear, double& zFar) const;


    /** Set the view matrix. Can be thought of as setting the position of the world relative to the camera in camera coordinates. */
    void setViewMatrix(const agx::Matrix4x4& matrix);

    /** Get the view matrix. */
    inline const agx::Matrix4x4& getViewMatrix() const;

    /** Set to the position and orientation of view matrix, using the same convention as gluLookAt. */
    void setViewMatrixAsLookAt(const agx::Vec3& eye,const agx::Vec3& center,const agx::Vec3& up);

    /** Get to the position and orientation of a modelview matrix, using the same convention as gluLookAt. */
    void getViewMatrixAsLookAt(agx::Vec3& eye,agx::Vec3& center,agx::Vec3& up,double lookDistance=1.0) const;


    void alignUpVector(const agx::Vec3& up = agx::Vec3(0, 0, 1));
    agx::Vec3 getForwardVector() const;

    // void setPosition(const agx::Vec3& position);
    // void setUpVector(const agx::Vec3& upVector);
    // void setFocusPoint(const agx::Vec3& focusPoint);
    // void setLookAtVector(const agx::Vec3& lookAtVector);
    //
    // inline agx::Vec3 getPosition();
    // inline agx::Vec3 getUpVector();
    // inline agx::Vec3 getFocusPoint();
    // inline agx::Vec3 getLookAtVector();
    // inline Real getFov();

    void move(const agx::Vec3& offset);
    void moveLocal(const agx::Vec3& offset);


    void calculatePickRay(const agx::Vec2& mousePoint, agx::Vec3& pos, agx::Vec3& direction);

    static Camera *mainCamera();
    static void setMainCamera(Camera *camera);

    void render();
  protected:
    class RenderTask;
    virtual ~Camera();
    void transformCallback(agxData::Value *);
    void updateFov(agxData::Value *);


    // void cameraAdjustUpVector();

  protected:
    // agx::FrameRef m_transform;
    // Real m_lookDistance;
    agxData::ValueRefT<agx::AffineMatrix4x4> m_transform;
    agxData::ValueRefT<agx::Matrix4x4>       m_viewMatrix;
    agxData::ValueRefT<agx::Matrix4x4>       m_projectionMatrix;
    agxData::ValueRefT<agx::Matrix4x4>       m_modelViewProjectionMatrix;
    agxData::ValueRefT<agx::Real>            m_fovX;
    agxData::ValueRefT<agx::Real>            m_fovY;
    agxData::ValueRefT<agx::UInt>            m_windowWidth;
    agxData::ValueRefT<agx::UInt>            m_windowHeight;

    // agxData::Val<agx::Matrix3x3> m_normalMatrix;

    agx::Real m_ratio;
    agx::ClockRef m_clock;
    // agx::Real m_fov;
    // agx::AffineMatrix4x4 m_projectionMatrix;
    // agx::AffineMatrix4x4 m_viewMatrix;
    agxData::Value::Event::CallbackType m_transformCallback;
    agxData::Value::Event::CallbackType m_projectionCallback;
    agx::TaskRef m_renderTask;
  };

  AGX_DECLARE_POINTER_TYPES(Camera);


  inline const agx::Matrix4x4& Camera::getProjectionMatrix() const { return m_projectionMatrix->get(); }
  inline const agx::Matrix4x4& Camera::getViewMatrix() const { return m_viewMatrix->get(); }

  // inline agx::Vec3 Camera::getPosition() { return m_transform->getPosition(); }
  // inline agx::Vec3 Camera::getUpVector() { return m_up; }
  // inline agx::Vec3 Camera::getFocusPoint() { return m_focus; }
  // inline agx::Vec3 Camera::getLookAtVector() { return m_focus - m_transform->getPosition(); }
  // inline Real Camera::getFov() { return m_fov; }
}


#endif /* _AGX_CAMERA_H_ */

