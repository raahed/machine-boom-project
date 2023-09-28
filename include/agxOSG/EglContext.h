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

#pragma once

#include <agx/config/AGX_USE_EGL.h>

#if AGX_USE_EGL()

#include <EGL/egl.h>
#include <EGL/eglext.h>

#undef Status
#undef Bool
#undef Convex


#include <agx/PushDisableWarnings.h>
#include <osg/GraphicsContext>
#include <agx/PopDisableWarnings.h>

namespace agxOSG
{

  /**
  EglContext is a class that provides a different way to setup
  a graphics context for 3D rendering. The main purpose is to
  simplify headless rendering in Linux without the need of
  running a X-server for context creation via GLX.

  Using this class in Linux and to avoid problems it is required that:
  - AGX is built with EGL support
  - Vendor-neutral OpenGL dispatch (glvnd) is used and not a
    legacy libGL without dispatch support.
  - Graphics card with EGL support and driver is present.
  */
  class EglContext : public osg::GraphicsContext
  {
    public:
      EglContext(osg::GraphicsContext::Traits* traits);

      virtual bool isSameKindAs(const Object* object) const;

      virtual const char* libraryName() const;
      virtual const char* className() const;

      virtual bool valid() const;

      /** Realise the GraphicsContext.*/
      virtual bool realizeImplementation();

      /** Return true if the graphics context has been realised and is ready to use.*/
      virtual bool isRealizedImplementation() const;

      /** Close the graphics context.*/
      virtual void closeImplementation();

      /** Make this graphics context current.*/
      virtual bool makeCurrentImplementation();

      /** Make this graphics context current with specified read context implementation. */
      virtual bool makeContextCurrentImplementation(osg::GraphicsContext* readContext);

      /** Release the graphics context.*/
      virtual bool releaseContextImplementation();

      /** Bind the graphics context to associated texture implementation.*/
      virtual void bindPBufferToTextureImplementation(GLenum buffer);

      /** Swap the front and back buffers.*/
      virtual void swapBuffersImplementation();

    protected:
      ~EglContext();

      void init();

      void queryDevices();
      EGLDisplay getDisplayFromDevice( EGLDeviceEXT dev );

      EGLSurface m_eglSurface;
      EGLDisplay m_eglDisplay;
      EGLContext m_eglContext;

      EGLDeviceEXT* m_eglDevices;

      agx::Int32   m_numDevices;

      bool m_valid;
      bool m_initialized;
      bool m_realized;

  };

}

#endif

