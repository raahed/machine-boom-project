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


#ifndef AGX_CL_GL_SHARE_HANDLE_H
#define AGX_CL_GL_SHARE_HANDLE_H

#if 0 // Not used right now.

#include <agx/config/AGX_USE_OPENCL.h>
#include <agx/config/AGX_USE_OPENGL.h>
#include <agx/config.h>

#if AGX_USE_OPENCL() && AGX_USE_OPENGL()

#include <agx/agxCore_export.h>

#include <agxGL/OpenGL.h>
#include <agxCL/OpenCL.h>
#include <agxData/Buffer.h>
#include <agxData/ShareHandle.h>


namespace agxData
{
  class AGXCORE_EXPORT ClGlShareHandle : public ShareHandle
  {
  public:

    // Creating a sharing between an OpenCL and an OpenGL buffer will discard the OpenCL content.
    ClGlShareHandle(agxData::Buffer* buffer1, agxData::Buffer* buffer2);

    virtual void acquire(const agxData::Buffer* buffer);
    virtual bool isAcquired(const agxData::Buffer* buffer);
    virtual agxData::Buffer* other(const agxData::Buffer* buffer);

    void acquireOpenCL();
    void acquireOpenGL();

  private:
    bool assignBuffers(agxData::Buffer* buffer1, agxData::Buffer* buffer2);
    void passMemoryToOpenCL();

  private:
    friend class agxCL::Buffer;
    friend class agxGL::Buffer;
    agxGL::BufferRef m_glBuffer;
    agxCL::BufferRef m_clBuffer;
    CurrentHolder m_holder;

  };
}

#endif // OPENCL & OPENGL

#endif

#endif // _CL_GL_SHARE_HANDLE_H_