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

#ifndef AGXCL_OPENCL_H
#define AGXCL_OPENCL_H


#include <agx/config/AGX_USE_OPENCL.h>
#include <agx/config.h>

#if AGX_USE_OPENCL()

#include <agx/agxCore_export.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/Singleton.h>
#include <agxData/Buffer.h>
#include <agx/Device.h>
#include <agx/Task.h>
#include <agx/Kernel.h>


#if defined(__APPLE__)
#include <OpenCL/OpenCL.h>
#include <OpenCL/cl_ext.h>
#include <OpenCL/cl_gl.h>
#elif defined(__linux__)
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#elif defined(_WIN32)
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#endif

namespace agx
{
  class TiXmlElement;
}


namespace agxCL
{
  const char* translateClError(cl_int error);
#define CHECK_CL_ERROR(x) if (x != CL_SUCCESS) { LOGGER_ERROR() << agx::String::format("OpenCL error: %d, %s\n", x, agxCL::translateClError(x)) << LOGGER_END(); }
#define CHECK_CL_ERROR_EXTENDED(x) if (x != CL_SUCCESS) { LOGGER_ERROR() << agx::String::format("[%s] OpenCL error: %d, %s\n", this->getPath().c_str(), x, agxCL::translateClError(x)) << LOGGER_END(); }

  AGX_DECLARE_POINTER_TYPES(Device);
  AGX_DECLARE_POINTER_TYPES(Buffer);
  AGX_DECLARE_POINTER_TYPES(Program);
  AGX_DECLARE_POINTER_TYPES(Kernel);
  AGX_DECLARE_POINTER_TYPES(SyncTag);

  class AGXCORE_EXPORT SyncTag : public agx::SyncTag
  {
  public:
    SyncTag(cl_event event);
    virtual Status wait();
    virtual Status getStatus();
    cl_event& getImplementation();

  protected:
    virtual ~SyncTag();

  private:
    cl_event m_event;
  };

  class AGXCORE_EXPORT Device : public agx::Device
  {
  public:
    static Device *instance();

  public:
    Device(cl_device_type type = CL_DEVICE_TYPE_GPU, int deviceIndex = 0);

    cl_context getContext() const;
    cl_device_id getId() const;

    using agx::Device::createBuffer;
    virtual agxData::Buffer *createBuffer(agxData::Format *format) override;

    virtual agx::SyncTag *sendData(const void *hostBuffer, agxData::Buffer *Buffer, size_t offset, size_t size, bool blocking) override;
    virtual agx::SyncTag *receiveData(void *hostBuffer, agxData::Buffer *Buffer, size_t offset, size_t size, bool blocking) override;

    cl_command_queue getCommandQueue();

    bool hasGlSharing() const;
    bool supportsExtension(const agx::String& extensionName) const;
    void getExtensions(agx::StringVector& extensions) const;

    unsigned long long getMaxAllocationSize();
  protected:
    virtual ~Device();

  private:
    void createContext(cl_platform_id platform);

  private:
    cl_device_id m_device;
    cl_context m_context;
    agx::Vector<cl_command_queue> m_threadQueues;
    bool m_hasGlSharing;
    unsigned long long m_maxAllocationSize;
  };



  class AGXCORE_EXPORT Buffer : public agxData::Buffer
  {
  public:
    Buffer(agxData::Format *format, Device *device, cl_mem_flags flags = CL_MEM_READ_WRITE);

    cl_mem& getImplementation();
    size_t queryImplementationSize();


    // using agxData::Buffer::read;
    // virtual void read(const void *buffer, size_t numElements, const agxData::Format *format) override;
    // void read(Buffer* source);

    #if 0
    virtual void read(const agx::String& filePath) override { agxAbort1("TODO"); };

    virtual void write(const agx::String& filePath, size_t numElements) const override { agxAbort1("TODO"); }
    #endif

    void setSharedGLBuffer(agx::UInt glHandle);

  protected:
    virtual ~Buffer();

    virtual void reallocate(size_t numElements) override;
    static void initializeElements(agxData::Buffer *buffer, const agx::IndexRange& range);

  private:
    friend class Device;
    void activate();

  private:
    cl_mem m_buffer;
    cl_mem_flags m_flags;
  };


  class AGXCORE_EXPORT Program : public agx::Referenced
  {
  public:
    Program(const agx::String& sourceFilePath, const agx::String& buildOptions = "");

    cl_program& getImplementation(Device *device);

  protected:
    virtual ~Program();
  public:
    agx::String m_sourceFilePath;
    agx::String m_buildOptions;
    typedef agx::HashTable<DeviceRef, cl_program> ProgramTable;
    ProgramTable m_programTable;
    uint64_t m_timestamp;
  };

  class AGXCORE_EXPORT ProgramManager : public agx::Singleton
  {
  public:
    static ProgramManager *instance();

    Program *getProgram(const agx::String& sourceFilePath, const agx::String& buildOptions = "");

    SINGLETON_CLASSNAME_METHOD();

  private:
    ProgramManager();

  private:
    typedef agx::HashTable<agx::String, ProgramObserver> ProgramTable;
    ProgramTable m_programTable;
  };

  class AGXCORE_EXPORT Kernel : public agx::Kernel
  {
  public:
    agxData::Val<agx::UInt> numWorkGroupsParameter;

    static Kernel *load(agx::TiXmlElement *eKernel, agx::Device *device);

  public:
    Kernel(const agx::Name& name, agx::Device *device, Program *program);
    Kernel(const agx::Name& name, agx::Device *device, const agx::String& programSourcePath, const agx::String& buildOptions = "");

    cl_kernel& getImplementation();

    // SyncTag *getTag();

    /** Global size of 0, the default, causes kernels to be launched with one thread per element in first argument buffer */
    void setGlobalSize(size_t globalSize);
    void setGlobalSize(size_t globalSizeX, size_t globalSizeY);
    void setGlobalSize(size_t globalSizeX, size_t globalSizeY, size_t globalSizeZ);

    void setWorkGroupSize(size_t groupSize);
    void setWorkGroupSize(size_t groupSizeX, size_t groupSizeY);
    void setWorkGroupSize(size_t groupSizeX, size_t groupSizeY, size_t groupSizeZ);

    // virtual void rebind() override;

    virtual void preDispatch() override;
    virtual void wait() override;

    cl_event getOpenCLEvent() const;

    virtual float getComputeTime();
    float getGpuTime();
    float getWaitTime();

  protected:
    virtual ~Kernel();
    virtual void parameterAddedCallback(agx::Parameter *parameter) override;
    virtual void parameterRemovedCallback(agx::Parameter *parameter) override;

  private:
    void dispatch();
    void waitCallback();
    void alignGlobalWorkSize(size_t globalSize[3]);
    void init();
    void parameterUpdateCallback(agx::Parameter *parameter);
    void calculateGlobalSize(size_t globalSize[3]);

  private:
    ProgramRef m_program;
    cl_kernel m_kernel;
    size_t m_workGroupSize[3];
    size_t m_globalSize[3];
    cl_uint m_workDim;
    // SyncTag m_tag;
    agx::Timer m_waitTimer;
    cl_event m_dispatchEvent;
    agx::Job m_waitJob;
    bool m_firstWaitIteration;
    agx::Parameter::Event::CallbackType m_parameterUpdateCallback;
  };


  /* Implementation */

  inline bool Device::hasGlSharing() const { return m_hasGlSharing; }

  // inline SyncTag *Kernel::getTag() { return &m_tag; }
  inline cl_kernel& Kernel::getImplementation() { return m_kernel; }
  inline cl_event Kernel::getOpenCLEvent() const { return m_dispatchEvent; }

  inline cl_mem& Buffer::getImplementation() { return m_buffer; }

  inline SyncTag::SyncTag(cl_event event) : m_event(event) {}
  inline cl_event& SyncTag::getImplementation() { return m_event; }

}

#endif

#endif
/* _AGX_OPENCL_H_ */
