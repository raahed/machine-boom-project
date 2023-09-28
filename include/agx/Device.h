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


#include <agx/config/AGX_USE_OPENGL.h>
#include <agx/config/AGX_USE_OPENCL.h>
#include <agx/agxCore_export.h>
#include <agx/Component.h>
#include <agx/HashTable.h>
#include <agx/Allocator.h>
#include <agx/MemoryPool.h>
#include <agxData/Buffer.h>
#include <agx/SyncTag.h>
#include <agx/Job.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
# pragma warning( disable : 4702)  // warning C4702: unreachable code
#endif



namespace agx
{
  AGX_DECLARE_POINTER_TYPES(Device);
  AGX_DECLARE_POINTER_TYPES(CpuDevice);
  AGX_DECLARE_POINTER_TYPES(DeviceGroup);
  // AGX_DECLARE_POINTER_TYPES(HostDevice);

  /**
  An agx::Device is an abstract representation of a device on which
  data can be stored and processed. Devices are hierarchical.

  Example:

  A distributed memory machine (computer cluster), is a device, in which
  each host machine is a device. Each host has a number of subdevices, eg
  each GPU is a device, with subdevices for compute groups
  the CPU is a device, with subdevices for each CPU socket, with subdevices for individual cores
  */
  class AGXCORE_EXPORT Device : public Component, public Object::EventListener
  {
  public:
    static agx::Model *ClassModel();

    static Device *load(TiXmlElement *eDevice, Device *device);

  public:

    /// Device dependent byte order, AGX prefix to avoid clash with macro definitions
    enum ByteOrder
    {
      AGX_LITTLE_ENDIAN,
      AGX_BIG_ENDIAN,
      UNDEFINED
    };

    /// Explicit list of device types
    enum Type
    {
      CPU,
      #if AGX_USE_OPENCL()
      OPEN_CL,
      #endif
      #if AGX_USE_OPENGL()
      OPEN_GL,
      #endif
      NUM_COMPUTE_DEVICES,

      DISK,
      REMOTE_HOST,
      GROUP
    };

  public:
    /**
    Constructor.
    \param byteOrder The byteOrder of the device.
    */
    Device(const agx::Name& name, Type type, const agx::String& typeName, ByteOrder byteOrder = UNDEFINED);

    /**
    \return The device type.
    */
    Type getType() const;

    /**
    \return The device type name.
    */
    const agx::String& getTypeName() const;

    /**
    \return The device byte order.
    */
    ByteOrder getByteOrder() const;

    /**
    Set the type format for a specific type.
    */
    void setTypeFormat(agxData::Type *type, agxData::Format *format);
    void setTypeFormat(const agx::String& typeName, const agx::String& formatName);

    /**
    \return The type format for a type. Default formatting if no special formatting is registered.
    */
    agxData::Format *getTypeFormat(agxData::Type *type);



    agxData::Buffer *createBuffer(agxData::Type *type);
    virtual agxData::Buffer *createBuffer(agxData::Format *format) = 0;

    agxData::Buffer *getBuffer(agxData::Buffer *buffer);


    /**
    Send data to a buffer on the device.
    \param hostBuffer The source buffer.
    \param Buffer The target buffer.
    \param offset The offset (number of elements).
    \param size The size (number of elements).
    \param blocking True if blocking.
    \return A transfer tag to track the transfer.
    */
    virtual agx::SyncTag *sendData(const void *hostBuffer, agxData::Buffer *Buffer, size_t offset, size_t size, bool blocking) = 0;

    /**
    Receive data from a buffer on the device.
    \param hostBuffer The target buffer.
    \param Buffer The source buffer.
    \param offset The offset (number of elements).
    \param size The size (number of elements).
    \param blocking True if blocking.
    \return A transfer tag to track the transfer.
    */
    virtual agx::SyncTag *receiveData(void *hostBuffer, agxData::Buffer *Buffer, size_t offset, size_t size, bool blocking) = 0;

  protected:
    virtual ~Device();
    // friend class HostDevice;
    void setByteOrder(ByteOrder byteOrder);

    virtual void destroyCallback(agx::Object* object) override;


  private:
    Type m_type;
    String m_typeName;
    ByteOrder m_byteOrder;
    typedef HashTable<agxData::Type *, agxData::Format *> TypeFormatTable;
    TypeFormatTable m_typeFormatTable;

    typedef HashTable<agxData::Buffer *, agxData::BufferRef> BufferTable;
    BufferTable m_bufferTable;
  };


  //---------------------------------------------------------------

  /**
  Representation of the CPU.
  */
  class AGXCORE_EXPORT CpuDevice : public Device
  {
  public:
    class SyncTag : public agx::SyncTag
    {
    public:
      virtual Status wait() override { return SYNC_TAG_COMPLETED; }
      virtual Status getStatus() override { return SYNC_TAG_COMPLETED; }
    };

  public:
    static CpuDevice *instance();

    using Device::createBuffer;
    virtual agxData::Buffer *createBuffer(agxData::Format *format) override;

    virtual agx::SyncTag* sendData(
      const void *hostBuffer, agxData::Buffer* buffer, size_t offset, size_t size, bool blocking) override;

    virtual agx::SyncTag* receiveData(
      void *hostBuffer, agxData::Buffer* buffer, size_t offset, size_t size, bool blocking) override;

  public:
    void enqueue(Task *task);
    // void setNumThreads(int numThreads);
    // int getNumThreads() const;
    void wait(bool passive = false);

  protected:
    virtual ~CpuDevice();

  private:
    CpuDevice(const Name& name = "CpuDevice");

  private:
    SyncTagRef m_dummyTag;

    // ThreadPtrVector m_threads;
    // ThreadPtrVector m_deactivatedThreads;
    MemoryPool<RangeJob> m_jobAllocator;
  };

  //---------------------------------------------------------------

  /**
  A device group contain other devices, enables hierarchical device structuring.
  */
  class AGXCORE_EXPORT DeviceGroup : public Device
  {
  public:
    DeviceGroup(const Name& name, Device *primaryDevice);

    void setPrimaryDevice(Device *device);
    Device *getPrimaryDevice();

    Device *getOrCreateDevice(Type deviceType);

    // const DeviceRefVector& getDevices() const;
    // Device *getDevice(const String& name);

    // void addDevice(const Device *device);
    // void removeDevice(const Device *device);
    // void removeDevice(const String& name);

    virtual agxData::Buffer *createBuffer(agxData::Format* /*format*/) override { agxAbort1("DeviceGroup"); return nullptr; }

    virtual agx::SyncTag *sendData(const void* /*hostBuffer*/, agxData::Buffer* /*Buffer*/, size_t /*offset*/, size_t /*size*/, bool /*blocking*/) override { agxAbort1("DeviceGroup"); return nullptr; }
    virtual agx::SyncTag *receiveData(void* /*hostBuffer*/, agxData::Buffer* /*Buffer*/, size_t /*offset*/, size_t /*size*/, bool /*blocking*/) override { agxAbort1("DeviceGroup"); return nullptr; }


  protected:
    virtual ~DeviceGroup();

  private:
    DeviceRef m_primaryDevice;
    // DeviceRefVector m_devices;
    typedef HashTable<UInt, DeviceRef> DeviceTable;
    DeviceTable m_deviceTable;
  };


  //---------------------------------------------------------------

  /* Implementation */
  AGX_FORCE_INLINE Device::Type Device::getType() const { return m_type; }
  AGX_FORCE_INLINE const String& Device::getTypeName() const { return m_typeName; }
  AGX_FORCE_INLINE Device::ByteOrder Device::getByteOrder() const { return m_byteOrder; }

  // AGX_FORCE_INLINE Device *DeviceGroup::getPrimaryDevice() { return m_primaryDevice; }
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

