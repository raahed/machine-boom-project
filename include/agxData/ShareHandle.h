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


#ifndef AGXDATA_SHARE_HANDLE_H
#define AGXDATA_SHARE_HANDLE_H


#include <agx/agxCore_export.h>
#include <agx/Vector.h>
#include <agx/Referenced.h>
#include <agx/SymmetricPair.h>

namespace agxData
{
  class Buffer;

  AGX_DECLARE_POINTER_TYPES(ShareHandle);
  class AGXCORE_EXPORT ShareHandle : public agx::Referenced
  {
  public:
    static bool hasBridge(Buffer *buffer1, Buffer *buffer2);
    static ShareHandle *create(Buffer *buffer1, Buffer *buffer2);

  public:
    ShareHandle();

    void addBuffer(Buffer *buffer);

    Buffer *getActiveBuffer();
    const Buffer *getActiveBuffer() const;

    void activate(Buffer* buffer);
    void removeBuffer(Buffer *buffer);

    void reinit();

  protected:
    virtual ~ShareHandle();

  public:
    typedef void (*ActivationFn)(Buffer *activationTarget, Buffer *currentHolder);
    typedef agx::SymmetricPair<agx::UInt> DevicePairDescriptor;

    AGX_DECLARE_POINTER_TYPES(PlatformBridge);
    class PlatformBridge : public agx::Referenced
    {
    public:
      PlatformBridge(agx::UInt type1, agx::UInt type2);
      virtual ~PlatformBridge();

      virtual void init(Buffer *buffer1, Buffer *buffer2);
      virtual void activate(Buffer *activationTarget, Buffer *currentHolder) = 0;

      DevicePairDescriptor getDescriptor() const;
    private:
      DevicePairDescriptor m_descriptor;
    };

  private:
    Buffer *m_activeBuffer;
    agx::Vector<Buffer *> m_buffers;
  };


  /* Implementation */
  AGX_FORCE_INLINE Buffer *ShareHandle::getActiveBuffer() { return m_activeBuffer; }
  AGX_FORCE_INLINE const Buffer *ShareHandle::getActiveBuffer() const { return m_activeBuffer; }

  AGX_FORCE_INLINE ShareHandle::DevicePairDescriptor ShareHandle::PlatformBridge::getDescriptor() const { return m_descriptor; }

}

#endif // AGXDATA_SHARE_HANDLE_H
