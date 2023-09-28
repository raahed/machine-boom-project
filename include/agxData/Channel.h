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

#ifndef AGXDATA_CHANNEL_H
#define AGXDATA_CHANNEL_H

#include <agx/agxCore_export.h>
#include <agx/Device.h>

namespace agxData
{
  class Buffer;

  /**
  A channel is a directional data transfer connection between two buffers. The buffers must have the same attribute type (eg. Vec3 -> Vec3),
  but does not need to be the same attribute instance (eg. it is possible to set up a channel of particle forces -> rigid body forces).
  If the type format differs, data transformation is automatically done via a temporary buffer.
  */
  class AGXCORE_EXPORT Channel
  {
  public:

    /**
    Default constructor.
    */
    Channel();

    /**
    Constructor.
    \param target The target buffer.
    \param source The source buffer.
    */
    Channel(Buffer *target, Buffer *source);

    /** Destructor */
    ~Channel();


    void connect(Buffer *target, Buffer *source);

    /**
    Transmit data on the channel. If the target buffer is not large enough, it is automatically resized.
    \param targetOffset The offset in the target buffer (num elements).
    \param sourceOffset The offset in the source buffer (num elements).
    \param numElements The number of elements to transfer.
    \param blocking Set true for blocking transfer.
    \return A transfer tag used to check the status of asynchronous transfers.
    */
    agx::SyncTag *transfer(size_t targetOffset, size_t sourceOffset, size_t numElements, bool blocking);

    /**
    Perform local data copying and transformations if needed. Must be called after transfer.
    TODO Use automated callbacks in transfer tag instead
    */
    void postTransfer();


    /**
    Flip the direction of the channel.
    */
    void flip();

    /**
    \return The target buffer.
    */
    Buffer *getTargetBuffer();

    /**
    \return The source buffer.
    */
    Buffer *getSourceBuffer();

  private:
    bool m_active;
    Buffer *m_target;
    Buffer *m_source;
    void *m_transformBuffer;
    size_t m_transformBufferSize;
    size_t m_currentTransferSize;
    size_t m_targetOffset;
  };

  /* Implementation */
  AGX_FORCE_INLINE Buffer *Channel::getTargetBuffer() { return m_target; }
  AGX_FORCE_INLINE Buffer *Channel::getSourceBuffer() { return m_source; }

}


#endif /* _AGXDATA_CHANNEL_H_ */
