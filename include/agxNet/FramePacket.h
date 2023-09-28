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

#include <agxData/Frame.h>
#include <agxData/BinaryData.h>
#include <agx/Json.h>
#include <agxData/SerializedFrame.h>

namespace agxNet
{

  /**
   * A representation of a Frame prepared to be sent on a socket. The Packet class
   * generates the header data we require.
   *
   *


   | id          | response id  | URI size | header size | offset to    | size of      | header            | pad | Binary data segment (buffers/values)
   |             |              |          | in bytes    | data segment | data segment |                   |     |
    ---------------------------------------------------------------------------------------------------------------------------------
   |  UInt32     |  UInt32      |  UInt32  |  UInt32     |  UInt32      |  UInt32      | JSON / XML string ||||||| Buffer 0 |   | Buffer 1 |   | .....| | Buffer n |
    ---------------------------------------------------------------------------------------------------------------------------------
                  The id of the packet                                                                           ^
                  that this packed                                                                      Aligned to 32 bytes
                  is a response to.

    Each buffer segment containing uncompressed data
    must be aligned to at least the size of the primitive
    type that the buffer contains. The primitive type
    of a Vec3:64bit is the C++ type double, so padding
    must be added before a Vec3:64bit buffer so that the
    buffer starts at an even multiple of sizeof(double)
    from the packet start.
   */



  AGX_DECLARE_POINTER_TYPES(StructuredMessage);
  class AGXCORE_EXPORT StructuredMessage : public agx::Referenced
  {
  public:
    struct PreHeader;
    static const agx::UInt32 BINARY_SEGMENT_ALIGNMENT = 32;

    static StructuredMessage* parse(const agx::UInt8* data, size_t numBytes, bool copyBinaryData = false);
  public:
    StructuredMessage();
    StructuredMessage(const agx::String& uri);
    StructuredMessage(const agx::String& uri, const agxJson::Value& header, agxData::BinaryData *binarySegment = nullptr);

    StructuredMessage *createResponse() const;

    void setResponseId(agx::UInt32 id);
    void setUri(const agx::String& uri);
    void setHeader(const agxJson::Value& header);
    void setBinarySegment(agxData::BinaryData *binarySegment);

    /** \return The ID of this message. Set by the WebSocket that sent the message. */
    agx::UInt32 getMessageId() const;

    /** \return The ID of the message that this message is a response to. */
    agx::UInt32 getResponseId() const;

    const agx::String& getUri() const;

    agxJson::Value& getHeader();
    const agxJson::Value& getHeader() const;

    agxData::BinaryData *getBinarySegment();
    const agxData::BinaryData *getBinarySegment() const;

    // PreHeader getPreHeader() const;

  protected:
    virtual ~StructuredMessage();

    friend class WebSocket;
    void setMessageId( agx::UInt32 id ) const;

  private:
    agx::String m_uri;
    agxJson::Value m_header;
    agxData::BinaryDataRef m_binarySegment;
    agx::UInt32 m_messageId;
    agx::UInt32 m_responseId;
  };

  // Used for parsing a structured message
  struct AGXCORE_EXPORT StructuredMessage::PreHeader
  {
    PreHeader();

    agx::UInt32 id;
    agx::UInt32 responseId;
    agx::UInt32 uriSize;
    agx::UInt32 headerSize;
    agx::UInt32 binarySegmentOffset;
    agx::UInt32 binarySegmentSize;

    static const agx::UInt32 STREAM_END_OF_FILE = 0;
  };








  AGX_DECLARE_POINTER_TYPES(FramePacket);
  class AGXCORE_EXPORT FramePacket : public agxData::SerializedFrame
  {
  public:
    FramePacket(const agxData::Frame* frame);

    /// \return The preheader
    const StructuredMessage::PreHeader& getPreHeader() const;

    /// \return The offset to the binary segment
    agx::UInt32 getBinarySegmentOffset() const;

    /// \return The total size in bytes of the binary segment, including buffer padding but not post-header padding.
    agx::UInt32 getBinarySegmentSize() const;

    const agx::String& getUri() const;
    const agxJson::Value& getHeader() const;
    const agx::String& getHeaderString() const;

    /// \return The size in bytes of the padding between the header and the buffer segment.
    agx::UInt32 getPostHeaderPaddingSize() const;


    static agxData::Frame* parseFrame(agx::UInt8* data, size_t size);
    static agxData::Frame* parseFrame(const agxJson::Value& eFrame, agxData::BinaryData* binaryData);

  protected:
    virtual ~FramePacket() {}

  private:
    // agxJson::Value& createNode(agxJson::Value& eParent, const char *componentName);

    void addChildren(agxJson::Value& eParent, agxData::SerializedFrame::Component *component);

    void add(agxJson::Value& eParent, agxData::SerializedFrame::Node *node);
    void add(agxJson::Value& eParent, agxData::SerializedFrame::Component *component);
    void add(agxJson::Value& eParent, agxData::SerializedFrame::Value *value);
    void add(agxJson::Value& eParent, agxData::SerializedFrame::Buffer *buffer);
    void add(agxJson::Value& eParent, agxData::SerializedFrame::CustomBuffer *buffer);
    void add(agxJson::Value& eParent, agxData::SerializedFrame::PartialBuffer *buffer);
    void add(agxJson::Value& eParent, agxData::SerializedFrame::EntityStorage *storage);

    static agxData::SerializedFrame::Component *createComponent(const agxJson::Value& eComponent, agxData::BinaryData *binaryData);
    static agxData::SerializedFrame::Node *createStorage(const agxJson::Value& eStorage, agxData::BinaryData *binaryData);
    static agxData::SerializedFrame::Node *createBuffer(const agxJson::Value& eBuffer, agxData::BinaryData *binaryData);
    static agxData::SerializedFrame::Node *createValue(const agxJson::Value& eValue, agxData::BinaryData *binaryData);
    static void addChildren(agxData::SerializedFrame::Component *parent, const agxJson::Value& eComponent, agxData::BinaryData *binaryData);


  private:
    StructuredMessage::PreHeader m_preHeader;
    agxJson::Value m_header;
    agx::String m_uri;
    agx::String m_headerString;
    agx::UInt32 m_postHeaderPaddingSize;
  };


  /* Implementation */

  //---------------------------------------------------------------
  AGX_FORCE_INLINE const agx::String& StructuredMessage::getUri() const { return m_uri; }
  AGX_FORCE_INLINE agxJson::Value& StructuredMessage::getHeader() { return m_header; }
  AGX_FORCE_INLINE const agxJson::Value& StructuredMessage::getHeader() const { return m_header; }

  AGX_FORCE_INLINE agxData::BinaryData *StructuredMessage::getBinarySegment() { return m_binarySegment; }
  AGX_FORCE_INLINE const agxData::BinaryData *StructuredMessage::getBinarySegment() const { return m_binarySegment; }

  //---------------------------------------------------------------
  AGX_FORCE_INLINE const StructuredMessage::PreHeader& FramePacket::getPreHeader() const { return m_preHeader; }
  AGX_FORCE_INLINE agx::UInt32 FramePacket::getBinarySegmentOffset() const { return m_preHeader.binarySegmentOffset; }
  AGX_FORCE_INLINE agx::UInt32 FramePacket::getBinarySegmentSize() const { return m_preHeader.binarySegmentSize; }

  AGX_FORCE_INLINE const agx::String& FramePacket::getUri() const { return m_uri; }
  AGX_FORCE_INLINE const agxJson::Value& FramePacket::getHeader() const { return m_header; }
  AGX_FORCE_INLINE const agx::String& FramePacket::getHeaderString() const { return m_headerString; }
  AGX_FORCE_INLINE agx::UInt32 FramePacket::getPostHeaderPaddingSize() const { return m_postHeaderPaddingSize; }

}
