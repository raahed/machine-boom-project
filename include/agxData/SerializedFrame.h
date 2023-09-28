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

#ifndef AGXDATA_SERIALIZEDDATA_H
#define AGXDATA_SERIALIZEDDATA_H

#include <agxData/BinaryData.h>
#include <agxData/Buffer.h>
#include <agxData/EntityStorage.h>
#include <agxData/Frame.h>
#include <agx/HashVector.h>
#include <agx/agx_vector_types.h>

namespace agx
{
  class TiXmlElement;
}

namespace agxData
{
  class FrameReader;

  /**
  Serialization of structured agx data.
  Handles serialization of non-trivial data types, eg agx::Name, agxData::Array, agxData::EntityPtr
  */
  AGX_DECLARE_POINTER_TYPES(SerializedFrame);
  class AGXCORE_EXPORT SerializedFrame : public agx::Referenced
  {
  public:
    enum HeaderFormat
    {
      INVALID=-1,
      XML_HEADER,
      JSON_HEADER,
      BINARY_HEADER
    };

    static const agx::UInt64 DATA_SEGMENT_ALIGNMENT = 32;

  public:
    AGX_DECLARE_POINTER_TYPES(Node);
    AGX_DECLARE_VECTOR_TYPES(Node);

    AGX_DECLARE_POINTER_TYPES(Component);
    AGX_DECLARE_POINTER_TYPES(DataNode);
    AGX_DECLARE_POINTER_TYPES(Value);
    AGX_DECLARE_POINTER_TYPES(Buffer);
    AGX_DECLARE_POINTER_TYPES(CustomBuffer);
    AGX_DECLARE_POINTER_TYPES(PartialBuffer);
    AGX_DECLARE_POINTER_TYPES(EntityStorage);

    AGX_DECLARE_POINTER_TYPES(StorageBatchEvent);

  public:
    SerializedFrame(const agxData::Frame* frame);
    SerializedFrame(SerializedFrame::Component* root);

    static Frame* createFrame(const Component* root, agx::UInt index, agx::Real timeStamp, agx::Real timeStep);

    size_t getTotNumBytes() const;

    Component* getRoot();

    // For binary header, names are stored as indices, with a common name table
    typedef agx::HashVector<agx::Name, agx::UInt32> ExportNameTable;
    typedef agx::HashVector<agx::UInt32, agx::Name> ImportNameTable;

    BinaryData* writeBinaryHeader(ExportNameTable& nameTable) const;
    BinaryData* writeBinaryHeader() const;

    agx::TiXmlElement* writeXmlHeader() const;
    // void writeJsonHeader(agxJson::Value& root) const;

    void writeDataSegment(std::ostream& binaryFile);

    static Frame* extractFrame(BinaryData* headerSegment, BinaryData* dataSegment);
    static Frame* extractFrame(BinaryData* headerSegment, BinaryData* dataSegment, ImportNameTable& nameTable, bool clearNameTableOnKeyFrame = true, FrameReader* frameReader = nullptr);
    static Frame* extractFrame(agx::TiXmlElement* eFrame, BinaryData* dataSegment);


    static void updateNameTable(BinaryData* headerSegment, ImportNameTable& nameTable);

    struct BinarySegment
    {
      BinarySegment() {}
      BinarySegment(BinaryData* _data, size_t _offset) : data(_data), offset(_offset) {}

      BinaryDataRef data;
      size_t offset;
    };

    typedef agx::Vector<BinarySegment> BinarySegmentVector;

    const BinarySegmentVector& getBinarySegments() const;

  protected:
    virtual ~SerializedFrame();



  private:
    agxData::Buffer* createCustomBuffer(const agx::Name& name, Format* format, size_t size);
    void addChildren(Component* eParent, const agx::Component* component);
    void add(Component* eParent, const agxData::Frame::EntityStorage* storage);
    void add(Component* eParent, const agx::Component* component);
    void add(Component* eParent, const agxData::Buffer* buffer);
    void add(Component* eParent, const agxData::Frame::PartialBuffer* buffer);
    void add(Component* eParent, const agxData::Value* value);

    void addArrayBuffer(Component* eParent, const agxData::Buffer* buffer);
    void addPtrBuffer(Component* eParent, const agxData::Buffer* buffer);

    template <typename StringT>
    void addStringBuffer(Component* eParent, const agxData::Buffer* buffer);

    void writeXmlEvents(agx::TiXmlElement* eParent) const;

  private:
    static size_t createComponent(agx::Component* parent, const Component* serializedComponent);
    static size_t createChildren(agx::Component* parent, const Component* serializedComponent);
    static size_t createValue(agx::Component* parent, const Value* serializedValue);
    static size_t createStorage(agx::Component* parent, const EntityStorage* serializedStorage);
    static size_t createBuffer(agx::Component* parent, const Buffer* serializedBuffer);
    static size_t createBuffer(agx::Component* parent, const CustomBuffer* serializedBuffer);
    static size_t createBuffer(agx::Component* parent, const PartialBuffer* serializedBuffer);

    static agxData::Buffer* createBuffer(const Buffer* serializedBuffer);
    static agxData::Buffer* createBuffer(const CustomBuffer* serializedBuffer);
    static agxData::Buffer* createArrayBuffer(const CustomBuffer* serializedBuffer);
    static agxData::Buffer* createPtrBuffer(const CustomBuffer* serializedBuffer);

    template <typename T>
    static agxData::Buffer* createNameBuffer(const CustomBuffer* serializedBuffer);


  public:
    struct BinaryHeader;
    struct XmlHeader;
    struct JsonHeader;

  private:
    const Frame* m_frame;
    ComponentRef m_root;
    BinarySegmentVector m_binarySegments;
    BufferRefVector m_customBuffers;
    size_t m_currentOffset;
    size_t m_totNumBytes;
  };

  //---------------------------------------------------------------

  class AGXCORE_EXPORT SerializedFrame::StorageBatchEvent : public agx::Object::Event
  {
  public:
    enum Type
    {
      CREATE,
      DESTROY,
      PERMUTE
    };

  public:
    StorageBatchEvent(Type type, agxData::EntityStorage* storage, size_t reserve = 0);

    void add(agx::UInt64 id);
    void add(agxData::EntityRange& instances);
    void add(agxData::Array<agxData::EntityPtr>& instances);
    void add(agxData::Array<agx::Index>& instances);

    Type getType() const;
    agxData::EntityStorage* getStorage();
    const agx::UInt64Vector& getIdVector() const;

  private:
    Type m_type;
    agxData::EntityStorage* m_storage;
    agx::UInt64Vector m_ids;
  };


  struct SerializedFrame::BinaryHeader
  {
    /**
    The binary header consist of two segments, first
    a name table with shared strings/names. Then a
    in-place recursive hierarchy of the serialized node headers.

    ===========================
    FrameMeta (see declaration below)
    FrameMeta-EndTag
    ---------------------------
    NumNames
    NameStartId
    Name0-length
    Name0-charBuffer

    Name1-length
    Name1-charBuffer
    NameTable-EndTag
    ---------------------------
    NumEvents

    Event0
    Event1
    ...
    EventList-EndTag
    ---------------------------
    RootNode
    <in-place recursive child nodes>
    ...
    RootNode-EndTag
    ===========================
    */

    typedef agx::UInt32 NameId;

    struct FrameMeta
    {
      agx::UInt64 index;
      agx::Real64 timeStamp;
      agx::Real64 timeStep;
      agx::UInt32 id;
      agx::Bool isKeyFrame;
      agx::UInt64 headerSize;
      agx::UInt64 dataSegmentSize;
      agx::UInt32 endTag;
    };

    struct Node
    {
      agx::UInt32 type;
      agx::UInt32 id;
      NameId name;
    };

    struct DataNode : public Node
    {
      NameId format;
      agx::UInt32 numBytes;
      agx::UInt32 offset;
    };

    struct Value : public DataNode
    {
    };

    struct Buffer : public DataNode
    {
      agx::UInt32 numElements;
    };

    struct Component : public Node
    {
      agx::UInt32 numChildren;
    };

    struct CustomBuffer : public Component
    {
      NameId format;
      agx::UInt32 numElements;
    };

    struct PartialBuffer : public Component
    {
      NameId format;
      agx::UInt32 numElements;
      agx::UInt32 fullSize;
    };

    struct EntityStorage : public Component
    {
      NameId entityModel;
      agx::UInt32 numElements;
      agx::UInt32 capacity;
    };


    ////////////////////////////////////

    struct Event
    {
      NameId type;
    };

    struct StorageEvent : public Event
    {
      // NameId storagePath;
      agx::UInt32 storageId;
    };

    struct CreateInstanceEvent : public StorageEvent
    {
    };

    struct CreateInstancesEvent : public StorageEvent
    {
      agx::UInt32 numInstances;
    };

    struct DestroyInstanceEvent : public StorageEvent
    {
      agx::UInt32 instanceId;
    };

    struct DestroyInstancesEvent : public StorageEvent
    {
      agx::UInt32 rangeBegin;
      agx::UInt32 rangeEnd;
    };

    struct StorageBatchEvent : public StorageEvent
    {
      agx::UInt32 batchType;
      agx::UInt32 numIds;
      // ... list of ids
    };

    struct PermuteEvent : public StorageEvent
    {
    };



    // All components have an end-tag for safety, child nodes are written in-place
    static const agx::UInt32 EndTag;


    static NameId getNameId(const agx::Name& name, ExportNameTable& nameTable);

    static ByteStream writeEvents(const agx::Object::EventRefVector& events, ByteStream stream, ExportNameTable& nameTable);

    static size_t calculateSerializedEventSize(const agx::Object::EventRefVector& events);
    static BinaryData* writeEvents(const agx::Object::EventRefVector& events);

    static size_t calculateNameTableSize(const ExportNameTable& nameTable, size_t oldNumNames = 0);
    static ByteStream writeNameTable(const ExportNameTable& nameTable, ByteStream stream, size_t oldNumNames = 0);

    static ByteStream extractNameTable(ByteStream headerStream, ImportNameTable& nameTable);


    class FrameExtractor
    {
    public:
      FrameExtractor(BinaryData* headerSegment, BinaryData* dataSegment, ImportNameTable& nameTable, FrameReader* frameReader);
      ~FrameExtractor();

      Frame* extract(bool clearNameTableOnKeyFrame);


      ImportNameTable& extractNameTable();
      agx::Object::EventRefVector& extractEvents();

    private:

      agx::Name getName(NameId nameId);
      agxData::Format* getFormat(NameId formatNameId);
      agxData::EntityModel* getEntityModel(NameId entityModelNameId);

      SerializedFrame::Node* extractNode(bool skip = false);
      SerializedFrame::Component* extractComponent(bool skip);
      SerializedFrame::Value* extractValue(bool skip);
      SerializedFrame::Buffer* extractBuffer(bool skip);
      SerializedFrame::PartialBuffer* extractPartialBuffer(bool skip);
      SerializedFrame::CustomBuffer* extractCustomBuffer(bool skip);
      SerializedFrame::EntityStorage* extractEntityStorage(bool skip);
      void extractChildren(bool skip, SerializedFrame::Component* parent, size_t numChildren);

      FrameExtractor& operator=(const FrameExtractor&) = delete;

    private:
      ByteStream m_headerStream;
      BinaryDataRef m_headerSegment;
      BinaryDataRef m_dataSegment;
      ImportNameTable& m_nameTable;
      FrameReader* m_frameReader;
      agx::Object::EventRefVector m_events;
      agx::Path m_nodePath;
    };



  };

  //---------------------------------------------------------------

  struct SerializedFrame::XmlHeader
  {
    static SerializedFrame::Node* extractNode(const agx::TiXmlElement* eNode, BinaryData* binarySegment);
    static SerializedFrame::Component* extractComponent(const agx::TiXmlElement* eComponent, BinaryData* binarySegment);
    static SerializedFrame::Value* extractValue(const agx::TiXmlElement* eValue, BinaryData* binarySegment);
    static SerializedFrame::Buffer* extractBuffer(const agx::TiXmlElement* eBuffer, BinaryData* binarySegment);
    static SerializedFrame::PartialBuffer* extractPartialBuffer(const agx::TiXmlElement* eBuffer, BinaryData* binarySegment);
    static SerializedFrame::CustomBuffer* extractCustomBuffer(const agx::TiXmlElement* eBuffer, BinaryData* binarySegment);
    static SerializedFrame::EntityStorage* extractEntityStorage(const agx::TiXmlElement* eStorage, BinaryData* binarySegment);
    static void extractChildren(SerializedFrame::Component* parent, const agx::TiXmlElement* eNode, BinaryData* binarySegment);
    static void extractFrameEvents(const agx::TiXmlElement* eFrame, Frame* frame, BinaryData *);
  };

  //---------------------------------------------------------------

  struct SerializedFrame::JsonHeader
  {

  };

  //---------------------------------------------------------------

  class AGXCORE_EXPORT SerializedFrame::Node : public agx::Referenced
  {
  public:
    enum Type
    {
      INVALID=-1,
      VALUE,
      BUFFER,
      COMPONENT,
      CUSTOM_BUFFER,
      PARTIAL_BUFFER,
      ENTITY_STORAGE
    };

    Node(const agx::Name& name, agx::Index id);

    const agx::Name& getName() const;
    Component* getParent();
    agx::String getPath() const;

    void setId(agx::Index id);
    agx::Index getId() const;

    Type getType() const;
    void setType(Type type);

    const agx::Name& getTypeName() const;

    virtual size_t getBinaryHeaderNumBytes() const = 0;
    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable);
    void writeBinaryHeader(BinaryHeader::Node& eNode, ExportNameTable& nameTable);
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) = 0;
    virtual void writeDataSegment(std::ostream& binaryFile, agx::UInt64 baseOffset) = 0;

  protected:
    friend class Component;
    void setParent(Component* parent);

  private:
    Type m_type;
    agx::Index m_id;
    agx::Name m_name;
    Component* m_parent;
  };

  class AGXCORE_EXPORT SerializedFrame::Component : public Node
  {
  public:
    Component(const agx::Name& name, agx::Index id = agx::InvalidIndex);

    void addChild(Node* child);

    const NodeRefVector& getChildren() const;

    template <typename T>
    T* getChild(const char* name);

    template <typename T>
    const T* getChild(const char* name) const;

    virtual size_t getBinaryHeaderNumBytes() const override;
    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;

    template<typename ComponentT>
    ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable, ComponentT& eComponent);
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) override;
    void writeChildren(agx::TiXmlElement* eParent);
    virtual void writeDataSegment(std::ostream& binaryFile, agx::UInt64 baseOffset) override;

  private:
    NodeRefVector m_children;
  };

  class AGXCORE_EXPORT SerializedFrame::DataNode : public Node
  {
  public:
    DataNode(const agx::Name& name, agx::Index id, BinaryData* data, const Format* format, size_t numBytes, size_t offset);

    const Format* getFormat() const;
    size_t getNumBytes() const;
    size_t getOffset() const;
    BinaryData* getData();
    const BinaryData* getData() const;

    virtual size_t getBinaryHeaderNumBytes() const override;

    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;
    void writeBinaryHeader(BinaryHeader::DataNode& eNode, ExportNameTable& nameTable);
    virtual void writeDataSegment(std::ostream& binaryFile, agx::UInt64 baseOffset) override;

  private:
    FormatConstRef m_format;
    size_t m_numBytes;
    size_t m_offset;
    BinaryDataRef m_data;
  };

  class AGXCORE_EXPORT SerializedFrame::Value : public DataNode
  {
  public:
    Value(const agx::Name& name, agx::Index id, BinaryData* data, const Format* format, size_t offset);

    // virtual size_t getBinaryHeaderNumBytes() const override;
    // virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) override;

  private:
  };

  class AGXCORE_EXPORT SerializedFrame::Buffer : public DataNode
  {
  public:
    Buffer(const agx::Name& name, agx::Index id, BinaryData* data, const Format* format, size_t numElements, size_t byteOffset);

    size_t getNumElements() const;
    // size_t capacity() const;

    virtual size_t getBinaryHeaderNumBytes() const override;
    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) override;

  private:
    size_t m_numElements;
    // size_t m_capacity;
  };

  class AGXCORE_EXPORT SerializedFrame::CustomBuffer : public Component
  {
  public:
    CustomBuffer(const agx::Name& name, agx::Index id, const Format* format, size_t numElements);

    const Format* getFormat() const;
    size_t getNumElements() const;
    // size_t capacity() const;

    virtual size_t getBinaryHeaderNumBytes() const override;
    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) override;

  private:
    FormatConstRef m_format;
    size_t m_numElements;
    // size_t m_capacity;
  };

  class AGXCORE_EXPORT SerializedFrame::PartialBuffer : public Component
  {
  public:
    PartialBuffer(agxData::Frame::PartialBuffer* buffer);
    PartialBuffer(const agx::Name& name, agx::Index id, const Format* format, size_t numElements, size_t fullSize);

    const Format* getFormat() const;
    size_t getNumElements() const;
    size_t fullSize() const;
    // size_t fullCapacity() const;

    virtual size_t getBinaryHeaderNumBytes() const override;
    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) override;

  private:
    FormatConstRef m_format;
    size_t m_numElements;
    size_t m_fullSize;
    // size_t m_fullCapacity;
  };

  class AGXCORE_EXPORT SerializedFrame::EntityStorage : public Component
  {
  public:
    EntityStorage(const agxData::Frame::EntityStorage* storage);
    EntityStorage(const agx::Name& name, agx::Index id, EntityModel* model, size_t numElements, size_t capacity);

    EntityModel* getModel();
    const EntityModel* getModel() const;
    size_t getNumElements() const;
    size_t capacity() const;

    virtual size_t getBinaryHeaderNumBytes() const override;
    virtual ByteStream writeBinaryHeader(ByteStream stream, ExportNameTable& nameTable) override;
    virtual agx::TiXmlElement* writeXmlHeader(agx::TiXmlElement* eParent) override;

  private:
    EntityModelRef m_model;
    size_t m_numElements;
    size_t m_capacity;
  };




  /* Implementation */

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE SerializedFrame::Component* SerializedFrame::getRoot() { return m_root; }
  AGX_FORCE_INLINE size_t SerializedFrame::getTotNumBytes() const { return m_totNumBytes; }
  AGX_FORCE_INLINE const SerializedFrame::BinarySegmentVector& SerializedFrame::getBinarySegments() const { return m_binarySegments; }


  //---------------------------------------------------------------

  AGX_FORCE_INLINE SerializedFrame::Node::Type SerializedFrame::Node::getType() const { return m_type; }
  AGX_FORCE_INLINE const agx::Name& SerializedFrame::Node::getName() const { return m_name; }
  AGX_FORCE_INLINE agx::Index SerializedFrame::Node::getId() const { return m_id; }
  AGX_FORCE_INLINE SerializedFrame::Component* SerializedFrame::Node::getParent() { return m_parent; }
  AGX_FORCE_INLINE void SerializedFrame::Node::setParent(Component* parent) { m_parent = parent; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE const Format* SerializedFrame::DataNode::getFormat() const { return m_format; }
  AGX_FORCE_INLINE size_t SerializedFrame::DataNode::getNumBytes() const { return m_numBytes; }
  AGX_FORCE_INLINE size_t SerializedFrame::DataNode::getOffset() const { return m_offset; }

  AGX_FORCE_INLINE BinaryData* SerializedFrame::DataNode::getData() { return m_data; }
  AGX_FORCE_INLINE const BinaryData* SerializedFrame::DataNode::getData() const { return m_data; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE const SerializedFrame::NodeRefVector& SerializedFrame::Component::getChildren() const { return m_children; }


  template <typename T>
  AGX_FORCE_INLINE T* SerializedFrame::Component::getChild(const char* name)
  {
    for (size_t i = 0; i < m_children.size(); ++i)
    {
      if (m_children[i]->getName() == name && m_children[i]->is<T>())
        return m_children[i]->as<T>();
    }

    return nullptr;
  }

  template <typename T>
  AGX_FORCE_INLINE const T* SerializedFrame::Component::getChild(const char* name) const
  {
    return const_cast<SerializedFrame::Component *>(this)->getChild<T>(name);
  }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE size_t SerializedFrame::Buffer::getNumElements() const { return m_numElements; }
  // AGX_FORCE_INLINE size_t SerializedFrame::Buffer::capacity() const { return m_capacity; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE const Format* SerializedFrame::CustomBuffer::getFormat() const { return m_format; }
  AGX_FORCE_INLINE size_t SerializedFrame::CustomBuffer::getNumElements() const { return m_numElements; }
  // AGX_FORCE_INLINE size_t SerializedFrame::CustomBuffer::capacity() const { return m_capacity; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE const Format* SerializedFrame::PartialBuffer::getFormat() const { return m_format; }
  AGX_FORCE_INLINE size_t SerializedFrame::PartialBuffer::getNumElements() const { return m_numElements; }
  AGX_FORCE_INLINE size_t SerializedFrame::PartialBuffer::fullSize() const { return m_fullSize; }
  // AGX_FORCE_INLINE size_t SerializedFrame::PartialBuffer::fullCapacity() const { return m_fullCapacity; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE EntityModel* SerializedFrame::EntityStorage::getModel() { return m_model; }
  AGX_FORCE_INLINE const EntityModel* SerializedFrame::EntityStorage::getModel() const { return m_model; }
  AGX_FORCE_INLINE size_t SerializedFrame::EntityStorage::getNumElements() const { return m_numElements; }
  AGX_FORCE_INLINE size_t SerializedFrame::EntityStorage::capacity() const { return m_capacity; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE void SerializedFrame::StorageBatchEvent::add(agx::UInt64 id) { m_ids.push_back(id); }
  DOXYGEN_END_INTERNAL_BLOCK()

}


#endif /* AGXDATA_SERIALIZEDDATA_H */
