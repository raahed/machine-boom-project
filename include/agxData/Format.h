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

#ifndef AGXDATA_FORMAT_H
#define AGXDATA_FORMAT_H

#include <typeinfo>

#include <agx/agx.h>
#include <agx/agxCore_export.h>
#include <agx/String.h>
#include <agx/Model.h>
#include <agx/Name.h>
#include <iostream>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100) // : warning C4100: 'buffer' : unreferenced formal parameter
#endif


namespace agxData
{
  class Value;

  AGX_DECLARE_POINTER_TYPES(Format);

  class Type;
  typedef void (*TransformFunction)(void *target, const void *source, size_t numElements);

  // Used by the format-to-primitive type conversion system to let the user
  // call overloaded methods based on an abstract Format instance.
  class CallbackList;

  /** Register a format-format transformer */
  void AGXCORE_EXPORT registerTransformer(
    TransformFunction transformer, const Format* targetFormat, const Format* sourceFormat);

  /** Get a specific transformer */
  TransformFunction AGXCORE_EXPORT getTransformer(const Format* targetFormat, const Format* sourceFormat);

  /** Transform data between two buffers. */
  void AGXCORE_EXPORT transform(
    void* target, const Format* targetFormat, const void* source, const Format* sourceFormat, size_t numElements);


  /**
  A format is an implementation of a agxData::Type.
  */
  class AGXCORE_EXPORT Format : public agx::Model
  {
  public:
    static agx::Model *ClassModel();

  public:
    /**
    \return The auto-format (dynamic typing).
    */
    static Format *Auto();

    /**
    \return The void-format.
    */
    static Format *Void();

  public:

    Format(const agx::Name& formatName, const agx::Name& implementationName, size_t size, TransformFunction byteSwapFunction, TransformFunction copyFunction);

    /**
    \return The type which the format implements.
    */
    Type *getType();
    const Type *getType() const;

    /**
    \return The name of the format.
    */
    // const agx::Name& getName() const;

    /**
    \return Combined type:format name
    */
    agx::String fullName() const;

    /**
    \return The C++ language type name
    */
    const agx::Name& getImplementationName() const;

    /**
    \return The size in bytes of the format.
    */
    size_t getSize() const;

    /**
    \return The default format value.
    */
    agxData::Value *getDefaultValue();
    const agxData::Value *getDefaultValue() const;

    /**
    Swap byte order of elements in a buffer.
    */
    void swapByteOrder(void *target, const void *source, size_t numElements) const;

    /**
    Initialize elements.
    */
    virtual void initializeElement(void *element, const void *value) const;
    virtual void initializeElements(void *buffer, const void *value, size_t numElements) const;

    /**
    Construct elements.
    */
    virtual void constructElement(void *element) const;
    virtual void constructElements(void *buffer, size_t numElements) const;

    /**
    Destroy elements.
    */
    virtual void destroyElement(void *element) const;
    virtual void destroyElements(void *buffer, size_t numElements) const;

    /**
    Copy elements.
    */
    virtual void copyElement(void *target, const void *source) const;
    virtual void copyElements(void *buffer, const void *source, size_t numElements) const;

    /**
    Move elements.
    */
    virtual void moveElement(void *target, void *source) const;
    virtual void moveElements(void *target, void *source, size_t numElements) const;

    /**
    Permute elements.
    */
    virtual void permuteElements(void *target, void *source, const agx::Index *permutation, size_t numElements) const;

    /**
    Swap elements.
    */
    virtual void swapElements(void *element1, void *element2) const;

  public:
    typedef agx::HashTable<Format *, TransformFunction> TransformerTable;
    const TransformerTable& getTransformers() const;



    /**
    Setup a link between a full format name and it's C++ type representation.
    Should only be called by format initializer code.
    */
    template<typename T>
    static void registerFormatPrimitives( const agxData::Format* format, agx::UInt32 stride, agx::UInt32 padding );

    /** Call one of the callbacks in the callbackList based on the primitive type of the given format. */
    static bool callFormatPrimitive( const Format* format, CallbackList* callbackList );

  protected:
    virtual ~Format();

    void cleanup();

    friend class Type;
    void setType(Type *type);

    friend class TypeManager;
  private:
    Type *m_type;
    size_t m_size;
    // agx::Name m_name;
    agx::Name m_implementationName;
    TransformFunction m_swapFunction;
    agxData::Value *m_defaultValue;

    friend void AGXCORE_EXPORT registerTransformer(
      TransformFunction transformer, const Format* targetFormat, const Format* sourceFormat);
    friend TransformFunction AGXCORE_EXPORT getTransformer(const Format* targetFormat, const Format* sourceFormat);
    TransformerTable m_transformerTable;
  };

  /**
  Container format, eg. Array, Vector
  */
  class AGXCORE_EXPORT ContainerFormat : public Format
  {
  public:
    ContainerFormat(const agx::Name& containerName, Format *elementFormat, size_t size);

    Format *getElementFormat();
    const Format *getElementFormat() const;

  protected:
    virtual ~ContainerFormat();

    using Format::copyElement;
    using Format::moveElement;

    template <typename T>
    void copyElement(T& target, const T& source) const;

    template <typename T>
    void moveElement(T& target, T& source) const;


  private:
    FormatRef m_elementFormat;
  };



  class AGXCORE_EXPORT ArrayFormat : public ContainerFormat
  {
  public:
    ArrayFormat(Format *elementFormat);

    virtual void initializeElement(void *element, const void *value) const override;
    virtual void initializeElements(void *buffer, const void *value, size_t numElements) const override;
    virtual void constructElement(void *element) const override;
    virtual void constructElements(void *buffer, size_t numElements) const override;
    virtual void destroyElement(void *element) const override;
    virtual void destroyElements(void *buffer, size_t numElements) const override;
    virtual void copyElement(void *target, const void *source) const override;
    virtual void copyElements(void *target, const void *source, size_t numElements) const override;
    virtual void moveElement(void *target, void *source) const override;
    virtual void moveElements(void *target, void *source, size_t numElements) const override;
    virtual void permuteElements(void *target, void *source, const agx::Index *permutation, size_t numElements) const override;
    virtual void swapElements(void *element1, void *element2) const override;

  protected:
    virtual ~ArrayFormat();

  private:
  };


#if 0
  class AGXCORE_EXPORT VectorFormat : public ContainerFormat
  {
  public:
    VectorFormat(Format *elementFormat);

    virtual void initializeElement(void *element, const void *value) const override;
    virtual void initializeElements(void *buffer, const void *value, size_t numElements) const override;
    virtual void constructElement(void *element) const override;
    virtual void constructElements(void *buffer, size_t numElements) const override;
    virtual void destroyElement(void *element) const override;
    virtual void destroyElements(void *buffer, size_t numElements) const override;
    virtual void copyElement(void *target, const void *source) const override;
    virtual void copyElements(void *target, const void *source, size_t numElements) const override;
    virtual void moveElement(void *target, void *source) const override;
    virtual void moveElements(void *target, void *source, size_t numElements) const override;
    virtual void swapElements(void *element1, void *element2) const override;

  protected:
    virtual ~VectorFormat();

  private:
  };

  class AGXCORE_EXPORT HashSetFormat : public ContainerFormat
  {
  public:
    HashSetFormat(Format *elementFormat);

    virtual void initializeElement(void *element, const void *value) const override;
    virtual void initializeElements(void *buffer, const void *value, size_t numElements) const override;
    virtual void constructElement(void *element) const override;
    virtual void constructElements(void *buffer, size_t numElements) const override;
    virtual void destroyElement(void *element) const override;
    virtual void destroyElements(void *buffer, size_t numElements) const override;
    virtual void copyElement(void *target, const void *source) const override;
    virtual void copyElements(void *target, const void *source, size_t numElements) const override;
    virtual void moveElement(void *target, void *source) const override;
    virtual void moveElements(void *target, void *source, size_t numElements) const override;
    virtual void swapElements(void *element1, void *element2) const override;

  protected:
    virtual ~HashSetFormat();

  private:
  };

#endif


  class AGXCORE_EXPORT GenericFormat : public Format
  {
  public:
    GenericFormat(size_t size);

    virtual void initializeElement(void *element, const void *value) const override;
    virtual void initializeElements(void *buffer, const void *value, size_t numElements) const override;
    virtual void constructElement(void *element) const override;
    virtual void constructElements(void *buffer, size_t numElements) const override;
    virtual void destroyElement(void *element) const override;
    virtual void destroyElements(void *buffer, size_t numElements) const override;
    virtual void copyElement(void *target, const void *source) const override;
    virtual void copyElements(void *target, const void *source, size_t numElements) const override;
    virtual void moveElement(void *target, void *source) const override;
    virtual void moveElements(void *target, void *source, size_t numElements) const override;
    virtual void swapElements(void *element1, void *element2) const override;

  protected:
    virtual ~GenericFormat();

  };

  /**
  Templated format, connects the type abstractions to the programming language.
  */
  template <typename T>
  class FormatT : public Format
  {
  public:
    typedef T Type;

    FormatT(const agx::Name& formatName, TransformFunction byteSwapFunction = nullptr);

    virtual void initializeElement(void *element, const void *value) const override;
    virtual void initializeElements(void *buffer, const void *value, size_t numElements) const override;
    virtual void constructElement(void *element) const override;
    virtual void constructElements(void *buffer, size_t numElements) const override;
    virtual void destroyElement(void *element) const override;
    virtual void destroyElements(void *buffer, size_t numElements) const override;
    virtual void copyElement(void *target, const void *source) const override;
    virtual void copyElements(void *target, const void *source, size_t numElements) const override;
    virtual void moveElement(void *target, void *source) const override;
    virtual void moveElements(void *target, void *source, size_t numElements) const override;
    virtual void permuteElements(void *target, void *source, const agx::Index *permutation, size_t numElements) const override;


    virtual void swapElements(void *element1, void *element2) const override;

  protected:
    virtual ~FormatT();

  private:
    // T m_defaultValue;
  };


  // Used to register formats
  class AGXCORE_EXPORT FormatInitializer
  {
  public:
    typedef void (*FormatConstructorFn)();

    FormatInitializer(const char *name, FormatConstructorFn constructor);

    static void createFormats();
    static void removeFormats();

    void init();

    const char *getName() const;

  private:
    const char *m_name;
    FormatConstructorFn m_constructor;
    bool m_isInitialized;
  };


  template <size_t NUM_BYTES>
  struct GenericStruct
  {
    char elements[NUM_BYTES];
  };






  /**
  List of callbacks used by the primitive type handling by user code. Supply a
  format and an instance of the CallbackList to Format::callFormatPrimitive(.)
  and the correct callback from the list will be called based on the primitive
  type of the format.
  */
  class AGXCORE_EXPORT CallbackList
  {
  public:
    typedef agx::Callback2<agx::UInt32, agx::UInt32> CallbackType;
    CallbackList( CallbackType callbackReal32, CallbackType callbackReal64,
                  CallbackType callbackInt8,   CallbackType callbackUInt8,
                  CallbackType callbackInt16,  CallbackType callbackUInt16,
                  CallbackType callbackInt32,  CallbackType callbackUInt32,
                  CallbackType callbackInt64,  CallbackType callbackUInt64,
                  CallbackType callbackBool);

    void callReal32( agx::UInt32 stride, agx::UInt32 padding ) { m_callbackReal32( stride, padding ); }
    void callReal64( agx::UInt32 stride, agx::UInt32 padding ) { m_callbackReal64( stride, padding ); }
    void callInt8(   agx::UInt32 stride, agx::UInt32 padding ) { m_callbackInt8(   stride, padding ); }
    void callUInt8(  agx::UInt32 stride, agx::UInt32 padding ) { m_callbackUInt8(  stride, padding ); }
    void callInt16(  agx::UInt32 stride, agx::UInt32 padding ) { m_callbackInt16(  stride, padding ); }
    void callUInt16( agx::UInt32 stride, agx::UInt32 padding ) { m_callbackUInt16( stride, padding ); }
    void callInt32(  agx::UInt32 stride, agx::UInt32 padding ) { m_callbackInt32(  stride, padding ); }
    void callUInt32( agx::UInt32 stride, agx::UInt32 padding ) { m_callbackUInt32( stride, padding ); }
    void callInt64(  agx::UInt32 stride, agx::UInt32 padding ) { m_callbackInt64(  stride, padding ); }
    void callUInt64( agx::UInt32 stride, agx::UInt32 padding ) { m_callbackUInt64( stride, padding ); }
    void callBool(   agx::UInt32 stride, agx::UInt32 padding ) { m_callbackBool(   stride, padding ); }

    bool hasReal32() const { return m_callbackReal32.isValid(); }
    bool hasReal64() const { return m_callbackReal64.isValid(); }
    bool hasInt8()   const { return m_callbackInt8.isValid();   }
    bool hasUInt8()  const { return m_callbackUInt8.isValid();  }
    bool hasInt16()  const { return m_callbackInt16.isValid();  }
    bool hasUInt16() const { return m_callbackUInt16.isValid(); }
    bool hasInt32()  const { return m_callbackInt32.isValid();  }
    bool hasUInt32() const { return m_callbackUInt32.isValid(); }
    bool hasInt64()  const { return m_callbackInt64.isValid();  }
    bool hasUInt64() const { return m_callbackUInt64.isValid(); }
    bool hasBool()   const { return m_callbackBool.isValid();   }

  private:
    CallbackType m_callbackReal32;
    CallbackType m_callbackReal64;
    CallbackType m_callbackInt8;
    CallbackType m_callbackUInt8;
    CallbackType m_callbackInt16;
    CallbackType m_callbackUInt16;
    CallbackType m_callbackInt32;
    CallbackType m_callbackUInt32;
    CallbackType m_callbackInt64;
    CallbackType m_callbackUInt64;
    CallbackType m_callbackBool;
  };


  /*
  The primitive type callback system uses a template based dispatch mechanism
  that calls the correct callback in a Callback list, passing on the proper
  'primitivesPerElement' argument. Template specializations handles the
  calling of the correct callback function.
  */
  template< typename T>
  bool callDispatch( CallbackList* /*callbackList*/, agx::UInt32 /*stride*/, agx::UInt32 /*padding*/ )
  {
    // Should anything be done here? We have neither LOGGER nor agxData::getFormat<T>() due to include loops.
    std::cout << "Format: Hit typeless callDispatch() for C++ type " << typeid(T).name() << std::endl;
    return false;
  }

  // Template specializations for the primitive types we currently support.

  template<> bool callDispatch<agx::Real32>(  CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::Real64>(  CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::Int8>  ( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::UInt8> ( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::Int16> ( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::UInt16>( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::Int32> ( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::UInt32>( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::Int64> ( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::UInt64>( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );
  template<> bool callDispatch<agx::Bool>  ( CallbackList* callbackList, agx::UInt32 stride, agx::UInt32 padding );


  /*
  Class hierarchy that handles the calling of the correct dispatch function.
  There is a table that maps from a format name to an instance of one of these
  FormatCallers.
  */
  class FormatCaller
  {
  public:
    virtual bool call( CallbackList* callbackList ) = 0;
    virtual ~FormatCaller() {}
  };

  template<typename T>
  class FormatCallerT : public FormatCaller
  {
  public:
    typedef T Type;

    FormatCallerT( agx::UInt32 stride, agx::UInt32 padding ) : m_stride(stride), m_padding(padding) {}
    virtual bool call( CallbackList* callbackList )
    {
      return callDispatch<T>( callbackList, m_stride, m_padding );
    }

    virtual ~FormatCallerT() {}
  private:
    agx::UInt32 m_stride;
    agx::UInt32 m_padding;
  };



  DOXYGEN_START_INTERNAL_BLOCK()
  /**
  Clear the contents of the __agx_typeCallerTable table. This is done during
  shutdown. Needed because Generator.exe doesn't use the regular setup/teardown
  framework.
  */
  void AGXCORE_EXPORT forceClearTypeCallerTable();
  DOXYGEN_END_INTERNAL_BLOCK()

  // Table holding mappings from a full format name to a FormatCaller with the correct template type.
  typedef agx::HashTable<const agxData::Format*, FormatCaller*> TypeCallerTable;
  AGXCORE_EXPORT extern TypeCallerTable __agx_typeCallerTable;


  /*
  Bind a FormatCaller to a format. Each format can only be registered once.
  \param format The high level format that we want a primitive type mapping for.
  \param stride The number of primitives per object of the format.
  \param padding The number of primitives at the end of each object that are padding.
  */
  template<typename T>
  void Format::registerFormatPrimitives(const Format* format, agx::UInt32 stride, agx::UInt32 padding )
  {
    size_t elementSize = sizeof(T) * stride;
    agxVerifyN(elementSize == format->getSize(), "Size mismatch, format %s is %u bytes, packed element is %u bytes", format->fullName().c_str(), (unsigned)format->getSize(), (unsigned)elementSize);
    agxVerifyN( padding < stride, "Format: A format cannot be all padding, or contain more padding than total memory usage. "
      "stride: %d, padding: %d.", stride, padding );
    agxVerifyN( __agx_typeCallerTable.find( format ) == __agx_typeCallerTable.end(),
      "Format: The format \'%s\' have already been registered.", format->fullName().c_str() );
    __agx_typeCallerTable.insert( format, new FormatCallerT<T>(stride, padding) );
  }





  #define AGX_FORMAT_INITIALIZER(_Name)                                                                                               \
  static void AGX_CONCAT(__agx_createFormat, __LINE__)();                                                                             \
  static agxData::FormatInitializer AGX_CONCAT(__agx_formatInitializer, __LINE__)(_Name, &AGX_CONCAT(__agx_createFormat, __LINE__));  \
  static void AGX_CONCAT(__agx_createFormat, __LINE__)()


  /** Standard swap functions */
  void AGXCORE_EXPORT Swap8Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap16Function(void *target, const void *source, size_t numElements);

  void AGXCORE_EXPORT Swap32Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap2x32Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap3x32Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap4x32Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap6x32Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap8x32Function(void *target, const void *source, size_t numElements);

  void AGXCORE_EXPORT Swap64Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap2x64Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap3x64Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap4x64Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap6x64Function(void *target, const void *source, size_t numElements);
  void AGXCORE_EXPORT Swap8x64Function(void *target, const void *source, size_t numElements);

#ifdef _MSC_VER
#  pragma warning(disable: 4800) // Disable warnings about performance (bool true or false)
#endif

  template <typename TargetT, typename SourceT>
  void GenericTransformFunction(void *targetBuffer, const void *sourceBuffer, size_t numElements)
  {
    for (size_t i = 0; i < numElements; ++i)
      ((TargetT *)targetBuffer)[i] = (TargetT)((SourceT *)sourceBuffer)[i];
  }

  template <typename T>
  void GenericCopyFunction(void *target, const void *source, size_t numElements)
  {
    ::memcpy(target, source, numElements * sizeof(T));
  }


  /* Implementation */
  AGX_FORCE_INLINE Type *Format::getType() { return m_type; }
  AGX_FORCE_INLINE const Type *Format::getType() const { return m_type; }
  // AGX_FORCE_INLINE const agx::Name& Format::getName() const { return m_name; }
  AGX_FORCE_INLINE const agx::Name& Format::getImplementationName() const { return m_implementationName; }
  AGX_FORCE_INLINE size_t Format::getSize() const { return m_size; }
  AGX_FORCE_INLINE const Format::TransformerTable& Format::getTransformers() const { return m_transformerTable; }


  //-----------------------------------------------------------------------------------------------------

  AGX_FORCE_INLINE Format *ContainerFormat::getElementFormat() { return m_elementFormat; }
  AGX_FORCE_INLINE const Format *ContainerFormat::getElementFormat() const { return m_elementFormat; }

  //-----------------------------------------------------------------------------------------------------

  template <class T, typename std::enable_if<std::is_copy_constructible<T>::value>::type* = nullptr >
  inline void FormatTdoInitializeElement( T* element, const T* value )
  {
    ::new((T *)element) T(*(const T *)value);
  }

  template <class T, typename std::enable_if<!std::is_copy_constructible<T>::value>::type* = nullptr >
  inline void FormatTdoInitializeElement( T* element, const T* )
  {
    // No copy-constructor, default constructor instead.
    ::new((T *)element) T();
  }

  //-----------------------------------------------------------------------------------------------------
  template <typename T>
  FormatT<T>::FormatT(const agx::Name& formatName, TransformFunction byteSwapFunction) : Format(formatName, typeid(T).name(), sizeof(T), byteSwapFunction, GenericCopyFunction<T>)
  {}

  template <typename T>
  FormatT<T>::~FormatT()
  {
    this->cleanup();
  }



  template <typename T>
  void FormatT<T>::initializeElement(void *element, const void *value) const
  {
    // ::new((T *)element) T(*(const T *)value);
    FormatTdoInitializeElement( (T*) element, (const T*) value );

    #if 0
    // Use GenericStruct to copy raw data, discarding custom assignment operator semantics
    typedef GenericStruct<sizeof(T)> SizeStruct;
    *(SizeStruct *)element = *(const SizeStruct *)value;
    #endif
  }

  template <typename T>
  void FormatT<T>::initializeElements(void *buffer, const void *value, size_t numElements) const
  {
    for (size_t i = 0; i < numElements; ++i) {

      // ::new(&((T *)buffer)[i]) T(*(const T *)value);
      FormatTdoInitializeElement( &((T *)buffer)[i], (const T*) value );
    }

    #if 0
    // Use GenericStruct to copy raw data, discarding custom assignment operator semantics
    typedef GenericStruct<sizeof(T)> SizeStruct;

    for (size_t i = 0; i < numElements; ++i)
      ((SizeStruct *)buffer)[i] = *(const SizeStruct *)value;
    #endif
  }


  template <typename T>
  void FormatT<T>::constructElement(void *element) const
  {
    ::new((T *)element) T();
  }


  template <typename T>
  void FormatT<T>::constructElements(void *buffer, size_t numElements) const
  {
    for (size_t i = 0; i < numElements; ++i)
      ::new(&((T *)buffer)[i]) T();
  }

  template <typename T>
  void FormatT<T>::destroyElement(void *element) const
  {
    ((T *)element)->~T();

    #ifdef AGX_DEBUG
    ::memset(element, 0xFF, sizeof(T));
    #endif
  }

  template <typename T>
  void FormatT<T>::destroyElements(void *buffer, size_t numElements) const
  {
    for (size_t i = 0; i < numElements; ++i)
      ((T *)buffer)[i].~T();

    #ifdef AGX_DEBUG
    if (buffer != nullptr) {
      ::memset(buffer, 0xFF, numElements * sizeof(T));
    }
    #endif
  }


  template <typename T>
  void FormatT<T>::copyElement(void *target, const void *source) const
  {
    *(T *)target = *(const T *)source;
  }

  template <typename T>
  void FormatT<T>::copyElements(void *target, const void *source, size_t numElements) const
  {
    for (size_t i = 0; i < numElements; ++i)
      ((T *)target)[i] = ((const T *)source)[i];
  }



  template <typename T>
  struct FormatMoveSwap
  {
    AGX_FORCE_INLINE static void move(T& target, T& source)
    {
      target = source;
    }

    AGX_FORCE_INLINE static void swap(T& a, T& b)
    {
      std::swap(a, b);
    }
  };


  template <typename T>
  struct FormatMoveSwapContainer
  {
    typedef GenericStruct<sizeof(T)> CopyElement;

    AGX_FORCE_INLINE static void move(T& target, T& source)
    {
      target.~T();

      // Redirect target vector to existing element buffer (avoids copying the whole element buffer)
      (CopyElement&)target = (CopyElement&)source;

      // Clear old vector header
      source._setBuffer(nullptr);
      source._setCapacity(0);
      source._setSize(0);
    }

    AGX_FORCE_INLINE static void swap(T& a, T& b)
    {
      std::swap(*(CopyElement *)&a, *(CopyElement *)&b);
    }

  };

  template <typename T>
  struct FormatMoveSwap< agx::Vector<T> >
  {
    typedef agx::Vector<T> VectorT;

    AGX_FORCE_INLINE static void move(VectorT& target, VectorT& source) { FormatMoveSwapContainer<VectorT>::move(target, source); }
    AGX_FORCE_INLINE static void swap(VectorT& a, VectorT& b) { FormatMoveSwapContainer<VectorT>::swap(a, b); }
  };

  template <typename T>
  struct FormatMoveSwap< agx::HashSet<T> >
  {
    typedef agx::HashSet<T> HashT;

    AGX_FORCE_INLINE static void move(HashT& target, HashT& source) { FormatMoveSwapContainer<HashT>::move(target, source); }
    AGX_FORCE_INLINE static void swap(HashT& a, HashT& b) { FormatMoveSwapContainer<HashT>::swap(a, b); }
  };


  template <typename KeyT, typename DataT>
  struct FormatMoveSwap< agx::HashTable<KeyT, DataT> >
  {
    typedef agx::HashTable<KeyT, DataT> HashT;

    AGX_FORCE_INLINE static void move(HashT& target, HashT& source) { FormatMoveSwapContainer<HashT>::move(target, source); }
    AGX_FORCE_INLINE static void swap(HashT& a, HashT& b) { FormatMoveSwapContainer<HashT>::swap(a, b); }
  };


  template <typename T>
  void FormatT<T>::moveElement(void *target, void *source) const
  {
    FormatMoveSwap<T>::move(*(T *)target, *(T *)source);
  }

  template <typename T>
  void FormatT<T>::moveElements(void *target, void *source, size_t numElements) const
  {
    for (size_t i = 0; i < numElements; ++i)
      FormatMoveSwap<T>::move(((T *)target)[i], ((T *)source)[i]);
  }

  template <typename T>
  void FormatT<T>::permuteElements(void *target, void *source, const agx::Index *permutation, size_t numElements) const
  {
    for (size_t i = 0; i < numElements; ++i)
      FormatMoveSwap<T>::move(((T *)target)[i], ((T *)source)[permutation[i]]);
  }


  template <typename T>
  void FormatT<T>::swapElements(void *element1, void *element2) const
  {
    FormatMoveSwap<T>::swap(*(T *)element1, *(T *)element2);
  }

}

#if defined(_MSC_VER)
  #pragma warning( pop ) // restoring: warning( disable : 4100 )
#endif

#endif /* _AGXDATA_FORMAT_H_ */
