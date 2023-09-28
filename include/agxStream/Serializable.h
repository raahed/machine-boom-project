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

#include <agx/agxCore_export.h>

#include <agx/Integer.h>
#include <functional>
#include <agxStream/StorageAgent.h>
#include <agx/Uuid.h>

namespace agxStream
{
  class InputArchive;
  class OutputArchive;
  class StorageAgent;
  class StorageStream;

  /**
  This class is an abstract base class for all classes that can be stored and retrieved from
  an Archive. Derive from this class and implement The two methods:

  store - Stream an object to an archive
  restore - Restore an object from an archive

  The two above methods are declared in the AGXSTREAM_DECLARE_SERIALIZABLE which must be part of
  any class that should be possible to serialize.

  Also remember to create a storage object using the \p Storage struct: Storage<MyClass> myClassStorage.
  */
  class AGXCORE_EXPORT Serializable
  {
    public:

#if !defined(SWIG)
      /// Destructor for normal C++ use but hidden from SWIG bindings
      virtual ~Serializable();
#endif

      /**
      Tells this class that it is restored correctly and should not be deleted during destruction of an Archive.
      */
      void setFinished();

      /**
      \return true if this class is restored correctly and should not be deleted during destruction of an Archive.
      */
      bool isFinished() const;

      /// \return a unique id for this instance.
      agx::Uuid getUuid() const;

      /**
      Explicitly set a Uuid on a serializable object.
      Use this method with great care, as it can result in non-unique identifiers in the system!
      */
      void setUuid( const agx::Uuid& uuid );

      /**
      This index is given at creation of this object. It is guaranteed to be unique for the lifetime of this library instance.
      As long as the application is running. This value is not serialized or restored. A new object will always get a new index no matter
      how it is created.

      \return an instance index of this object. This is unique for the lifetime of this library instance.
      */
      agx::UInt32 getIndex() const;

      /**
      Set to false to disable serialization of this object. It will just be ignored during serialization (writing).
      It is still possible to restore TO this object.
      */
      void setEnableSerialization( bool flag );

      /**
      \return true if serialization is enabled for this object
      */
      bool getEnableSerialization( ) const;

#ifndef SWIG
      /**
      \return the storage agent
      */
      virtual StorageAgent* getStorageAgent() const = 0;

      /// \return the class name of this class.
      virtual const char* getClassName() const;
#endif // SWIG

      /// Specify if there should be UUID:s generated for each new  Serializable object. By default it is enabled
      static void setEnableUuidGeneration( bool flag );

      /// \return true if Uuid:s should be generated for each new Serializable object
      bool getEnableUuidGeneration( );

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

#ifndef SWIG
      /**
      Store non-structural data to stream. Not supported for all serializable objects.
      */
      virtual void storeLightData( agxStream::StorageStream& ) const {}

      /**
      Restore non-structural data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& ) {}

      bool _store( agxStream::OutputArchive& out ) const;
      void _restore( agxStream::InputArchive& in );
      DOXYGEN_END_INTERNAL_BLOCK()
#endif
    protected:

      /**
      Mock-declaration of destructor for SWIG to fix an issue occurring when Serializable is part of multiple inheritence
      together with another abstract base class whose destructor which is declared as protected such as agx::Referenced
      */
#if defined(SWIG)
      virtual ~Serializable();
#endif

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      InputArchive uses these methods to keep track of which
      objects it dynamically created, in order to delete them
      if un-serialization fails.
      */
      void setDynamicallyAllocated();

      bool isDynamicallyAllocated() const;


#ifndef SWIG
      virtual void store( agxStream::OutputArchive& out ) const = 0;
      virtual void restore( agxStream::InputArchive& in ) = 0;
#endif

      friend class InputArchive;
      friend class OutputArchive;

      /// Set the current input archive
      void setArchive( InputArchive* archive );

      /// \return the associated input archive
      InputArchive* getArchive( );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      int m_flags;
      static StorageAgent* s_storageAgent;

      enum {
        FINISHED = 1,
        DYNAMICALLY_ALLOCATED = 2
      };

    protected:
      /// Default constructor
      Serializable();

      void generateUuid();

      /// Copy constructor
      Serializable(const Serializable& other);

    private:
      InputArchive* m_archive;
      agx::Uuid m_uuid;
      agx::UInt32 m_index;
      bool m_serializationEnabled;
      static bool s_uuidGeneration_enabled;
  };
#ifndef SWIG
/// Macro for retrieving a class name based on the function scope
#define AGXSTREAM_CLASS_NAME(T) \
  /** \return the class name of this class.  */ \
  inline const char* getClassName() const override { return #T; } \


#define AGXSTREAM_DECLARE_SERIALIZABLE_BASE( T ) \
  inline agxStream::StorageAgent* getStorageAgent() const override { return agxStream::StorageManager::instance()->find( #T );  } \
  inline static const char* getConstructClassId() { return #T; } \
  friend class agxStream::DefStorageAgent<T>; \
  AGXSTREAM_CLASS_NAME(T) \
  static agxStream::Serializable *create() { return new T(); } \
  virtual void store( agxStream::OutputArchive& out ) const override;\
  virtual void restore( agxStream::InputArchive& in ) override

/** Use this in a Serializable class to add the required methods
    Important: Use full namespace in the declaration! For example: agx::deprecated::SomeClass
*/
#define AGXSTREAM_DECLARE_SERIALIZABLE( T ) \
  AGXSTREAM_DECLARE_SERIALIZABLE_BASE(T); \
  static agxStream::Serializable *create(agxStream::InputArchive& /*in*/) { return new T(); }

/** Use this in a Serializable class to add the required methods
    Important: Use full namespace in the declaration! For example: agx::deprecated::SomeClass
*/
#define AGXSTREAM_DECLARE_SERIALIZABLE_CUSTOM_CREATE( T ) \
  AGXSTREAM_DECLARE_SERIALIZABLE_BASE(T); \
  static agxStream::Serializable *create(agxStream::InputArchive& in)

/** Use this in a pure abstract Serializable class to add the required methods
    Important: Use full namespace in the declaration! For example: agx::deprecated::SomeClass
*/
#define AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( T ) \
  inline agxStream::StorageAgent* getStorageAgent() const override { return agxStream::StorageManager::instance()->find(#T); } \
  inline static const char* getConstructClassId() { return  #T; } \
  AGXSTREAM_CLASS_NAME(T)

/// Use this macro to instantiate a storage for a Serializable class. If you have '::' in the class name, you need to use the AGXSTREAM_INSTANTIATE_STORAGE_VARIABLE macro
#define AGXSTREAM_INSTANTIATE_STORAGE( C ) \
  static agxStream::Storage<C> storage ## C;

/// Use this macro for instantiate a storage as a specific variable name if necessary. For example if C contains :: the previous macro cannot be used.
#define AGXSTREAM_INSTANTIATE_STORAGE_VARIABLE( V,C ) \
  static agxStream::Storage<C> storage ## V;

 // Macro for delaying expansion of comma (,) in a macro argument. This allows for submitting arguments such as: "first", "second" etc..
#define AGXARCHIVE_MODIFICATIONS(...) {__VA_ARGS__}


/** This macro will instantiate a storage for a serializable class
  V - the name of the variable that will be instantiated
  DEFAULT_CLASS - The new implementation of a class with the same name as the old one, for example agxDriveTrain::TorqueConverter
  DEPRECATED_CLASS - The old implementation, now with a different namespace, for example agxDriveTrain::deprecated::TorqueConverter
  MODIFICATION_LIST - List of modifications that if they are all present in the input archive, the default class will be used.
*/
#define AGXSTREAM_INSTANTIATE_STORAGE_DEPRECATION_CHECK_VARIABLE( V, DEFAULT_CLASS, DEPRECATED_CLASS, MODIFICATION_LIST ) \
  static agxStream::DeprecatedStorage<DEFAULT_CLASS, DEPRECATED_CLASS> V(MODIFICATION_LIST);


  AGX_FORCE_INLINE agx::UInt32 Serializable::getIndex() const
  {
    return m_index;
  }

  AGX_FORCE_INLINE bool Serializable::isFinished() const
  {
    return (m_flags & FINISHED) != 0;
  }

  AGX_FORCE_INLINE bool Serializable::isDynamicallyAllocated() const
  {
    return (m_flags & DYNAMICALLY_ALLOCATED) != 0;
  }


  AGX_FORCE_INLINE InputArchive* Serializable::getArchive()
  {
    return m_archive;
  }

  AGX_FORCE_INLINE agx::Uuid Serializable::getUuid() const
  {
    return m_uuid;
  }

  AGX_FORCE_INLINE bool Serializable::getEnableSerialization() const
  {
    return m_serializationEnabled;
  }

#else

  // Prevent swig from seeing macro usage of hidden macros
  #define AGXSTREAM_DECLARE_SERIALIZABLE( T )
  #define AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( T )
  #define AGXSTREAM_DECLARE_SERIALIZABLE_CUSTOM_CREATE( T )

#endif // SWIG
} // Namespace agxStream
