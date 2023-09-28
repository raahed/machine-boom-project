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

#ifndef AGX_REFERENCED_H
#define AGX_REFERENCED_H

#include <agx/config/AGX_DEBUG_REFERENCED.h>
#include <agx/config/AGX_DEBUG_TRACK_REFERENCED_OBJECTS.h>

#include <agx/ref_ptr.h>
#include <agx/observer_ptr.h>
#include <agx/agxCore_export.h>
#include <agx/AtomicValue.h>
#include <agx/ScopeLock.h>

#include <mutex>

#if AGX_DEBUG_REFERENCED()
#  include <map>
#  include <agx/StackWalker.h>
#endif


#if AGX_DEBUG_TRACK_REFERENCED_OBJECTS()
# include <agx/ReferencedManager.h>
#endif


#define OBSERVER_CONTAINER_TYPE 1 // 1: Vector< Observer* > pointer (new/delete)
                                  // 2: HashSet< Observer* > pointer (new/delete)
                                  // 3: Vector< Observer* > member on stack

#if OBSERVER_CONTAINER_TYPE == 1
#include <agx/Vector.h>
#define ObserverContainer VectorPOD< Observer* >*
#define INSERT_OBSERVER( obs ) \
  if ( !m_observers->contains( obs ) ) m_observers->push_back( obs )
#define REMOVE_OBSERVER( obs ) \
  if ( m_observers ) m_observers->findAndErase( obs )
#define REPORT_REFERENCED_DELETED \
  if ( m_observers ) { \
    for ( size_t i = 0, numObservers = m_observers->size(); i < numObservers; ++i ) \
      (*m_observers)[ i ]->objectDeleted( this ); \
    delete m_observers; \
    m_observers = nullptr; \
  }
#elif OBSERVER_CONTAINER_TYPE == 2
#include <agx/HashSet.h>
#define ObserverContainer HashSet< Observer* >*
#define INSERT_OBSERVER( obs ) \
  if ( m_observers == nullptr ) m_observers = new HashSet< Observer* >; \
  m_observers->insert( obs )
#define REMOVE_OBSERVER( obs ) \
  if ( m_observers ) m_observers->erase( obs )
#define REPORT_REFERENCED_DELETED \
  if ( m_observers ) { \
    for ( HashSet< Observer* >::iterator i = m_observers->begin(), end = m_observers->end(); i != end; ++i ) \
      (*i)->objectDeleted( this ); \
    delete m_observers; \
    m_observers = nullptr; \
  }
#elif OBSERVER_CONTAINER_TYPE == 3
#include <agx/Vector.h>
#define ObserverContainer Vector< Observer* >
#define INSERT_OBSERVER( obs ) \
  if ( !m_observers.contains( obs ) ) m_observers.push_back( obs )
#define REMOVE_OBSERVER( obs ) \
  m_observers.findAndErase( obs )
#define REPORT_REFERENCED_DELETED \
  for ( size_t i = 0, numObservers = m_observers.size(); i < numObservers; ++i ) \
    m_observers[ i ]->objectDeleted( this );
#endif

namespace agx
{
  // forward declare, declared after Referenced below.
  class DeleteHandler;
  class Observer;

  using Mutex = std::mutex;

#define AGX_EXPIRED_SCOPE_ALLOCATION (-0xABC)

  /**
  Base class providing referencing counted objects.
  Derive from this class when you need automatic reference counting for objects.
  Always keep the destructor of your derived class protected. One must never create a reference counted
  object on the stack. This will lead to corrupted memory if any reference pointers try to delete it.

  \code
  // when ptr goes out of scope, the reference to the instance of YourClass will be
  // decremented. If reference count is 0, it will be deleted.
  ref_ptr<YourClass> ptr = new YourClass();

  // Keep an observer pointer to the allocated object. When the instance of YourClass is deleted,
  // the oPtr.valid() returns true.
  observer_ptr<YourClass> oPtr = ptr.get();
  \endcode
  */
  class AGXCORE_EXPORT Referenced
  {

    public:

      /// Default constructor
      Referenced();

      //explicit Referenced(/*bool threadSafeRefUnref*/);
#ifndef SWIG
      Referenced(const Referenced&);
#endif

      /// Assignment operator. Will increment the number of references to the referenced object.
      Referenced& operator = (const Referenced&);

      /**
      Explicitly increment the reference count by one, indicating that
      this object has another pointer which is referencing it.
      This method should be used with care, always balanced with the same number of calls to unreference.
      */
      void reference(void* ptr = nullptr) const;

      /**
      Decrement the reference count by one, indicating that
      a pointer to this object is referencing it.  If the
      reference count goes to zero, it is assumed that this object
      is no longer referenced and is automatically deleted.

      This method should be used with care, always balanced with the same number of calls to reference.
      */
      void unreference(void* ptr = nullptr) const;

      /**
      \return the number pointers currently referencing this object.
      */
      int getReferenceCount() const;

      /**
      Decrement the reference count by one, indicating that
      a pointer to this object is referencing it.  However, do
      not delete it, even if ref count goes to 0.  Warning, unref_nodelete()
      should only be called if the user knows exactly who will
      be responsible for, one should prefer unref() over unref_nodelete()
      as the later can lead to memory leaks.
      */
      void unreference_nodelete() const;

    public:

      DOXYGEN_START_INTERNAL_BLOCK()

      /// Internal: Explicitly set the number of references
      void setReferenceCount(int count) const;

      /**
      Internal: Add a Observer that observes this object, notify the Observer when this object gets deleted.
      */
      void addObserver( Observer* observer ) const;

      /**
      Internal: Add a Observer that observes this object, notify the Observer when this object gets deleted.
      */
      void removeObserver( Observer* observer ) const;

      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      Subclass test.
      */
      template <typename T>
      bool is() const;

      /**
      Subclass casting. Template type must match actual type, dynamic_cast assertion will fail otherwise.
      */
      template <typename T>
      T *as();

      template <typename T>
      const T *as() const;


      /**
      Safe subclass casting, return nullptr if template type does not match.
      */
      template <typename T>
      T *asSafe();

      template <typename T>
      const T *asSafe() const;

      friend class DeleteHandler;

#if AGX_DEBUG_REFERENCED()
      static bool getEnableStackTrace();
      static void setEnableStackTrace( bool flag );

      typedef std::map<void*, agx::StackTrace> StackMap;
      const StackMap& getStackMap() const;
#endif
    public:

      template <typename T>
      static bool ValidateCast(const Referenced *object);

    protected:
      /// Destructor
      virtual ~Referenced();

      /**
      Internal: Set a DeleteHandler to which deletion of all referenced counted objects
      will be delegated to.
      */
      static void setDeleteHandler(DeleteHandler* handler);

      /**
      \return a DeleteHandler.
      */
      static DeleteHandler* getDeleteHandler();

      void deleteUsingDeleteHandler() const;

      void allocateObserverVector() const;

    protected:
      mutable ObserverContainer      m_observers;
      mutable Mutex                  m_mutex;
      mutable AtomicValue            m_refCount;

#if AGX_DEBUG_REFERENCED()
      mutable StackMap m_stackMap;
#endif
  };

#define AGX_DECLARE_POINTER_TYPES(type)             \
  class type;                                         \
  typedef agx::ref_ptr< type > type ## Ref;           \
  typedef agx::observer_ptr< type > type ## Observer; \
  typedef agx::ref_ptr< const type > type ## ConstRef;\
  typedef agx::observer_ptr< const type > type ## ConstObserver

  DOXYGEN_START_INTERNAL_BLOCK()

  /**
  Class for override the default delete behavior so that users can implement their own object
  deletion schemes.  This might be done to help implement protection of multiple threads from deleting
  objects unintentionally.
  Note, the DeleteHandler cannot itself be reference counted, otherwise it
  would be responsible for deleting itself!
  */
  class AGXCORE_EXPORT DeleteHandler
  {
    public:

      virtual ~DeleteHandler();

      void doDelete(const Referenced* object);

      /** Request the deletion of an object.
      * Depending on users implementation of DeleteHandler, the delete of the object may occur
      * straight away or be delayed until doDelete is called.
      * The default implementation does a delete straight away.*/
      virtual void requestDelete(const Referenced* object);
  };

#if AGX_DEBUG_REFERENCED()

  AGX_FORCE_INLINE void Referenced::reference( void* ptr ) const
  {
    if ( getEnableStackTrace() )
      m_stackMap.insert( std::make_pair(ptr, agx::generateStackTrace()) );
#else
  AGX_FORCE_INLINE void Referenced::reference( void*) const
  {
#endif
    m_refCount.inc();
  }

#if AGX_DEBUG_REFERENCED()

  AGX_FORCE_INLINE void Referenced::unreference( void* ptr ) const
#else
  AGX_FORCE_INLINE void Referenced::unreference( void* ) const
#endif
  {
    int old = m_refCount.dec();

    agxAssert(old > 0 || old == AGX_EXPIRED_SCOPE_ALLOCATION);

#if AGX_DEBUG_REFERENCED()
    if ( getEnableStackTrace() )
      m_stackMap.erase( ptr );
#endif

    if (old == 1) {
      if (getDeleteHandler()) deleteUsingDeleteHandler();
      else delete this;
    }
  }

  AGX_FORCE_INLINE void Referenced::addObserver( Observer* observer ) const
  {
    agx::ScopeLock<agx::Mutex> scopeLock(m_mutex);

    if (!m_observers)
      allocateObserverVector();

    INSERT_OBSERVER( observer );
  }

  AGX_FORCE_INLINE void Referenced::removeObserver( Observer* observer ) const
  {
    agx::ScopeLock<agx::Mutex> scopeLock(m_mutex);

    REMOVE_OBSERVER( observer );
  }

  AGX_FORCE_INLINE Referenced& Referenced::operator = (const Referenced&)
  {
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE bool Referenced::is() const { return T::template ValidateCast<T>(this); }
  // AGX_FORCE_INLINE bool Referenced::is() const { return dynamic_cast<const T *>(this) != nullptr; }


  // Compiler warns on '!(void*)this' because it is illegal to call member
  // functions on nullptr. However, if it helps us find a nullptr dereference
  // earlier, and with an assert instead of segmentation fault, then I think
  // it's ok to leave the check.
  #include <agx/PushDisableWarnings.h>
  template <typename T>
  AGX_FORCE_INLINE T *Referenced::as() { agxAssert(!(void *)this || this->is<T>()); return static_cast<T *>(this); }
  #include <agx/PopDisableWarnings.h>


  template <typename T>
  AGX_FORCE_INLINE const T *Referenced::as() const { return const_cast<Referenced *>(this)->as<T>(); }


  template <typename T>
  AGX_FORCE_INLINE T *Referenced::asSafe() { return dynamic_cast<T *>(this); }

  template <typename T>
  AGX_FORCE_INLINE const T *Referenced::asSafe() const { return const_cast<Referenced *>(this)->asSafe<T>(); }

  template <typename T>
  AGX_FORCE_INLINE bool Referenced::ValidateCast(const Referenced *object) { return dynamic_cast<const T *>(object) != nullptr; }

  AGX_FORCE_INLINE int Referenced::getReferenceCount() const
  {
    return m_refCount.get();
  }

  AGX_FORCE_INLINE void DeleteHandler::doDelete(const Referenced* object)
  {
    delete object;
  }

  AGX_FORCE_INLINE void DeleteHandler::requestDelete(const Referenced* object)
  {
    doDelete(object);
  }

  DOXYGEN_END_INTERNAL_BLOCK()

}


#endif
