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

#ifndef AGX_LOCALVECTOR_H
#define AGX_LOCALVECTOR_H

#include <agx/Vector.h>
#include <agx/ThreadLocalAllocator.h>

#include <agxData/Type.h>


namespace agxData
{
  /**
  Local scope vector. Fast allocation/deallocation using a thread local memory buffer.

  Should only be used for local, short-lived, frequently created containers.

  Beware that a LocalVector's allocation is unstable. All LocalVectors owned by
  the same thread get their memory allocations from the same memory pool. If the
  memory pool is exhausted then it is reallocated and ALL LocalVectors owned by
  that thread get a new buffer. In C++-speak, an operation that invalidates
  iterators/pointers into one LocalVectors invalidates pointers/iterators to ALL
  LocalVectors owned by the same thread. Avoid calling functions that may create
  LocalVectors when holding pointers/interators into LocalVectors, including
  using range for loops.
  */
  template <typename T>
  class LocalVector : public agx::VectorPOD<T, agx::ThreadLocalAllocator>
  {
  public:
    LocalVector();
    LocalVector(size_t size, const T& value = T());
    LocalVector(const LocalVector<T>& other);
    LocalVector(LocalVector<T>&& other);
  };


  /* Implementation */


  template <typename T>
  AGX_FORCE_INLINE LocalVector<T>::LocalVector()
    : agx::VectorPOD<T, agx::ThreadLocalAllocator>()
  {
  }

  template <typename T>
  AGX_FORCE_INLINE LocalVector<T>::LocalVector(size_t size, const T& value)
    : agx::VectorPOD<T, agx::ThreadLocalAllocator>(size, value)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE LocalVector<T>::LocalVector(const LocalVector<T>& other)
    : agx::VectorPOD<T, agx::ThreadLocalAllocator>(other)
  {
  }

  template <typename T>
  inline LocalVector<T>::LocalVector(LocalVector<T>&& other)
    : agx::VectorPOD<T, agx::ThreadLocalAllocator>(std::move(other))
  {
  }
}


AGX_TEMPLATED_TYPE_BINDING(agxData::LocalVector, "LocalVector")

#endif /* AGX_LOCALVECTOR_H */
