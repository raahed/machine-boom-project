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

#ifndef AGX_DATA_H
#define AGX_DATA_H

#include <agx/Vector.h>


namespace agxData
{
  // Forward declarations
  class AbstractScalar;
  class AbstractArray;

  template <typename T>
  class Scalar;

  template <typename T>
  class Array;


  /**
  Generic data
  */
  class AGXCORE_EXPORT Data
  {
  public:
    Data(void *ptr = nullptr);

    /**
    \return A pointer to the raw data.
    */
    void *ptr();
    const void *ptr() const;


    // Convenience methods to static cast to Scalar (implemented in Scalar.h)
    const AbstractScalar& asScalar() const;
    AbstractScalar& asScalar();

    template <typename T>
    Scalar<T>& asScalar();

    template <typename T>
    const Scalar<T>& asScalar() const;

    // Convenience methods to static cast to Array (implemented in Array.h)
    const AbstractArray& asArray() const;
    AbstractArray& asArray();

    template <typename T>
    Array<T>& asArray();

    template <typename T>
    const Array<T>& asArray() const;

  protected:
    void *m_ptr;
  };


  typedef agx::VectorPOD<Data *> DataPtrVector;


  /* Implementation */

  AGX_FORCE_INLINE Data::Data(void *ptr) : m_ptr(ptr) {}


  AGX_FORCE_INLINE void *Data::ptr() { return m_ptr; }
  AGX_FORCE_INLINE const void *Data::ptr() const { return m_ptr; }

}


#endif /* AGX_DATA_H */

