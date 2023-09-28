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


#ifndef AGXDATA_REAL_H
typedef double Real;
#else
//using namespace agx;
#endif

#include <agx/Vector.h>

DOXYGEN_START_INTERNAL_BLOCK()

/// \todo Added for things to compile. Should probably be something else.
struct steps
{
  agx::Real theta0;
  agx::Real theta1;

  enum states
  {
    LOWER = 0,
    UPPER = 1,
  };

  void update(agx::Real, int, int) {}
};


namespace agx
{
  template<typename T>
  class ValarrayPOD : public agx::VectorPOD<T>
  {
  public:

    ValarrayPOD<T>& operator=(const T& val)
    {
      std::for_each(this->begin(), this->end(), [val](T& v) { v = val; });

      return *this;
    }

    ValarrayPOD<T>& operator +=(double val)
    {
      std::for_each(this->begin(), this->end(), [val](T& v) { v += val; });
      return *this;
    }

    ValarrayPOD<T>& operator *=(double val)
    {
      std::for_each(this->begin(), this->end(), [val](T& v) { v *= val; });
      return *this;
    }

    ValarrayPOD<T>& operator *(const ValarrayPOD<T>& other)
    {
      size_t m = std::min(this->size(), other.size());
      for (auto i = 0; i < m; i++)
        (*this)[i] *= other[i];

      return *this;
    }



    ValarrayPOD<T>& operator +=(const ValarrayPOD<T>& other)
    {
      size_t m = std::min(this->size(), other.size());
      for (size_t i = 0; i < m; i++)
        (*this)[i] += other[i];

      return *this;
    }

    ValarrayPOD<T>& operator -=(const ValarrayPOD<T>& other)
    {
      size_t m = std::min(this->size(), other.size());
      for (size_t i = 0; i < m; i++)
        (*this)[i] -= other[i];

      return *this;
    }

    ValarrayPOD<T>& operator *(double val)
    {
      std::for_each(this->begin(), this->end(), [val](T& v) { v *= val; });

      return *this;
    }

    ValarrayPOD(size_t n) : agx::VectorPOD<T>(n)
    {
    }

    ValarrayPOD() : agx::VectorPOD<T>()
    {
    }

    ValarrayPOD(const T& val, size_t n) : agx::VectorPOD<T>(n)
    {
      std::for_each(this->begin(), this->end(), [val](T& v) { v = val; });
    }


    void resize(size_t n, const T& val=T())
    {
      agx::VectorPOD<T>::resize(n);
      std::for_each(this->begin(), this->end(), [val](T& v) { v = val; });
    }
  };

  typedef agx::ValarrayPOD<agx::Real> RealValarrayPOD;
  typedef agx::ValarrayPOD<bool> BoolValarrayPOD;
  typedef agx::ValarrayPOD<int> IntValarrayPOD;
}


DOXYGEN_END_INTERNAL_BLOCK()
