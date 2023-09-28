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

#ifndef AGX_BOUND_H
#define AGX_BOUND_H

#include <agx/Real.h>
#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Integer.h>
#include <agxData/Type.h>

namespace agx
{

  /**
  A BoundT represents a range defined by a minimum and a maximum value. Template
  specializations are provided for various VecNt types so that axis aligned
  bounds in space can be represented.
  */
  template <typename T>
  class BoundT
  {
  public:
    typedef T Type;

  public:

    /**
    Create a bound initialized to the empty range located at the type's default value.
    */
    BoundT();


    /**
    \param min The lower bound of the range.
    \param max The upper bound of the range.
    */
    BoundT(const T& min, const T& max);

    /*
    Default copy constructor
    */
    BoundT(const BoundT& other);

    /**
    Convert the bound to a bound with a different underlying type.
    */
    template <typename T2>
    operator BoundT<T2>() const;

    /// \return A reference to the lower end of the bound.
    const T& min() const;
    /// \return A reference to the upper end of the bound.
    const T& max() const;

    /// \return A reference to the lower end of the bound.
    T& min();
    /// \return A reference to the upper end of the bound.
    T& max();

    /// \return The distance from the lower bound to the upper bound.
    T size() const;

    /// \return The midpoint between the lower bound and the upper bound.
    T mid() const;

    /// Relocate the bound to the new location.
    void set(const T& min, const T& max);

    /**
    Scale the bound so that is size is changed according to the given scale, but
    the midpoint remains unchanged.
    */
    template <typename T2>
    BoundT& operator*= (T2 scale);

    /**
    Create a new bound that has the same midpoint as the current bound, but with
    a size that is scaled according to the given scale.
    */
    template <typename T2>
    BoundT operator* (T2 scale);

    /// \return True if the given value lies within the bound. False otherwise.
    bool contains(const T& value) const;

    /// \return A new bound that is the intersection of the current and the given bound.
    BoundT calculateOverlap(const BoundT& other) const;

    /// \return True if the given bound overlaps with the current bound.
    bool hasOverlap(const BoundT& other) const;

    /// Move the bound's midpoint without altering its size.
    BoundT translate(const T& offset) const;

    bool operator== (const BoundT& other) const;
    bool operator!= (const BoundT& other) const;
    BoundT& operator= (const BoundT& other);

    /// \return True if the lower bound is less than or equal to the upper bound.
    bool isValid() const;

  protected:
    T m_min;
    T m_max;

  };


  typedef BoundT<Real> Bound1;
  typedef BoundT<Real32> Bound1f;
  typedef BoundT<Real64> Bound1d;
  typedef BoundT<Int> Bound1i;
  typedef BoundT<Int32> Bound1i32;
  typedef BoundT<Int64> Bound1i64;
  typedef BoundT<UInt> Bound1u;
  typedef BoundT<UInt32> Bound1u32;
  typedef BoundT<UInt64> Bound1u64;


  typedef BoundT<Vec2> Bound2;
  typedef BoundT<Vec2f> Bound2f;
  typedef BoundT<Vec2d> Bound2d;
  typedef BoundT<Vec2i> Bound2i;
  typedef BoundT<Vec2i32> Bound2i32;
  typedef BoundT<Vec2i64> Bound2i64;
  typedef BoundT<Vec2u> Bound2u;
  typedef BoundT<Vec2u32> Bound2u32;
  typedef BoundT<Vec2u64> Bound2u64;


  typedef BoundT<Vec3> Bound3;
  typedef BoundT<Vec3f> Bound3f;
  typedef BoundT<Vec3d> Bound3d;

  typedef BoundT<Vec3i> Bound3i;
  typedef BoundT<Vec3i32> Bound3i32;
  typedef BoundT<Vec3i64> Bound3i64;

  typedef BoundT<Vec3u> Bound3u;
  typedef BoundT<Vec3u32> Bound3u32;
  typedef BoundT<Vec3u64> Bound3u64;



  /* Implementation */

  #define AGX_BOUND_VEC2_OVERRIDE(VecType)                               \
  template <>                                                                 \
  AGX_FORCE_INLINE bool BoundT<VecType>::contains(const VecType& value) const           \
  {                                                                           \
    return value[0] >= m_min[0] && value[1] >= m_min[1] &&                        \
           value[0] <= m_max[0] && value[1] <= m_max[1];                          \
  } \
  template <>                                                                 \
  AGX_FORCE_INLINE bool BoundT<VecType>::hasOverlap(const BoundT<VecType>& other) const \
  {                                                                                       \
    const bool overlaps =                                                                         \
      (m_max[0] >= other.min()[0]) && (other.max()[0] >= m_min[0]) &&                            \
      (m_max[1] >= other.min()[1]) && (other.max()[1] >= m_min[1]);                             \
    return overlaps;                                                                       \
  }                                                                                         \
  template <>                                                                 \
  AGX_FORCE_INLINE bool BoundT<VecType>::isValid() const \
  {                                                                                       \
    return (m_min[0] <= m_max[0] && m_min[1] <= m_max[1]); \
  }                                                                                         \
  template <>                                                                 \
  AGX_FORCE_INLINE BoundT<VecType> BoundT<VecType>::calculateOverlap(const BoundT<VecType>& other) const \
  { \
    return BoundT<VecType>( \
      VecType(std::max(m_min[0], other.min()[0]), std::max(m_min[1], other.min()[1])), \
      VecType(std::min(m_max[0], other.max()[0]), std::min(m_max[1], other.max()[1]))); \
  }

  #define AGX_BOUND_VEC3_OVERRIDE(VecType)                               \
  template <>                                                                 \
  AGX_FORCE_INLINE bool BoundT<VecType>::contains(const VecType& value) const           \
  {                                                                           \
    return value[0] >= m_min[0] && value[1] >= m_min[1] && value[2] >= m_min[2] &&  \
           value[0] <= m_max[0] && value[1] <= m_max[1] && value[2] <= m_max[2];    \
  } \
  template <>                                                                 \
  AGX_FORCE_INLINE bool BoundT<VecType>::hasOverlap(const BoundT<VecType>& other) const \
  {                                                                                       \
  const bool overlaps =                                                                         \
  (m_max[0] >= other.min()[0]) && (other.max()[0] >= m_min[0]) &&                            \
  (m_max[1] >= other.min()[1]) && (other.max()[1] >= m_min[1]) &&                             \
  (m_max[2] >= other.min()[2]) && (other.max()[2] >= m_min[2]);                             \
  return overlaps;                                                                       \
  }                                                                                         \
  template <>                                                                 \
  AGX_FORCE_INLINE bool BoundT<VecType>::isValid() const \
  {                                                                                       \
  return (m_min[0] <= m_max[0] && m_min[1] <= m_max[1] && m_min[2] <= m_max[2]); \
  }                                                                                         \
  template <>                                                                 \
  AGX_FORCE_INLINE BoundT<VecType> BoundT<VecType>::calculateOverlap(const BoundT<VecType>& other) const \
  { \
    return BoundT<VecType>( \
      VecType(std::max(m_min[0], other.m_min[0]), std::max(m_min[1], other.m_min[1]), std::max(m_min[2], other.m_min[2])), \
      VecType(std::min(m_max[0], other.m_max[0]), std::max(m_min[1], other.m_min[1]), std::max(m_min[2], other.m_min[2]))); \
  }

  AGX_BOUND_VEC2_OVERRIDE(Vec2f)
  AGX_BOUND_VEC2_OVERRIDE(Vec2d)

  AGX_BOUND_VEC2_OVERRIDE(Vec2i32)
  AGX_BOUND_VEC2_OVERRIDE(Vec2i64)

  AGX_BOUND_VEC2_OVERRIDE(Vec2u32)
  AGX_BOUND_VEC2_OVERRIDE(Vec2u64)

  AGX_BOUND_VEC3_OVERRIDE(Vec3f)
  AGX_BOUND_VEC3_OVERRIDE(Vec3d)

  AGX_BOUND_VEC3_OVERRIDE(Vec3i32)
  AGX_BOUND_VEC3_OVERRIDE(Vec3i64)

  AGX_BOUND_VEC3_OVERRIDE(Vec3u32)
  AGX_BOUND_VEC3_OVERRIDE(Vec3u64)


  template <typename T> template <typename T2>
  AGX_FORCE_INLINE BoundT<T>::operator BoundT<T2>() const { return BoundT<T2>(static_cast<T2>(m_min), static_cast<T2>(m_max)); }

  template <typename T>
  AGX_FORCE_INLINE BoundT<T>::BoundT() {}

  template <typename T>
  AGX_FORCE_INLINE BoundT<T>::BoundT(const T& min, const T& max) : m_min(min), m_max(max) {}

  template <typename T>
  AGX_FORCE_INLINE BoundT<T>::BoundT(const BoundT& other) : m_min(other.m_min), m_max(other.m_max) {}

  template <typename T>
  AGX_FORCE_INLINE const T& BoundT<T>::min() const { return m_min; }

  template <typename T>
  AGX_FORCE_INLINE const T& BoundT<T>::max() const { return m_max; }

  template <typename T>
  AGX_FORCE_INLINE T& BoundT<T>::min() { return m_min; }

  template <typename T>
  AGX_FORCE_INLINE T& BoundT<T>::max() { return m_max; }

  template <typename T>
  AGX_FORCE_INLINE void BoundT<T>::set(const T& min, const T& max) { m_min = min; m_max = max; }

  template <typename T>
  AGX_FORCE_INLINE T BoundT<T>::size() const { return m_max - m_min; }

  template <typename T>
  AGX_FORCE_INLINE T BoundT<T>::mid() const { return (m_min + m_max) * 0.5; }

  template <typename T>
  AGX_FORCE_INLINE bool BoundT<T>::isValid() const { return m_min <= m_max; }


  template <typename T> template <typename T2>
  AGX_FORCE_INLINE BoundT<T>& BoundT<T>::operator*= (T2 scale)
  {
    T mid = this->mid();
    T halfSize = this->size() * scale * 0.5;
    m_min = mid - halfSize;
    m_max = mid + halfSize;
    return *this;
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE BoundT<T> BoundT<T>::operator* (T2 scale)
  {
    BoundT bound(m_min, m_max);
    bound *= scale;
    return bound;
  }

  template <typename T>
  AGX_FORCE_INLINE bool BoundT<T>::contains(const T& value) const { return value >= m_min && value <= m_max; }

  template <typename T>
  AGX_FORCE_INLINE BoundT<T> BoundT<T>::calculateOverlap(const BoundT& other) const
  {
    return BoundT(std::max(m_min, other.m_min), std::min(m_max, other.m_max));
  }

  template <typename T>
  AGX_FORCE_INLINE bool BoundT<T>::hasOverlap(const BoundT&) const
  {
    agxAbort1("Type specific implementation missing");
    return false;
  }

  template <typename T>
  AGX_FORCE_INLINE BoundT<T> BoundT<T>::translate(const T& offset) const
  {
    return BoundT(m_min + offset, m_max + offset);
  }

  template <typename T>
  AGX_FORCE_INLINE bool BoundT<T>::operator== (const BoundT& other) const
  {
    return m_min == other.m_min && m_max == other.m_max;
  }


  template <typename T>
  AGX_FORCE_INLINE BoundT<T>& BoundT<T>::operator= (const BoundT& other)
  {
    m_min = other.m_min;
    m_max = other.m_max;
    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE bool BoundT<T>::operator!= (const BoundT& other) const
  {
    return m_min != other.m_min || m_max != other.m_max;
  }

  template<typename T>
  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const BoundT<T>& bound)
  {
    output << bound.min() << " : " << bound.max();
    return output;
  }

}

AGX_TYPE_BINDING(agx::Bound1f, "Bound1")
AGX_TYPE_BINDING(agx::Bound1d, "Bound1")
AGX_TYPE_BINDING(agx::Bound2f, "Bound2")
AGX_TYPE_BINDING(agx::Bound2d, "Bound2")
AGX_TYPE_BINDING(agx::Bound3f, "Bound3")
AGX_TYPE_BINDING(agx::Bound3d, "Bound3")

AGX_TYPE_BINDING(agx::Bound1i32, "Bound1i")
AGX_TYPE_BINDING(agx::Bound1i64, "Bound1i")
AGX_TYPE_BINDING(agx::Bound2i32, "Bound2i")
AGX_TYPE_BINDING(agx::Bound2i64, "Bound2i")
AGX_TYPE_BINDING(agx::Bound3i32, "Bound3i")
AGX_TYPE_BINDING(agx::Bound3i64, "Bound3i")

AGX_TYPE_BINDING(agx::Bound1u32, "Bound1u")
AGX_TYPE_BINDING(agx::Bound1u64, "Bound1u")
AGX_TYPE_BINDING(agx::Bound2u32, "Bound2u")
AGX_TYPE_BINDING(agx::Bound2u64, "Bound2u")
AGX_TYPE_BINDING(agx::Bound3u32, "Bound3u")
AGX_TYPE_BINDING(agx::Bound3u64, "Bound3u")


#endif /* _AGX_BOUND_H_ */
