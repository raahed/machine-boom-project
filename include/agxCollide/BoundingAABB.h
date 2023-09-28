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

#ifndef AGXCOLLIDE_BOUNDINGAABB_H
#define AGXCOLLIDE_BOUNDINGAABB_H

#include <agx/agxPhysics_export.h>

#include <agx/agx.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/Bound.h>
#include <agxData/Type.h>

#include <iosfwd>


namespace agxCollide
{

  /**
  Axis aligned bounding box implementation.
  */
  class AGXPHYSICS_EXPORT BoundingAABB : public agx::Bound3
  {
    public:
      /// Creates an empty bounding volume. isCleared() will return 'true'.
      BoundingAABB();

      /**
      Creates a bounding volume given a component-wise minimum and maximum point.
      \param min The component-wise minimum.
      \param max The component-wise maximum.
      If min is not the component-wise minimum and max is not the component-wise maximum,
      unexpected results might occur.
      */
      BoundingAABB(const agx::Vec3& min, const agx::Vec3& max);

      /**
      Creates a bounding volume given a bounding volume given in another coordinate-system.
      \param other The bounding volume, given in another coordinate system.
      \param transform The transformation matrix from the coordinate system where
         'other' is defined to the desired one.
      */
      BoundingAABB(const agx::Bound3& other, const agx::AffineMatrix4x4& transform);

      /**
      Removes any extent of the bound. isCleared() will return 'true'.
      */
      void reset();

      /**
      Sets a bounding volume given a bounding volume given in another coordinate-system.
      \param other The bounding volume, given in another coordinate system.
      \param transform The transformation matrix from the coordinate system where
         'other' is defined to the desired one.
      */
      void set(const agx::Bound3& other, const agx::AffineMatrix4x4& transform);

      /**
      Expands a bounding volume to contain both itself and another bound.
      If isCleared() was true before, the bound will be set to the other one.
      */
      void expand(const agx::Bound3& other);

      /**
      Expands a bounding volume to contain both itself and another point.
      If isCleared() was true before, the bound will be set to the point.
      */
      void expand(const agx::Vec3& point);

      /**
      Expands a bounding volume to contain both itself and another bound.
      If isCleared() was true before, the bound will be set the other one.
      \param otherMin The component-wise minimum.
      \param otherMax The component-wise maximum.
      If min is not the component-wise minimum and max is not the component-wise maximum,
      unexpected results might occur.
      */
      void expand(const agx::Vec3& otherMin, const agx::Vec3& otherMax);

      /**
      Expands a bounding volume given a bounding volume given in another coordinate-system.
      If isCleared() was true before, the bound will be set to the point.
      \param other The bounding volume, given in another coordinate system.
      \param transform The transformation matrix from the coordinate system where
         'other' is defined to the desired one.
      */
      void expand(const agx::Bound3& other, const agx::AffineMatrix4x4& transform);

      /**
      Expands a bounding volume given a bounding volume given in another coordinate-system.
      If isCleared() was true before, the bound will be set to the point.
      \param otherMin The component-wise minimum, given in another coordinate system.
      \param otherMax The component-wise maximum, given in another coordinate system.
      \param transform The transformation matrix from the coordinate system where
      'otherMin' and 'otherMax' are defined to the desired one.
      */
      void expand(const agx::Vec3& otherMin, const agx::Vec3& otherMax, const agx::AffineMatrix4x4& transform);

      /**
      Tests if the bound intersects with another bounding volume.
      \param other The other bounding volume, to be given in same coordinate system.
      \retval Do the bounding volumes overlap?
      */
      bool hasOverlap(const BoundingAABB* other) const;

      /**
      Test if a sphere overlaps the bound.
      May give false positives outside the bound's edges or corners.
      \param pos The position of the sphere.
      \param radius The radius of the sphere
      \retval Do the sphere and the bound overlap? May give false positives outside the bound's edges or corners.
      */
      bool testSphereOverlap(const agx::Vec3& pos, agx::Real radius) const;

      /**
      Is the bounding volume cleared? Will be true e.g. after call to reset().
      */
      bool isCleared() const;

      using agx::Bound3::set;
      using agx::Bound3::hasOverlap;
  };

#ifndef SWIG
  static_assert(sizeof(agx::Bound3) == sizeof(BoundingAABB), "Size mismatch");
#endif



  /* Implementation */
  AGX_FORCE_INLINE BoundingAABB::BoundingAABB()
  {
    reset();
  }


  AGX_FORCE_INLINE BoundingAABB::BoundingAABB(const agx::Vec3& min, const agx::Vec3& max) : agx::Bound3(min, max)
  {
  }


  AGX_FORCE_INLINE BoundingAABB::BoundingAABB(const agx::Bound3& other, const agx::AffineMatrix4x4& transform)
  {
    this->reset();
    this->expand(other, transform);
  }


  AGX_FORCE_INLINE void BoundingAABB::set(const agx::Bound3& other, const agx::AffineMatrix4x4& transform)
  {
    this->reset();
    this->expand(other, transform);
  }


  AGX_FORCE_INLINE void BoundingAABB::reset()
  {
    min().set(agx::RealMax);
    max().set(-agx::RealMax);
  }


  AGX_FORCE_INLINE bool BoundingAABB::isCleared() const
  {
    return min()[0] > max()[0];
  }


  AGX_FORCE_INLINE void BoundingAABB::expand(const agx::Bound3& other)
  {
    expand(other.min(), other.max());
  }


  AGX_FORCE_INLINE void BoundingAABB::expand(const agx::Vec3& point)
  {
    expand(point, point);
  }


  AGX_FORCE_INLINE void BoundingAABB::expand(const agx::Vec3& otherMin, const agx::Vec3& otherMax)
  {
    min() = agx::Vec3::componentMin(min(), otherMin);
    max() = agx::Vec3::componentMax(max(), otherMax);
  }


  AGX_FORCE_INLINE void BoundingAABB::expand(const agx::Bound3& other, const agx::AffineMatrix4x4& transform)
  {
    expand( other.min(), other.max(), transform );
  }


  AGX_FORCE_INLINE bool BoundingAABB::hasOverlap(const BoundingAABB* other) const
  {
    return agx::Bound3::hasOverlap(*other);
  }


  AGX_FORCE_INLINE bool BoundingAABB::testSphereOverlap(const agx::Vec3& pos, agx::Real radius) const
  {
    const bool retval =
      pos[0] > (min()[0] - radius) && pos[1] > (min()[1] - radius) && pos[2] > (min()[2] - radius) &&
      pos[0] < (max()[0] + radius) && pos[1] < (max()[1] + radius) && pos[2] < (max()[2] + radius);
    return retval;
  }


  inline std::ostream& operator << ( std::ostream& output, const BoundingAABB& bound )
  {
    output << "min(): " << bound.min() << ", max(): " << bound.max();
    return output;
  }

}

AGX_TYPE_BINDING(agxCollide::BoundingAABB, "BoundingAABB")

#endif
