/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_PLANE_H
#define AGX_PLANE_H

#include <agx/Vec4.h>
#include <agx/AffineMatrix4x4.h>

namespace agx
{

  /**
   * Class representing the mathematical concept of a plane, also called a half-
   * space. Not to be confused with the geometrical shape Plane defined in agxCollide.
   */
  template <typename T>
  class PlaneT
  {
    public:

      /** Create a plane with it surface on the X and Y axes. */
      PlaneT();

      /**
      Create a plane from a Vec4T. The first three elements become the normal and the last element
      is the distance from the origin to the newly created plane.
      */
      PlaneT( const agx::Vec4T<T>& plane);

      /**
      a, b, and c forms the normal of the plane and d is the distance from the origin.
      ax+by+cz+d=0
      */
      PlaneT( const T a, const T b, const T c, const T d );

      /**
      Constructor
      \param normal Plane normal (a,b,c) in plane equation.
      \param d Distance from origin.
      */
      PlaneT( const agx::Vec3T<T>& normal, const T d );

      /**
      Constructor
      \param normal - Normal (a,b,c) in the plane equation
      \param point - A point on the plane
      */
      PlaneT( const agx::Vec3T<T>& normal, const agx::Vec3T<T>& point );

      /**
      Constructor. Plane defined by three points in space.
      */
      PlaneT(const agx::Vec3T<T>& p1, const agx::Vec3T<T>& p2, const agx::Vec3T<T>& p3);

      /**
      Cast to Vec4. The first three elements of the returned Vec4T hold the normal and the last
      element holds the distance.
      */
      operator const agx::Vec4T<T>& () const;

      /** /return The normal of the plane. */
      agx::Vec3T<T> getNormal() const;

      /** \return The shortest distance from the origin to the plan. */
      T getDistance() const;

      /** Change the orientation of the plane so that the given normal becomes the new normal of the plane */
      void setNormal( const agx::Vec3T<T>& normal );

      /** Translate the plane parallel to the normal so that the given distance becomes the new distance from the origin. */
      void setDistance( const T distance );

      /**
      Reconfigure this plane according the given specification. The first three elements of the Vec4 are taken to
      be the normal and the last element becomes the distance from the origin.
       */
      void set(const agx::Vec4T<T>& plane);

      /** Make this plane a copy of the given plane. */
      void set(const agx::PlaneT<T>& plane);

      /**
      Calculates the distance from the plane to a point. Negative value if the point is behind the plane, otherwise positive.
      */
      T signedDistanceToPoint( const agx::Vec3T<T>& point ) const;

      /**
      Projects the point \p point onto this plane.
      \param point - point to project onto this plane
      \return point in plane
      */
      agx::Vec3T<T> project( const agx::Vec3T<T>& point ) const;

      /** Create a new plane that is equal to the current plane transformed by the given transform. */
      PlaneT calculatePlane( const agx::AffineMatrix4x4& transform ) const;

      /** Create a new plane that is equal to the current plane transformed by the given transform. */
      PlaneT calculateTransformedPlane(const agx::AffineMatrix4x4& transform ) const;

    private:
      void init(const agx::Vec3T<T>& normal, const T distance);

    private:
      agx::Vec4T<T> m_data;
  };

  typedef PlaneT<agx::Real> Plane;
  typedef PlaneT<agx::Real32> Plane32;
  typedef PlaneT<agx::Real64> Plane64;





  /////// Implementation

  template <typename T>
  std::ostream& operator<<(std::ostream& output, const PlaneT<T>& plane)
  {
    output << plane.getNormal() << " : "  << plane.getDistance();
    return output;
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::PlaneT()
  {
    this->init(Vec3T<T>(0, 0, 1), 0);
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::PlaneT(const Vec4T<T>& plane) : m_data(plane)
  {}

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::PlaneT( const T a, const T b, const T c, const T d )
  {
    this->init(Vec3T<T>(a, b, c), d);
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::PlaneT( const Vec3T<T>& normal, const T d )
  {
    this->init(normal, d);
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::PlaneT( const Vec3T<T>& normal, const Vec3T<T>& point )
  {
    T d = - (point * normal);
    this->init(normal, d);
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::PlaneT(const Vec3T<T>& p1, const Vec3T<T>& p2, const Vec3T<T>& p3)
  {
    Vec3T<T> one = p1 - p2;
    Vec3T<T> two = p1 - p3;
    Vec3T<T> normal = one ^ two;
    normal.normalize();
    T d = - (p1*normal);
    this->init(normal, d);
  }

  template <typename T>
  AGX_FORCE_INLINE void PlaneT<T>::init(const Vec3T<T>& normal, T distance)
  {
    m_data.set(normal[0], normal[1], normal[2], distance);
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T>::operator const Vec4T<T>& () const
  {
    return m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> PlaneT<T>::getNormal() const { return Vec3T<T>(m_data[0], m_data[1], m_data[2]); }

  template <typename T>
  AGX_FORCE_INLINE T PlaneT<T>::getDistance() const { return m_data[3]; }

  template <typename T>
  AGX_FORCE_INLINE void PlaneT<T>::setNormal( const Vec3T<T>& normal ) { m_data = Vec4T<T>(normal, m_data[3]); }

  template <typename T>
  AGX_FORCE_INLINE void PlaneT<T>::setDistance( T distance ) { m_data[3] = distance; }

  template <typename T>
  AGX_FORCE_INLINE void PlaneT<T>::set(const agx::Vec4T<T>& plane) { this->init(agx::Vec3T<T>(plane[0], plane[1], plane[2]), plane[3]); }

  template <typename T>
  AGX_FORCE_INLINE void PlaneT<T>::set(const agx::PlaneT<T>& plane) { this->init(plane.getNormal(), plane.getDistance()); }

  template <typename T>
  AGX_FORCE_INLINE T PlaneT<T>::signedDistanceToPoint( const Vec3T<T>& point ) const
  {
    return ( getNormal() * point ) + getDistance();
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> PlaneT<T>::project( const Vec3T<T>& point ) const
  {
    return point - signedDistanceToPoint( point ) * getNormal();
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T> PlaneT<T>::calculatePlane( const agx::AffineMatrix4x4& transform ) const
  {
    agx::Vec3T<T> normal, point;

    // calculate one known point on the current plane
    point = getNormal() * (-getDistance());

    // transform the point (so it'll be on the new plane)
    point = point * transform;

    // calculate the new planes normal
    normal = transform.transform3x3( getNormal() );

    // return the plane,  n dot p = -d
    return PlaneT<T>(normal, - (normal * point));
  }

  template <typename T>
  AGX_FORCE_INLINE PlaneT<T> PlaneT<T>::calculateTransformedPlane(const agx::AffineMatrix4x4& transform) const
  {
    Vec3T<T> normal, point;

    // calculate one known point on the current plane
    point = getNormal() * (-getDistance());

    // transform the point (so it'll be on the new plane)
    point = point * transform;

    // calculate the new planes normal
    normal = transform.transform3x3( getNormal() );

    // return the plane,  n dot p = -d
    return PlaneT<T>(normal, - (normal * point));
  }

}

AGX_TYPE_BINDING(agx::Plane32, "Plane")
AGX_TYPE_BINDING(agx::Plane64, "Plane")

#endif


