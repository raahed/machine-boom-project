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
#include <agx/Matrix3x3.h>
#include <agx/agx.h>
#include <agx/Vec3.h>
#include <agx/Quat.h>
#include <agx/EulerConvention.h>

namespace agx
{
  class OrthoMatrix3x3;

  template <typename T>
  class Matrix3x3T;

  template <typename T>
  class Matrix4x4T;

  /**
  This class provides conversion services between Euler angles in any of
  the 24 conventions and corresponding quaternions and orthogonal 3x3
  matrices.

  There are 24 distinct Euler angles conventions which can be used to
  parameterize \f$SO(3)\f$. Each of them has a singularity line in which
  only the sum of two of the angles is relevant to the final rotation
  matrix (or quaternion).  Euler angles are defined as three successive
  rotations along three successive axes, no two successive ones being
  identical.  This results in 12 different choice of axis sequence.
  Given that we can choose the reference axes or the rotated ones, this
  leads to 12 choices with co-rotated axes, 12 choices for fixed axes.

  The different conventions are labeled so that it is easy to extract
  the Real nature of each of them.  Following Shoemake, the last bit is
  used to indicate whether we are using the static or rotated frame for
  the coordinate axes.  The next bit indicates whether the first axis is
  repeated as the last.  The next bit is the parity of the axis
  sequence.  The next 3 bits indicate the first axis which is either 0,
  1 or 2.  That information is then used to reduce the number of
  conventions to just two: repeated axes or non-repeated axes.
  Computations are made using permutations so only two computations are
  needed.  The distinction between static axes and moving axes
  corresponds to an inversion of the permutation.  Negative parities
  just negate all the angles.  These last two operations are easily
  performed at the end of a computation.

  When computing a transform matrix from a given EulerAngles, the resulting
  matrix transforms the coordinates of a given vector from the body frame
  to the world frame.  However, in accordance to the OpenGL convention, the
  matrix is to be multiplied from the left which amounts to transposing.
  */
  class AGXCORE_EXPORT EulerAngles
  {
    public:

#define AGX_EULER_AXES()  const int axes[4]  = {0,1,2,0}
#define AGX_EULER_PAXES() const int paxes[4] = {1,2,0,1};

      class EulerOrder
      {
        private:
          EulerConvention::Convention m_c;        // the current convention
        //  const static int axes[4];   // a utility for computations
        //  const static int paxes[4];  // a utility for computations
          // used internally
          inline int  firstAxis() const {
            return ( m_c >> 3 )&3;
          }

        public:
#ifndef SWIG
          const static char *names[]; // needed for debug purposes
#endif
          /** Default constructor: EulerConvention::DEFAULT_CONVENTION convention and 0 angles.*/
          EulerOrder(EulerConvention::Convention = EulerConvention::DEFAULT_CONVENTION);

          /** Get the current convention. */
          int getConvention() const;

          /** Assignment operator.
          \return true if c is a valid EulerConvention
          */
          bool operator=( const EulerConvention::Convention &c ); // NOLINT

          /**
          Assignment operator and conversion from int.
          \return true if c is a valid EulerConvention
          */
          bool operator=( const int &c ); // NOLINT We should consider wether we really want to break convention here.

          /** The computational routines used by the EulerAngle class. */
          inline bool isStatic() const  {
            return ! ( m_c&1 );
          }
          inline bool isOdd()    const  {
            return ( m_c >> 2 )&1;
          }
          inline bool isRepeat() const  {
            return ( m_c >> 1 )&1;
          }
          inline int  i() const {
            AGX_EULER_AXES();
            return axes[firstAxis()];
          }
          inline int  j() const {
            AGX_EULER_PAXES();
            return paxes[i()+isOdd()];
          }
          inline int  k() const {
            AGX_EULER_PAXES();
            return paxes[i()+!isOdd()];
          }
      };


    private:
      Real m_angles[3];
      EulerOrder m_o;
      void applyParity() ;
      void shiftxz();
      void flip();
      void permute();
      enum Axes {X, Y, Z, W};

    public:

      /**Get the first angle. */
      Real x()const;

      /**Get or set the first angle. */
      Real& x();

      /**Get the second angle. */
      Real y()const;

      /**Get or set the second angle. */
      Real& y();

      /**Get the third angle. */
      Real z()const;

      /**Get or set the third angle. */
      Real& z();

      /**
      Get an element by index as a reference.
      */
      inline Real& operator [] (size_t i);

      /**
      Get an element by index.
      */
      inline Real operator [] (size_t i) const;

      /**
      Set the three angles in one call
      \param angles - Angles specified in radians.
      */
      inline void set( const Vec3& angles );

      /**
      Set the three angles in one call
      \param x,y,z - Angles specified in radians.
      */
      inline void set( Real x,
                       Real y,
                       Real z );


      /**Get the convention. */
      EulerAngles::EulerOrder& getOrder();

      /**Get or set the convention. */
      const EulerAngles::EulerOrder& getOrder() const;

      /** Default constructor.  Convention is set to EulerConvention::DEFAULT_CONVENTION and angles are set to 0. */
      //EulerAngles();

      /** Convention is set to C and angles are set to 0. */
      EulerAngles(EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION);

      /** Convention is set to C and angles are set to 0. */
      //EulerAngles( EulerConvention::Convention C );

      /** Copy constructor*/
      EulerAngles( const EulerAngles & e ) = default;

      /** Explicit constructor. */
      EulerAngles( Real a, Real b, Real c, int C );

      /** Explicit constructor. */
      EulerAngles(Real a, Real b, Real c, EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION);

      /** Assignment operator.*/
      EulerAngles & operator=( const EulerAngles& e ) = default;

      /** Explicit constructor. */
      explicit EulerAngles(const agx::Vec3& angles, EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION);

      /** Construct from 4x4 matrix in given convention.*/
      template <typename T>
      explicit EulerAngles( const Matrix4x4T<T> & m, EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION );

      /** Construct from 3x3 Matrix in given convention.*/
      template <typename T>
      explicit EulerAngles( const Matrix3x3T<T>& m, EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION );
      explicit EulerAngles(const OrthoMatrix3x3& m, EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION);

      /** Construct from quaternion in given convention.*/
      explicit EulerAngles( const Quat& q, EulerConvention::Convention C = EulerConvention::DEFAULT_CONVENTION );

      /** Generate an EulerAngles object with random angles in every axis [ 0 - 2PI ]. */
      static EulerAngles random(const Vec3& min = Vec3(Real(0.0)), const Vec3& max = Vec3(2 * agx::PI), EulerConvention::Convention c = EulerConvention::DEFAULT_CONVENTION);

      /**
      Compute Jacobian matrix between angular velocity in body coordinates
      and rate of change of angles.
      */
      //int setJacobianBody(Matrix3x3 & m);
      /**
      Compute Jacobian matrix between angular velocity in world coordinates
      and rate of change of angles.
      */
      //int setJacobianWorld(Matrix3x3 & m);

      /**
      Set the angles to correspond to matrix m in given convention.  By
      default, the convention used is the value stored in the present object.
      If the last argument is used, the convention is changed to the given
      value.
      \return a reference to this EulerAngles
      */
      template<typename T>
      EulerAngles& set(const Matrix4x4T<T>& m, EulerConvention::Convention c = EulerConvention::BAD);

      /**
      Set the angles to correspond to matrix m in given convention.  By
      default, the convention used is the value stored in the present object.
      If the last argument is used, the convention is changed to the given
      value.
      \return a reference to this EulerAngles
      */
      template<typename T>
      EulerAngles& set(const Matrix3x3T<T> & m, EulerConvention::Convention C = EulerConvention::BAD);
      EulerAngles& set(const OrthoMatrix3x3 & m, EulerConvention::Convention C = EulerConvention::BAD);

      /**
      Set the angles to correspond to quaternion q in given convention.  By
      default, the convention used is the value stored in the present object.
      If the last argument is used, the convention is changed to the given
      value.
      \return a reference to this EulerAngles
      */
      EulerAngles& set(const Quat& q, EulerConvention::Convention C = EulerConvention::BAD);

      /**
      Compute the value of matrix m that corresponds to current angles in
      current convention.
      \return true if the conversion was successful
      */
      template<typename T>
      bool get( Matrix4x4T<T>& m ) const;

      /**
      Compute the value of matrix m that corresponds to current angles in
      current convention.
      \return true if the conversion was successful
      */
      template<typename T>
      bool get( Matrix3x3T<T> & m ) const;
      bool get( OrthoMatrix3x3 & m ) const;

      /**
      Compute the value of quaternion q that corresponds to current angles in
      current convention.
      \return true if the conversion was successful
      */
      bool get( Quat &q ) const;

      /**
      \return a pointer to the data
      */
      Real* ptr();

      /**
      \return a const pointer to the data
      */
      const Real* ptr() const;
  };

  inline int EulerAngles::EulerOrder::getConvention() const
  {
    return m_c;
  }

  inline bool EulerAngles::EulerOrder::operator=( const EulerConvention::Convention &c ) // NOLINT
  {
    this->m_c = c;
    return true;
  }

  inline EulerAngles::EulerOrder::EulerOrder( EulerConvention::Convention c ) : m_c( c ) { }

  inline Real EulerAngles::x() const
  {
    return m_angles[0];
  }

  inline Real& EulerAngles::x()
  {
    return m_angles[0];
  }

  inline Real EulerAngles::y() const
  {
    return m_angles[1];
  }

  inline Real& EulerAngles::y()
  {
    return m_angles[1];
  }

  inline Real EulerAngles::z() const
  {
    return m_angles[2];
  }

  inline Real& EulerAngles::z()
  {
    return m_angles[2];
  }

  inline Real& EulerAngles::operator [] (size_t i)
  {
    return m_angles[i];
  }

  inline Real EulerAngles::operator [] (size_t i) const
  {
    return m_angles[i];
  }


  inline EulerAngles::EulerOrder& EulerAngles::getOrder()
  {
    return m_o;
  }

  inline const EulerAngles::EulerOrder& EulerAngles::getOrder() const
  {
    return m_o;
  }


  inline void EulerAngles::set( const Vec3& angles )
  {
    m_angles[0] = angles[0];
    m_angles[1] = angles[1];
    m_angles[2] = angles[2];
  }

  inline void EulerAngles::set( Real x, Real y, Real z )
  {
    m_angles[0] = x;
    m_angles[1] = y;
    m_angles[2] = z;
  }


  inline std::ostream& operator<< ( std::ostream& os, const EulerAngles& e )
  {
    os << "[" << e[0] << " " << e[1] << " " << e[2] << "]";
    return os;
  }

  AGX_FORCE_INLINE Real* EulerAngles::ptr()
  {
    return m_angles;
  }

  AGX_FORCE_INLINE const Real* EulerAngles::ptr() const
  {
    return m_angles;
  }

  //AGXCORE_EXPORT std::ostream & operator<< ( std::ostream &os, const EulerAngles & a );
  //AGXCORE_EXPORT std::ostream & operator<<( std::ostream &os, const EulerAngles::EulerOrder& o );


} // namespace agx
