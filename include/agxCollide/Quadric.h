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

#include <agx/agxPhysics_export.h>
#include <agx/Vec3.h>
#include <agx/Matrix3x3.h>
#include <agx/Matrix4x4.h>


DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide {


  /**
  Internal base class used to represent implicit shapes
  */
  class AGXPHYSICS_EXPORT Quadric
  {
  public:
    virtual void printEquation() const = 0;

    virtual ~Quadric() {}

  };


  /**
  2D Quadric:  Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
  */
  class AGXPHYSICS_EXPORT Quadric2D : public Quadric
  {
    public:
      Quadric2D();

      /**
      */
      static Quadric2D circle( agx::Real radius );

      /**
      */
      static Quadric2D empty();

      /**
      */
      void set( agx::Real A, agx::Real B, agx::Real C, agx::Real D, agx::Real E, agx::Real F );

      agx::Real evaluate( agx::Vec2 point ) const;

      virtual void printEquation() const override;

      const agx::Matrix3x3& getMatrix() const;

      void setToCircle( agx::Real radius );


    private:

      // Quadric3D accesses setMatrix.
      friend class Quadric3D;


      /*
      Set the elemtents in the matrix
      */
      void setMatrix( agx::Real e00, agx::Real e01, agx::Real e02,
                                     agx::Real e11, agx::Real e12,
                                                    agx::Real e22 );

      agx::Matrix3x3 m_A;
  };



  /**
  3D Quadric: Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
  */
  class AGXPHYSICS_EXPORT Quadric3D : public Quadric
  {

    public:
      Quadric3D();

      void set( agx::Real A, agx::Real B, agx::Real C, agx::Real D, agx::Real E,
                agx::Real F, agx::Real G, agx::Real H, agx::Real I, agx::Real J );


      /**
      Return a Quadric3D representing a cylinder along the y-axis.
      */
      static Quadric3D cylinder( agx::Real radius );

      /**
      Return a Quadric3D representing a cone along the y-axis.
      The parameter angle specifies the angle between the cone axis and the its side.
      The value must be in the range (0,PI/2).
      */
      static Quadric3D cone( agx::Real angle );

      /**
      Return a degenerate Quadric3D which consists of a single point (0,0,0)
      */
      static Quadric3D empty();

      /**
      \return Value of the quadric evaluated at point
      */
      agx::Real evaluate( agx::Vec3 point ) const;

      virtual void printEquation() const override;


      const agx::Matrix4x4& getMatrix() const;


      Quadric2D projectXY() const;

      /**
      Return a new Quadric
      */
      Quadric3D transform( agx::Matrix4x4 M ) const;

    private:

      agx::Matrix4x4 m_A;
  };


}

DOXYGEN_END_INTERNAL_BLOCK()

