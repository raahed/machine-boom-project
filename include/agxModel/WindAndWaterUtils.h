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

#ifndef AGXMODEL_WINDANDWATERUTILS_H
#define AGXMODEL_WINDANDWATERUTILS_H

#include <agxModel/export.h>

#include <agx/debug.h>
#include <agx/Vector.h>
#include <agx/Vec3.h>
#include <agx/agx_valarray_types.h>

#include <agxCollide/Convex.h>

namespace agxModel
{
  /**
  Vertex data for all global vertices in a mesh.
  */
  struct AGXMODEL_EXPORT VertexData
  {
    typedef agx::VectorPOD< VertexData > Container;

    /**
    \return true if this vertex is below "the surface", i.e., in the water
    */
    agx::Bool isBelowSurface() const;
    agx::Bool isOnSurface( agx::Real threshold = agx::REAL_SQRT_EPSILON ) const;

    agx::Vec3 worldPosition;          /**< World position of this vertex. */
    agx::Vec3 worldVelocity;          /**< Velocity of this vertex given in world coordinate system. */
    agx::Vec3 worldVelocityOfMedium;  /**< Velocity of the medium this vertex is in (water current or wind velocity). */
    agx::Real heightFromSurface;      /**< Height from surface, negative if below the surface. */
    agx::Real pressure;               /**< Hydro- or aerostatic pressure. */
  };

  /**
  Triangle data for a normal or a clipped triangle.
  */
  struct AGXMODEL_EXPORT TriangleData
  {
    typedef agx::VectorPOD< TriangleData > Container;

    enum State
    {
      BELOW_SURFACE       = 1 << 0,
      ABOVE_SURFACE       = 1 << 1,
      CLIPPED             = 1 << 2,
      CALCULATIONS_ENABLE = 1 << 3
    };

    /**
    \return the calculated area of this triangle (note that the area is cached, this method is usually only used in debug purposes)
    */
    agx::Real calculateArea() const;

    /**
    \return the calculated normal of this triangle (not that the normal is cached, this method is usually only used in debug purposes)
    */
    agx::Vec3 calculateNormal() const;

    /**
    \return true if this is a clipped triangle
    */
    agx::Bool isClipped() const;

    /**
    \return true if this triangle is below the surface
    */
    agx::Bool isBelowSurface() const;

    /**
    \return true if calculations should be done on this triangle
    */
    agx::Bool isCalculationsEnable() const;

    agxModel::VertexData vertices[ 3 ]; /**< Vertices of this triangle. */
    agx::Vec3 worldNormal;              /**< Normal to this triangle given in world coordinate system. */
    agx::Real area;                     /**< Area of this triangle. */
    agx::Int32 state;                   /**< Current state of this triangle. */
    agx::UInt pfrIndex;
  };

  /**
  Global triangle data (mapped with agxCollide::Mesh object) holding vertex indices.
  */
  struct AGXMODEL_EXPORT GlobalTriangleData : public TriangleData
  {
    typedef agx::Vector< GlobalTriangleData > Container;

    agx::UInt vertexIndices[ 3 ];              /**< Vertex indices. */
    agxModel::TriangleData::Container clipped; /**< Container with clipped triangles if this global triangle has been clipped. */
  };

  namespace dense
  {
    class AGXMODEL_EXPORT Vector
    {
      public:
        typedef agx::RealValarray DataType;

      public:
        Vector();
        Vector( agx::UInt size );

        agx::UInt size() const;

        void resize( agx::UInt size );

        DataType& getData();
        const DataType& getData() const;

        agx::Real& operator()( const agx::UInt i );
        const agx::Real& operator()( const agx::UInt i ) const;

      private:
        DataType m_data;
    };

    class AGXMODEL_EXPORT SquareMatrix
    {
      public:
        SquareMatrix( agx::UInt size );

        agx::UInt getSize() const;
        agx::Bool isFactored() const;
        agx::Bool factorize();
        agx::Bool solve( agxModel::dense::Vector& x );

        agx::Real operator()( agx::UInt i, agx::UInt j ) const;
        agx::Real& operator()( agx::UInt i, agx::UInt j );

      private:
        void solveL( Vector& x ) const;
        void solveU( Vector& x ) const;

      private:
        agx::RealValarray m_data;
        agx::UInt m_n;
        agx::Bool m_factored;
    };

    AGX_FORCE_INLINE agx::UInt Vector::size() const
    {
      return (agx::UInt)m_data.size();
    }

    AGX_FORCE_INLINE agx::Real& Vector::operator()( const agx::UInt i )
    {
      return m_data[ i ];
    }

    AGX_FORCE_INLINE const agx::Real& Vector::operator()( const agx::UInt i ) const
    {
      return m_data[ i ];
    }

    AGX_FORCE_INLINE agx::Real SquareMatrix::operator()( agx::UInt i, agx::UInt j ) const
    {
      agxAssert( !m_factored );
      return m_data[ m_n * i + j ];
    }
    AGX_FORCE_INLINE agx::Real& SquareMatrix::operator()( agx::UInt i, agx::UInt j )
    {
      agxAssert( !m_factored );
      return m_data[ m_n * i + j ];
    }
  }

  AGX_FORCE_INLINE agx::Bool VertexData::isBelowSurface() const
  {
    return heightFromSurface < agx::Real( 0 );
  }

  AGX_FORCE_INLINE agx::Bool VertexData::isOnSurface( agx::Real threshold /* = agx::REAL_SQRT_EPSILON */ ) const
  {
    return heightFromSurface > -threshold && heightFromSurface < threshold;
  }

  AGX_FORCE_INLINE agx::Bool TriangleData::isBelowSurface() const
  {
    return (state & BELOW_SURFACE) != 0;
  }

  AGX_FORCE_INLINE agx::Bool TriangleData::isClipped() const
  {
    return (state & CLIPPED) != 0;
  }

  AGX_FORCE_INLINE agx::Bool TriangleData::isCalculationsEnable() const
  {
    return (state & CALCULATIONS_ENABLE) != 0;
  }
}

#endif // AGXMODEL_WINDANDWATERUTILS_H
