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

#ifndef AGXCOLLIDE_CONVEX_H
#define AGXCOLLIDE_CONVEX_H

#include <agx/agxPhysics_export.h>
#include <agxCollide/Shape.h>

#include <agxStream/Serializable.h>

#include <agx/agx_vector_types.h>
#include <agx/SPDMatrix3x3.h>
#include <agxCollide/Trimesh.h>

namespace agxCollide
{

  AGX_DECLARE_POINTER_TYPES(Convex);
  /**
  A convex class for geometric intersection tests.
  More detailed ways of creating it can be found in agxUtil::ConvexReaderWriter.
  */
  class AGXPHYSICS_EXPORT Convex : public Trimesh
  {
    public:
      /**
      Constructs a new Convex object from a vector of vertices and indices each.
      The vertices are assumed to be a convex hull, and the indices to connect
      them as triangle lists, where the triangles lie on the convex hull.
      \param vertices Pointer to a Vector of vertices.
                      Vertices should not be duplicates of each other.
                      Has to have at least size 1.
                      All content will be copied. No ownership will be taken.
      \param indices Pointer to a Vector of size_t indicating the vertices
                     for each triangle by referring to the vertices parameter.
                     All triangles described by a set of 3 indices each have
                     to have counter-clockwise winding (all-clockwise-winding
                     can be specified in TrimeshOptionsFlags).
                     Has to have at least size 3.
                     All content will be copied. No ownership will be taken.
      \param sourceName Name of the data source for debugging.
      \param optionsMask Options can be set as in enum TrimeshOptionsFlags.
      */
      Convex( const agx::Vec3Vector* vertices, const agx::UInt32Vector* indices,
              const char* sourceName,
              uint32_t optionsMask = Trimesh::REMOVE_DUPLICATE_VERTICES );

      /**
      Inherited from Shape.
      \return true if the class have a support function?
      */
      virtual bool hasSupportFunction() const override;

      /// \return point furthest away along a direction.
      virtual agx::Vec3 getSupportPoint( const agx::Vec3& supportDirection ) const override;

      /**
      Creates a new Convex which shares data with this one.
      (should not be used for data which will be changed afterwards).
      Quick to create, minimum memory overhead.
      The caller takes over possession of the pointer.
      \return pointer to a new Convex which share data with this Convex.
      */
      Convex* shallowCopy() const;

      /**
      Creates a new Convex which copies data from this one.
      (modifying one's data will not modify the other's).
      Less quick to create than shallowCopy (but faster than
      creating mesh from input data), double memory usage.
      The caller takes over possession of the pointer.
      \return pointer to a new Convex which do not share any data with this Convex.
      */
      Convex* deepCopy() const;

      virtual Shape *clone() const override;


      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Convex);

    private:

      /// Hiding default constructor
      Convex();

      /// Hiding copy constructor
      Convex( const Convex& convex );

      /// Hiding assignment
      Convex& operator=(const Convex& convex);

    protected:
      /// Hiding destructor
      virtual ~Convex();

  };

  /// Implementation
  AGX_FORCE_INLINE bool Convex::hasSupportFunction() const
  {
    return true;
  }


  AGX_FORCE_INLINE agx::Vec3 Convex::getSupportPoint( const agx::Vec3& supportDirection ) const
  {
    if ( m_collisionMeshData->m_vertices.empty() )
      return agx::Vec3();
    else {
      agx::Real maxDist = -agx::RealMax;
      size_t maxIndex = 0;
      for (size_t i = 0; i < m_collisionMeshData->m_vertices.size(); ++i) {
        const agx::Real dist = m_collisionMeshData->m_vertices[i] * supportDirection;
        if (dist > maxDist) {
          maxDist = dist;
          maxIndex = i;
        }
      }
      return m_collisionMeshData->m_vertices[maxIndex];
    }
  }



}

#endif

