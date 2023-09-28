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

#ifndef AGXUTIL_RAWMESH_H
#define AGXUTIL_RAWMESH_H

#include <agx/Vec3.h>
#include <vector>
#include <agx/Vector.h>
#include <agx/Referenced.h>
#include <agxStream/Serializable.h>

namespace agxUtil
{
  AGX_DECLARE_POINTER_TYPES(RawMesh);
  AGX_DECLARE_VECTOR_TYPES(RawMesh);

  /**
  A class to hold a general represenation of a mesh with arbitrary number of vertices in a face.
  */
  class AGXPHYSICS_EXPORT RawMesh : public agx::Referenced, public virtual agxStream::Serializable
  {
  public:
    typedef agx::VectorPOD< agx::UInt > IndexVector;

    // Used for serialization
    RawMesh();

    RawMesh(const agx::String& meshName);

    /**
      Get the number of vertices stored in the mesh. Observe, that depending on initialization, not
      all vertices need to be part of a face.
      \return the number of vertices stored in the object
    */
    size_t getNumVertices() const;

    /**
      Get the number of faces stored in the mesh
      \return the number of faces stored in the object
    */
    size_t getNumFaces() const;

    /**
      Add a vertex to the object on the specified position
      \param position the position of the added vertex
      \return the ID of the added vertex
    */
    size_t addVertex(const agx::Vec3& position);

    /**
      Add a face to the object stored with the specified vertices.
      The vertices is assumed to be stored clockwise from the 'up' side.
      \param vertices the IDs of the vertices of the face
      \return the ID of the added face
    */
    size_t addFace(const IndexVector& vertices);

    /**
      Get the position of a vertex
      \param vertex the ID of the vertex
      \return the position of the vertex
    */
    agx::Vec3 getVertexPosition(size_t vertex) const;

    /**
      Get the area of a face. This is calculated by taking the mid point
      of all the vertices, and then calculating the area of each triangle
      defined by two neighbours and the mid point. Unfortunately, this
      only gives the correct answer if the polygon is convex.
      \param face the ID of the face
      \return the area of the face
    */
    agx::Real getFaceArea(size_t face) const;

    /**
      Get the area represented by a single vertex. This is calculated as an equal part of
      each face the vertex is part of.
    */
    agx::Real getVertexArea(size_t vertex) const;
    /**
      Get the area of all the triangles calculated as described for getFaceArea.
      \return the total area of the mesh
    */
    agx::Real getTotalArea() const;

    /**
      Get the index each vertex in the specified face. They are ordered clockwise, with the
      last index being connected to the first.
    */
    const IndexVector& getFaceVertices(size_t face) const;

    /**
      Get the faces which contain a specified vertex.
      \param vertex the ID of the vertex
      \return a list of the IDs of all the faces
    */
    const IndexVector& getFacesFromVertex(size_t vertex) const;

    /**
      Get the name of the mesh
      \return the name which the mesh is initialized with
    */
    const agx::String& getName() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::RawMesh);

   private:
     agx::Real calculateFaceArea(size_t face) const;

   private:
     agx::String m_name;
     agx::Real m_totalArea;

     agx::Vector<agx::Vec3> m_vertices;
     agx::Vector<agx::Real> m_vertexAreas;
     agx::Vector< IndexVector > m_vertexToFaces;

     agx::Vector< IndexVector > m_faces;
     agx::Vector<agx::Real> m_faceAreas;
  };
}

#endif

