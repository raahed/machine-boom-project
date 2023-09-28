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

#include <agx/config/AGX_USE_ASSIMP.h>
#if AGX_USE_ASSIMP()

#include <agxIO/MeshReader.h>

namespace Assimp
{
  class Importer;
}

namespace agxIO
{

  /**
  Data is returned as vertices (getVertices()) and indices (getIndices())
  Some formats supports non-triangle primitives. These shapes will be ignored. However, the vertices
  for this data will still be part of the read data.
  */
  class AGXPHYSICS_EXPORT MeshReaderAssimp : public MeshReader
  {
    public:

      /// Default Constructor
      MeshReaderAssimp();

      /**
      Constructor that use references for data where the read data will be added.
      */
      MeshReaderAssimp(agx::Vec3Vector& vertices, agx::Vec3Vector& normals, agx::UInt32Vector& indices, agx::Vec2Vector& texCoordinates);

      /**
      Read a mesh file using the Assimp loader.
      Any format (Collada) that supports hierarchical models will be merged into one single mesh.
      \param fileName - Path to a mesh file
      \returns number of triangles read.
      */
      size_t readFile( const agx::String& fileName ) override;

      /**
      Parse a mesh model from a string of a specific format defined as a string.
      \param meshString - string containing the mesh model
      \param fet - Mesh format of the string data
      */
      size_t readString( const agx::String& meshString, FileExtension::FileType fet) override;


      /**
      Parse a mesh model from an input stream of a specific format.
      \param stream - input stream from which we will read the data
      \param fet - Mesh format of the input stream
      */
      size_t readStream(std::istream& stream, FileExtension::FileType fet) override;

      using MeshReader::readString;

  protected:
    size_t importModel(std::stringstream& stream, const agx::String& fileName, FileExtension::FileType fet, Assimp::Importer& importer);


    virtual ~MeshReaderAssimp();
    agx::ref_ptr<agx::Referenced> m_logger;

    agx::Vec3Vector& m_assimp_vertices;
    agx::Vec3Vector& m_assimp_normals;
    agx::UInt32Vector& m_assimp_indices;
    agx::Vec2Vector& m_assimp_texCoordinates;
  };

  typedef agx::ref_ptr<MeshReaderAssimp> MeshReaderAssimpRef;
}

// AGX_USE_ASSIMP.
#endif
