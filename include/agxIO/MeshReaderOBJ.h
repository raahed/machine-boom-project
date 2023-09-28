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

#ifndef AGXIO_MESHREADEROBJ_H
#define AGXIO_MESHREADEROBJ_H

#include <agxIO/MeshReader.h>
namespace agxIO
{

  /**
  Class for reading a Wavefront OBJ file.
  Data is returned as vertices (getVertices()) and indices (getIndices())
  \deprecated Use MeshReader for reading all types of mesh files

  */
  class AGXPHYSICS_EXPORT MeshReaderOBJ : public MeshReader
  {
    public:

      /// Constructor
      MeshReaderOBJ( ) : MeshReader() {}

      /**

      OBSOLETE

      USE MeshReader instead, for any type of mesh.

      Read a file.
      \param filename Wavefront OBJ file containing triangles
      \return number of found triangles.
      */
      size_t readFile( const agx::String& filename );

      /**
      OBSOLETE

      USE MeshReader instead, for any type of mesh.

      Read a file.
      \param mesh - Wavefront OBJ description as a string
      \return number of found triangles.
      */
      size_t readString( const agx::String& mesh );
      using MeshReader::readString;

  protected:
    virtual ~MeshReaderOBJ() {}

  };

  typedef agx::ref_ptr<MeshReaderOBJ> MeshReaderOBJRef;
}

#endif
