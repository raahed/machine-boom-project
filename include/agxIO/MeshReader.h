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

#include <agx/Referenced.h>
#include <agx/Vector.h>
#include <agx/agxPhysics_export.h>
#include <string>
#include <agx/Vec3.h>
#include <agx/agx_vector_types.h>
#include <agxIO/FileSystem.h>

namespace agxIO
{
  /**
  Support mesh file formats.
  */
  struct FileExtension
  {
    enum FileType {
      UNDEFINED = 0,
      OBJ,      /**< Wavefront OBJ */
      SHL,      /**< Shell data file */
      FBX,      /**< Filmbox AutoDesk */
      ASSIMP,   /**< AssImp reader/writer */
      DAE,      /**< Collada */
      BLEND,    /**< Blender 3D */
      MAX3DS,   /**< 3ds Max 3DS */
      ASE,      /**< 3ds Max ASE */
      GLTFB,    /**< Binary GL Transmission format */
      GLTF,     /**< Ascii GL Transmission format */
      XGL,      /**< XGL */
      ZGL,      /**< ZGL */
      PLY,      /**< Stanford Polygon Library */
      DXF,      /**< AutoCAD DXF */
      LWO,      /**< LightWave */
      LWS,      /**< LightWave Scene */
      LXO,      /**< Modo */
      STL,      /**< Stereolithography */
      X,        /**< DirectX X */
      AC,       /**< AC3D */
      MS3D,     /**< Milkshape 3D */
      SCN,      /**< TrueSpace */
      XML,      /**< Ogre XML */
      IRRMESH,  /**< Irrlicht Mesh */
      IRR,      /**< Irrlicht Scene */
      MDL,      /**< Quake I */
      MD2,      /**< Quake II */
      MD3,      /**< Quake III Mesh */
      PK3,      /**< Quake III Map/BSP */
      MD5,      /**< Doom 3 */
      SMD,      /**< Valve Model */
      M3,       /**< Starcraft II M3 */
      UNREAL3D, /**< Unreal */
      Q3D,      /**< Quick3D */
      OFF,      /**< Object File Format */
      TER,      /**< Terragen Terrain */
      IFC,      /**< Industry Foundation Classes (IFC/Step) */
      OGEX      /**< Open Game Engine Exchange(.ogex) */
    };
  };

  AGX_DECLARE_POINTER_TYPES(MeshReader);
  /**
  Class for reading a mesh object from file and extracting vertices, normals and indices for collision detection
  Notice that this class will read the mesh vertices/indices as is. No merging of nearby vertices will be done.
  */
  class AGXPHYSICS_EXPORT MeshReader : public agx::Referenced
  {
  public:

    MeshReader();

    /**
    Read a specified filename containing a mesh description in some mesh format.
    \return number of found triangles.
    */
    virtual size_t readFile( const agx::String& filename );

    /**
    Read a specified string containing a mesh description in some mesh format defined as a string.
    \return number of found triangles.
    */
    virtual size_t readString( const agx::String& mesh, FileExtension::FileType fet );

    /**
    Read a specified stream containing a mesh description in some mesh format.
    \return number of found triangles.
    */
    virtual size_t readStream( std::istream& stream, FileExtension::FileType fet );

    /**
    Get vertices from previously read mesh file.
    \note Must be called after the readFile() method
    \return a const reference to a vector containing vertices found in the model data.
    */
    const agx::Vec3Vector& getVertices() const;

    /**
    Get indices from previously read mesh file.
    \note Must be called after the readFile() method
    \return a const reference to a vector containing indices found in the model data.
    */
    const agx::UInt32Vector& getIndices() const;

    /**
    Get normals from previously read mesh file.
    \note Must be called after the readFile() method
    \return a const reference to a vector containing normals found in the model data.
    */
    const agx::Vec3Vector& getNormals() const;

    /**
    Get texture coordinates from previously read mesh file.
    \note Must be called after the readFile() method
    \return a const reference to a vector containing texture coordinates normals found in the model data
    */
    const agx::Vec2Vector& getTextureCoordinates() const;

    /**
    Get number of triangles from previously read mesh file.
    \note Must be called after the readFile() method
    \return number of triangles.
    */
    size_t getNumTriangles() const;


    /**
    Convert a filename to a known (or unknown) file extension enum
    \param fileName - filename including extension of a mesh file
    \returns enum defining the filetype of the filename
    */
    static FileExtension::FileType getFileExtensionType( const agx::String& fileName);

    /**
    Convert a filetype enum to a string with the file extension (without '.')
    \param type - Enum with the filetype to be returned as a string
    \returns a string with the file type without '.'
    */
    static agx::String getFileExtension( FileExtension::FileType type );

  protected:


    /**
    Read a specified stream containing a mesh description in SHL format.
    \return number of found triangles.
    */
    size_t readShlStream( std::istream& stream );

    /**
    Reset all allocated memory
    */
    void clear();

    virtual ~MeshReader() {}

    agx::Vec3Vector m_vertices;
    agx::Vec3Vector m_normals;
    agx::UInt32Vector m_indices;
    agx::Vec2Vector m_texCoordinates;

    static agx::StringVector m_extensions;
  };

  inline const agx::Vec3Vector& MeshReader::getVertices() const
  {
    return m_vertices;
  }

  inline const agx::UInt32Vector& MeshReader::getIndices() const
  {
    return m_indices;
  }

  inline const agx::Vec3Vector& MeshReader::getNormals() const
  {
    return m_normals;
  }

  inline const agx::Vec2Vector& MeshReader::getTextureCoordinates() const
  {
    return m_texCoordinates;
  }


  inline size_t MeshReader::getNumTriangles() const
  {
    return m_indices.size() / 3;
  }


} // namespace agxIO
