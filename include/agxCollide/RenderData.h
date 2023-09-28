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
#include <agx/Referenced.h>
#include <agx/Vector.h>
#include <agx/Vec3.h>
#include <agx/Vec2.h>
#include <agx/agx_vector_types.h>
#include <agxStream/Serializable.h>
#include <agxCollide/RenderMaterial.h>

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(RenderData);
  /**
  Class for storing rendering information for a Shape.
  This data can later be accessed by a rendering engine.
  You can also derive from this class and implement your own containing more relevant information for your
  specific needs, such as textures, multitexcoords, shaders etc.
  Just make sure you implement the store/restore method too.
 */
  class AGXPHYSICS_EXPORT RenderData : public agx::Referenced, public agxStream::Serializable
  {
    public:

      /// Specifies which type of primitive this render data represents
      enum Mode
      {
//         POINTS=0,
//         LINES=1,
//         LINE_STRIP=2,
//         LINE_LOOP=3,
        TRIANGLES=4
//         TRIANGLE_STRIP=5,
//         TRIANGLE_FAN=6,
//         QUADS=7,
//         QUAD_STRIP=8,
//         POLYGON=9,
      };

      enum StateInfo
      {
        STANDARD=0x0,
        NO_RENDER=0x1
      };

    /// Constructor
    RenderData();

    /// Set the renderMaterial of this RenderData
    void setRenderMaterial( agxCollide::RenderMaterial* renderMaterial );

    /// \return the RenderMaterial for this RenderData. Null if none is set.
    agxCollide::RenderMaterial *getRenderMaterial(  ) const;

    /// \return true if this RenderData has a RenderMaterial specified.
    bool hasRenderMaterial(  ) const;

    /// Set the vertices
    void setVertexArray( const agx::Vec3Vector& vertices );

    /// \return a reference to the vertices
    const agx::Vec3Vector& getVertexArray() const;

    /// Set the indices
    void setIndexArray( const agx::UInt32Vector& indices, Mode mode );

    /// \return a reference to the indices
    const agx::UInt32Vector& getIndexArray() const;

    /// Set the per vertex normals
    void setNormalArray( const agx::Vec3Vector& normals );

    /// \return a reference to the per vertex normals
    const agx::Vec3Vector& getNormalArray( ) const;

    /// Set the per vertex texture coordinates
    void setTexCoordArray( const agx::Vec2Vector& texCoords );

    /// \return a reference to the texture per vertex coordinates
    const agx::Vec2Vector& getTexCoordArray( ) const;

    /// Set and copy the per vertex colors
    void setColorArray( const agx::Vec4Vector& color );

    /// \return a reference to the per vertex colors
    const agx::Vec4Vector& getColorArray( ) const;


    /// \return the current primitive mode
    Mode getMode() const;

    /// Set if the data stored should be rendered at all.
    void setShouldRender(bool shouldRender);

    /// \return if the data should be rendered
    bool getShouldRender() const;

    /// Returns an exact copy of the render data as a new object
    RenderData* clone() const;

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::RenderData );
#endif

    protected:

      virtual ~RenderData();


      agx::Vec3Vector m_vertices;
      agx::UInt32Vector m_indices;
      agx::Vec3Vector m_normals;
      agx::Vec2Vector m_texCoords;
      agx::Vec4Vector m_vertexColor;
      agxCollide::RenderMaterialRef m_renderMaterial;
      agx::UInt32 m_renderState;
      Mode m_mode;

  };

}

