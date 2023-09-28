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
#include <agx/Vec4.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{

   AGX_DECLARE_POINTER_TYPES(RenderMaterial);

  /**
  Class for storing rendering material
 */
  class AGXPHYSICS_EXPORT RenderMaterial : public agx::Referenced, public agxStream::Serializable
  {
    public:

      enum TextureMode
      {
        DIFFUSE_TEXTURE=0,
        NORMALMAP_TEXTURE = 1,
        HEIGHTMAP_TEXTURE = 2,
        SPECULARMAP_TEXTURE = 3,
        NUM_TEXTURE_MODES=4
      };

    /// Constructor
    RenderMaterial();

    /// Set the name of this material
    void setName( const agx::String& name );

    /// \return the name of the material
    agx::String getName() const;

    /// Set the transparency of the material 1 - transparent, 0 - opaque
    void setTransparency( float transparency );

    /// \return true if transparency != -1
    bool hasTransparency() const;

    /// \return transparency of material
    float getTransparency() const;

    /// Set the shininess of the material
    void setShininess( float shininess );

    /// \return true if shininess != -1, which is default
    bool hasShininess() const;

    /// \return the shininess
    float getShininess() const;

    /// Set the diffuse color.
    void setDiffuseColor( const agx::Vec4f& color );

    /// \return true if the diffuse color is different from -1,-1,-1 (which is default)
    bool hasDiffuseColor() const;

    /// \return the current diffuse color
    agx::Vec4f getDiffuseColor() const;

    /// Set the specular color.
    void setSpecularColor( const agx::Vec4f& color );

    /// \return true if the diffuse color is different from -1,-1,-1 (which is default)
    bool hasSpecularColor() const;

    /// \return the current diffuse color
    agx::Vec4f getSpecularColor() const;

    /// Set the ambient color.
    void setAmbientColor( const agx::Vec4f& color );

    /// \return true if the ambient color is different from -1,-1,-1 (which is default)
    bool hasAmbientColor() const;

    /// \return the current ambient color
    agx::Vec4f getAmbientColor() const;

    /// Set the emissive color.
    void setEmissiveColor( const agx::Vec4f& color );

    /// \return true if the emissive color is different from -1,-1,-1 (which is default)
    bool hasEmissiveColor() const;

    /// \return the current emissive color
    agx::Vec4f getEmissiveColor() const;

    /// Store a string with a material specification, for example Keyshot material string.
    void setMaterialSpecification( const agx::String& material );

    /// \return a stored material specification
    agx::String getMaterialSpecification() const;

    /// \return true if a material specification is specified
    bool hasMaterialSpecification() const;


    /**
    Set the path to a texture for the specified texturing mode (default DIFFUSE)
    \param path - path to the texture
    \param mode - texturing mode
    */
    void setTexturePath( const agx::String& path,  TextureMode mode = DIFFUSE_TEXTURE);

    /**
    \returns the path to a texture for the specified texturing mode (default DIFFUSE)
    */
    agx::String getTexturePath(TextureMode mode = DIFFUSE_TEXTURE) const;

    /**
    \returns true if the specified mode has a texture specified
    */
    bool hasTexturePath(TextureMode mode = DIFFUSE_TEXTURE) const;

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::RenderMaterial );
#endif

      /**
      \param other - Material to compare to
      \returns true if the two materials are content wise equal to each other. Fast comparison
      */
      bool equal(const RenderMaterial& other) const;

      /**
      \returns true if the materials has equal value (uses equal method)
      */
      bool operator ==(const RenderMaterial& other) const;


      /**
      Slow string comparison function that uses the toString() method to convert each material to a string.
      \param other - Material to compare to
      \returns 0 if materials are equal, -1 if this is string wise larger, and 1 if other is string wise larger
      */
      int compare(const RenderMaterial& other) const;


      /**
      \return Hash value calculated from the string where all attributes are concatenated.
      */
      agx::UInt32 getHash() const;

      /**
      Returns a cloned copy of the RenderMaterial.
      */
      RenderMaterial* clone() const;

    protected:

      /// \return a concatenated string of all attributes of this class
      agx::String toString() const;

      virtual ~RenderMaterial();

      agx::String m_materialSpecification;
      agx::Vec4f m_diffuseColor;
      agx::Vec4f m_specularColor;
      agx::Vec4f m_ambientColor;
      agx::Vec4f m_emissiveColor;
      float m_transparency;
      float m_shininess;
      agx::String m_name;
      agx::StringVector m_texturePaths;
  };


  typedef agx::HashTable<agx::UInt32, RenderMaterialRef> RenderMaterialValueHashTable;
}

