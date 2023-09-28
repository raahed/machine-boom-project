/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#ifndef AGXOSG_TEXTURE_ATLAS_BUILDER_H
#define AGXOSG_TEXTURE_ATLAS_BUILDER_H

#include <agx/debug.h>
#include <agxCollide/Trimesh.h>

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Image>
#include <osg/Array>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  class AGXOSG_EXPORT TextureAtlasBuilder
  {
  public:
    virtual osg::ref_ptr<osg::Image> createAtlas(const agxCollide::Trimesh* mesh, osg::Vec2Array& textureCoordinates, agx::Real texelsPerMeter) = 0;

    class AtlasCreationFailure : public agx::Error
    {
    public:
      AtlasCreationFailure(const agxCollide::Trimesh* mesh, const std::string& what);
      const agxCollide::Mesh* mesh;
    };
  };


  class AGXOSG_EXPORT OsgTextureAtlasBuilder : public TextureAtlasBuilder
  {
  public:
    virtual osg::ref_ptr<osg::Image> createAtlas(const agxCollide::Trimesh* mesh, osg::Vec2Array& textureCoordinates, agx::Real texelsPerMeter) override;

    class AttemptsExhausted : public AtlasCreationFailure
    {
    public:
      AttemptsExhausted(const agxCollide::Trimesh* mesh, int atlasSizeWidth, int atlasSizeHeight, unsigned attempts);
      int atlasSizeWidth;
      int atlasSizeHeight;
      unsigned attempts;
    };
  };
}

#endif

