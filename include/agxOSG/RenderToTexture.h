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

#ifndef AGXOSG_RENDERTOTEXTURE_H
#define AGXOSG_RENDERTOTEXTURE_H

#include <agxOSG/export.h>
#include <agxOSG/RenderTarget.h>
#include <agxOSG/Node.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(RenderToTexture);
  AGX_DECLARE_VECTOR_TYPES(RenderToTexture);

  class AGXOSG_EXPORT RenderToTexture : public agxOSG::RenderTarget
  {
    public:
      RenderToTexture(agx::UInt width, agx::UInt height, BufferComponent bufferComponent = COLOR_BUFFER, agx::UInt multiSamples = 8, RenderTarget::TextureFormat textureFormat = RenderTarget::RGBA);

      agxOSG::Texture2D* getTexture();

    protected:
      virtual ~RenderToTexture();

    private:
      agx::ref_ptr<agxOSG::Texture2D> m_texture;
  };
}

#endif /* AGXOSG_RENDERTOTEXTURE_H */
