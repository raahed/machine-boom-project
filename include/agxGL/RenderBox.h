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
#ifndef AGXGL_RENDERBOX_H
#define AGXGL_RENDERBOX_H

#include <agxData/Type.h>
#include <agx/Vec3.h>

namespace agxGL
{
  #define AGX_NUM_RENDER_BOX_VERTICES (2 * 3 * 6)
  struct RenderBoxVertices
  {
    agx::Vec3f vertices[AGX_NUM_RENDER_BOX_VERTICES];
  };

  struct RenderBoxNormals
  {
    agx::Vec3f normals[AGX_NUM_RENDER_BOX_VERTICES];
  };

  struct RenderBoxIndices
  {
    agx::UInt32 indices[AGX_NUM_RENDER_BOX_VERTICES];
  };


  struct RenderBoxOutlines
  {
    #ifdef AGX_APPLE_IOS
    typedef agx::UInt16 IndexType;
    #else
    typedef agx::UInt32 IndexType;
    #endif

    IndexType indices[12 * 2];
  };


  AGX_FORCE_INLINE std::ostream& operator<<(std::ostream& output, const RenderBoxVertices& )
  {
    output << "{a render box}";
    return output;
  }

  AGX_FORCE_INLINE std::ostream& operator<<(std::ostream& output, const RenderBoxIndices& )
  {
    output<< "{a render box}";
    return output;
  }

  AGX_FORCE_INLINE std::ostream& operator<<(std::ostream& output, const RenderBoxNormals& )
  {
    output<< "{a render box}";
    return output;
  }

  AGX_FORCE_INLINE std::ostream& operator<<(std::ostream& output, const RenderBoxOutlines& )
  {
    output<< "{a render box outline}";
    return output;
  }
}

AGX_TYPE_BINDING(agxGL::RenderBoxVertices, "agxGL.RenderBoxVertices")
AGX_TYPE_BINDING(agxGL::RenderBoxNormals, "agxGL.RenderBoxNormals")
AGX_TYPE_BINDING(agxGL::RenderBoxOutlines, "agxGL.RenderBoxOutlines")


#endif /* _AGXGL_RENDERBOX_H_ */
