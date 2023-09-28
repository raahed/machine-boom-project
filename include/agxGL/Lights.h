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

#ifndef AGXGL_LIGHTS_H
#define AGXGL_LIGHTS_H

#include <agx/agxPhysics_export.h>
#include <agx/Object.h>
#include <agx/Vec3.h>
#include <agx/Clock.h>

namespace agxGL
{
  /**
  Container class for lights that is used for synchronizing light data with shaders, such as particle sprite shaders.
  */
  class AGXPHYSICS_EXPORT Lights : public agx::Component
  {
  public:
    static agx::Model *ClassModel();

    static Lights* load(agx::TiXmlElement* eLights, agx::Device* device);
    virtual void configure(agx::TiXmlElement* eLights) override;

  public:

    Lights(const agx::Name& name = "lights");

    bool setLightPosition(agx::UInt index, const agx::Vec4& position);

    bool setLightDirection(agx::UInt index, const agx::Vec3& direction);

  protected:
    virtual ~Lights();

  protected:
    agx::Vector<agxData::ValueRefT<agx::Vec4>> m_light_pos;
    agx::Vector<agxData::ValueRefT<agx::Vec3>> m_light_dir;
  };

  AGX_DECLARE_POINTER_TYPES(Lights);
}

#endif /*AGXGL_LIGHTS_H*/
