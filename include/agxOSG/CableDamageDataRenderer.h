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

#include <agx/Real.h>
#include <agx/Vector.h>

#include <agxOSG/export.h>

#include <agxSDK/StepEventListener.h>

#include <functional>

namespace osg
{
  class Group;
}


namespace agxCable
{
  class Cable;
  class CableDamageState;
  class SegmentDamageState;
}



namespace agxOSG
{
  class GeometryNode;
  class Group;
  class ExampleApplication;
}



namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(CableDamageDataRenderer);

  class AGXOSG_EXPORT CableDamageDataRenderer : public agxSDK::StepEventListener
  {
    public:
      CableDamageDataRenderer(agxCable::Cable* cable, agxCable::CableDamageState* damage, osg::Group* root,
                              agxOSG::ExampleApplication& application);


      void setDamageGetter(std::string name, std::function<agx::Real(const agxCable::SegmentDamageState*)> f);

      virtual void post(const agx::Real& timeStamp) override;

      void updateColors();

      void updateText();

    protected:
      virtual ~CableDamageDataRenderer() {}

    private:
      CableDamageDataRenderer& operator=(CableDamageDataRenderer&) = delete;

    private:
      agxCable::CableDamageState* m_damage;
      agxOSG::ExampleApplication& m_application;
      std::function<agx::Real(const agxCable::SegmentDamageState*)> m_getDamage;
      std::string m_getDamageName;
      agx::Vector<agxOSG::GeometryNode*> m_nodes;
  };
}
