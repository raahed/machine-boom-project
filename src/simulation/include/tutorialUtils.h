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

#include <agxOSG/utils.h>

#include <agxSDK/StepEventListener.h>
#include <agxSDK/Simulation.h>
#include <agx/version.h>


#include <agxCollide/Box.h>
#include <agxCollide/Cylinder.h>
#include <agxCollide/Sphere.h>
#include <agxCollide/HeightField.h>
#include <agxCollide/Capsule.h>

#include <agxRender/RenderManager.h>


namespace agxOSG
{
  class TextEventListener : public agxSDK::StepEventListener
  {
    public:
      TextEventListener( const agx::String& str, const agx::Vec2& pos, const agx::Vec3& color=agx::Vec3(1,1,1) )
        : m_pos(pos), m_color(color), m_string(str)
      {
        setMask( PRE_STEP );
      }

      void pre(const agx::TimeStamp& )
      {
        agxRender::RenderProxy *text = getSimulation()->getRenderManager()->acquireText(m_string.c_str(), agx::Vec3(m_pos[0], m_pos[1],0));
        text->setColor( m_color );
      }

      void setText( const agx::String& text )
      {
        m_string = text;
      }

      agx::String getText() const
      {
        return m_string;
      }

    private:
      agx::Vec2 m_pos;
      agx::Vec3 m_color;
      agx::String m_string;
  };
}

