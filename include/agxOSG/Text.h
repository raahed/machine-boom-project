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

#ifndef AGXOSG_TEXT_H
#define AGXOSG_TEXT_H

#include <agxOSG/export.h>
#include <agxOSG/Node.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgText/Text>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  class AGXOSG_EXPORT Text : public osg::Geode
  {
    public:

      enum AxisAlignment
      {
        XY_PLANE,
        REVERSED_XY_PLANE,
        XZ_PLANE,
        REVERSED_XZ_PLANE,
        YZ_PLANE,
        REVERSED_YZ_PLANE,
        SCREEN,
        USER_DEFINED_ROTATION
      };

      enum BackdropType
      {
        DROP_SHADOW_BOTTOM_RIGHT = 0,  // usually the type of shadow you see
        DROP_SHADOW_CENTER_RIGHT,
        DROP_SHADOW_TOP_RIGHT,
        DROP_SHADOW_BOTTOM_CENTER,
        DROP_SHADOW_TOP_CENTER,
        DROP_SHADOW_BOTTOM_LEFT,
        DROP_SHADOW_CENTER_LEFT,
        DROP_SHADOW_TOP_LEFT,
        OUTLINE,
        NONE
      };

      enum AlignmentType
      {
        LEFT_TOP,
        LEFT_CENTER,
        LEFT_BOTTOM,

        CENTER_TOP,
        CENTER_CENTER,
        CENTER_BOTTOM,

        RIGHT_TOP,
        RIGHT_CENTER,
        RIGHT_BOTTOM,

        LEFT_BASE_LINE,
        CENTER_BASE_LINE,
        RIGHT_BASE_LINE,

        LEFT_BOTTOM_BASE_LINE,
        CENTER_BOTTOM_BASE_LINE,
        RIGHT_BOTTOM_BASE_LINE,

        BASE_LINE = LEFT_BASE_LINE ///< default.

      };


      enum DrawModeMask
      {
        TEXT              = 1, ///< default
        BOUNDINGBOX       = 2,
        FILLEDBOUNDINGBOX = 4,
        ALIGNMENT         = 8
      };

      Text();
      Text( const agx::String& text, const agx::Vec3& position=agx::Vec3(), const agx::Vec4& color = agx::Vec4( 1, 1, 1, 1 ) );


      void setFont( const agx::String& font );
      void setCharacterSize( float size );
      void setColor( const agx::Vec4f& color );
      void setPosition( const agx::Vec3& position );
      void setAxisAlignment( AxisAlignment alignment );
      void setBackdropType( BackdropType type );
      void setDrawMode( DrawModeMask drawMode );
      void setText( const agx::String& text );
      void setAlignment(AlignmentType alignment);

  private:
    void init(const agx::String& text="", const agx::Vec3& position=agx::Vec3(), const agx::Vec4& color = agx::Vec4( 1, 1, 1, 1 ) );
    osg::ref_ptr<osgText::Text> m_text;
  };



}

#endif
