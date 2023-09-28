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

#ifndef AGXOSG_RENDERTEXT_H
#define AGXOSG_RENDERTEXT_H

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Geometry>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Vec2.h>
#include <string>

namespace agxOSG {

  /// Render the bitmap text using previously created bitmap fonts.
  //void AGXOSG_EXPORT renderText(int x, int y, const char *s);
  void AGXOSG_EXPORT renderText(float x, float y, const char *s);
  unsigned int AGXOSG_EXPORT makeRasterFont(void);


  /// Class for drawing simple/fast 2D text onto screen
  class AGXOSG_EXPORT TextGeometry : public osg::Geometry
  {
  public:

    TextGeometry();

    /// Set the position of the text
    void setPosition( const agx::Vec2& pos );

    /// Set the text string
    void setText( const std::string& text );

  protected:

    virtual ~TextGeometry() {}

    class TextDrawCallback : public virtual osg::Drawable::DrawCallback
    {
    public:
      TextDrawCallback() {}

      void drawImplementation(osg::RenderInfo& /*renderInfo*/,const osg::Drawable* /*drawable*/) const
      {
        agxOSG::renderText((float)m_pos[0], (float)m_pos[1] ,m_text.c_str());
      }

      void setPosition( const agx::Vec2& pos ) { m_pos = pos; }
      void setText( const std::string& text ) { m_text = text; }

      agx::Vec2 m_pos;
      std::string m_text;
    };


    TextDrawCallback *m_textDrawCallback;
  };

}

#endif
