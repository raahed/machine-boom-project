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

#ifndef AGXOSG_STATISTICSRENDERER_H
#define AGXOSG_STATISTICSRENDERER_H

#include <agx/Referenced.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgText/Text>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/SceneDecorator.h>

namespace agxOSG
{
  class AGXOSG_EXPORT StatisticsRenderer : public agx::Referenced
  {
  public:
    StatisticsRenderer(SceneDecorator* decorator);

    void registerEntry(const std::string& module, const std::string& data);
    void unregisterEntry(const std::string& module, const std::string& data);
    void clearEntries();

    // void setPosition(const agx::Vec2& position);

    void update();

  protected:
    virtual ~StatisticsRenderer() {}

  private:
    agx::Vec2 m_position;
    agx::Vector<std::pair<std::string, std::string> > m_entries;
    osg::ref_ptr<osgText::Text> m_textField;
    osg::ref_ptr<SceneDecorator> m_decorator;
  };

  typedef agx::ref_ptr<StatisticsRenderer> StatisticsRendererRef;
}

#endif /* _AGXOSG_STATISTICSRENDERER_H_ */
