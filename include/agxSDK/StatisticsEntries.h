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

#ifndef AGXSDK_STATISTICS_ENTRIES
#define AGXSDK_STATISTICS_ENTRIES

#include <agx/Statistics.h>
#include <agxRender/Graph.h>

namespace agxSDK
{
  class StatisticsEntries
  {
  public:
    StatisticsEntries();
    void init();

    void registerEntry(const agx::String& title, const agx::String& module, const agx::String& data, const agx::Vec4& color, bool optional = false);
    void registerGraphEntry(const agx::String& module, const agx::String& data, const agx::String& title, const agx::Vec4& color,
      float minVal, float maxVal, float scaleX, float scaleY);

    void unregisterEntry(const agx::String& module, const agx::String& data);
    void unregisterGraphEntry(const agx::String& module, const agx::String& data);

    void render(Simulation* simulation);
    ~StatisticsEntries() {}

    void reset();

    bool firstRun();

  protected:

  private:
    struct Entry {
      Entry() : optional(false), m_value(0) {}
      Entry(const agx::String& t, const agx::String& m, const agx::String& d, const agx::Vec4& c, bool o = false) :  color(c), module(m), data(d), title(t), optional(o), m_value(0) {}

      agx::Vec4 color;
      agx::String module;
      agx::String data;
      agx::String title;
      bool optional;

      agx::Statistics::AbstractDataObserver m_value;

      bool operator==(const Entry& other) {
        return (module == other.module && data == other.data );
      }
    };
    agx::Vector<Entry> m_entries;

    struct GraphEntry {
      GraphEntry() : m_value(0) {}
      GraphEntry(const agx::String& m, const agx::String& d, agxRender::Graph::Channel* c) : module(m), data(d), channel(c) {}
      agx::String module;
      agx::String data;


      agx::Statistics::AbstractDataObserver m_value;
      agx::ref_ptr<agxRender::Graph::Channel> channel;

      bool operator==(const GraphEntry& other) {
        return (module == other.module && data == other.data );
      }
    };
    agx::Vector<GraphEntry> m_graphEntries;

    bool m_firstRun;
    agx::TimeStamp m_prevStamp;
  };


}
#endif
