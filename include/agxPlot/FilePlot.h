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

#ifndef AGXPLOT_FILEPLOT_H
#define AGXPLOT_FILEPLOT_H

#include <agxPlot/Output.h>

#include <fstream>
#include <map>
#include <memory>

namespace agxPlot
{

  class AGXPHYSICS_EXPORT FilePlot : public agxPlot::Output
  {
    public:
      FilePlot(const agx::String& fileName);

      virtual void closeOutput();
      virtual void handlePacket(DataPacket *packet);
      virtual void handlePacket(DescriptionPacket *packet);
      virtual void handlePacket(TimePacket* packet);

    protected:
      virtual ~FilePlot();
    private:
      bool m_closed;
      agx::String m_outputFile;
      std::map<CurveID, agx::String>                   m_subFiles;
      std::map<CurveID, std::shared_ptr<std::ofstream>> m_fileOutStreams;
  };
}

#endif
