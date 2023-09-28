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

#ifndef AGXPLOT_OUTPUT_H
#define AGXPLOT_OUTPUT_H

#include <agxPlot/Queue.h>
#include <agxPlot/DataPacket.h>
#include <agxPlot/DescriptionPacket.h>
#include <agxPlot/TimePacket.h>

namespace agxPlot
{
  class System;
  AGX_DECLARE_POINTER_TYPES(System);
  AGX_DECLARE_POINTER_TYPES(Output);
  AGX_DECLARE_VECTOR_TYPES(Output);

  class AGXPHYSICS_EXPORT Output : public agx::Referenced
  {
    public:
      Output();

      agxPlot::System *getSystem();
      agxPlot::Queue* getQueue();

      virtual void handlePacket(DataPacket *packet) = 0;
      /**
       The description is as follows:

       root is the description of a agxPlot::System

       agxPlot::System description:

       ["windows"] = an array of all agxPlot::Window descriptions.
       ["data"] = an array of tuples. Contains ["id"] with the data id
       and ["values"] which is an array of real values. Optional field

       agxPlot::Window description:

       ["name"] = the name of the window
       ["curves"] = an array of descriptions of
       all agxPlot::Curve s owned by the window

       agxPlot::Curve description:

       ["name"] = the name of the curve
       ["id"] = the unique id of the curve
       ["color_r"] = the red value of the line color
       ["color_g"] = the green value of the line color
       ["color_b"] = the blue value of the line color
       ["color_a"] = the alpha value of the line color
       ["lineType"] = an int describing the type of line
       ["lineWidth"] = the width of the line in pixels
       ["symbol"] = an int describing the type of symbol
       ["xAxis"] = the description of the X axis DataSeries
       ["yAxis"] = the description of the Y axis DataSeries

       agxPlot::DataSeries description:

       ["name"] = the name of the data series (axis)
       ["unit"] = the unit of the data series
       ["isLogarithmic"] = should the data series be viewed as logarithmic
       ["isTime"] = the data series represents time
       ["id"] = the id of the data series
       */
      virtual void handlePacket(DescriptionPacket *packet) = 0;
      virtual void handlePacket(TimePacket* packet) = 0;
      virtual void closeOutput() = 0;

      // TODO: Refactor
      virtual bool isReady() { return true; };

    protected:
      virtual ~Output();

    private:
      friend class System;
      /**
      Set the System that the output should get data from.
      \param system The system to attach the output to.
      */
      void setSystem(System *system);

      void update();


    protected:
      System *m_system;
      QueueRef m_queue;
  };
}

#endif
