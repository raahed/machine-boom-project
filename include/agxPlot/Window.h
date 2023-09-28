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

#ifndef AGXPLOT_WINDOW_H
#define AGXPLOT_WINDOW_H

#include <agxPlot/Curve.h>
#include <agxPlot/DataPacket.h>

#include <agx/String.h>

namespace agxPlot
{
  class System;

  class AGXPHYSICS_EXPORT Window : public agx::Referenced
  {
    public:
      /**
       \param id The unique ID of the Window. Per default the
                 name will also be this, but it can be changed.
                 The ID is never changed though.
      */
      Window(const agx::String& id = "");

      /**
       Add the specified curve to the plot.
       \param curve The curve to add. Should not be nullptr.
          Also, getDataSeriesX and getDataSeriesY should be non-nullptr.
       \retval Was the curve added successfully?
       */
      bool add(Curve* curve);

      void clearCurves();

      /**
       Get all values that have been added since the last data push.
       Adds it to the end of the of the DataPacket
       \param target Where the data should be added
       */
      void getNewValues(DataPacket* target);

      /**
      Get all values that have been added since the last data push.
      Adds it to the end of the of the DataPacket
      \param target Where the data should be added
      */
      void getOldValues(DataPacket* target);

      System* getPlotSystem();

      /**
       Retrieve the description, as follows

       ["name"] = the name of the window
       ["curves"] = an array of descriptions of
                    all curves owned by the window
      */
      void getDescription(agxJson::Value& root) const;

      /**
       set from an description of the format of getDescription.
      */
      void setFromDescription(const agxJson::Value& root);

      const agx::String& getId() const;

      /**
       Get the name of the Window.
      */
      const agx::String& getName() const;
      /**
       Set the name of the Window
      */
      void setName(const agx::String& name);

      void setEnabled(bool enabled);

      bool getEnabled() const;

      /// Gets the curves belonging to the window.
      const CurveRefVector& getCurves() const;

      agx::Event representationChanged;
    public:
      /***************************************/
      /* HELPER FUNCTIONS FOR CREATING PLOTS */
      /***************************************/
    private:
      friend class System;
      void setPlotSystem(System *system);

    private:
      CurveRefVector m_curves;
      System *m_system;
      agx::String m_id;
      agx::String m_name;
      bool m_isEnabled;

      agx::Event::CallbackType m_curveRepresentationChangedCallback;
  };

  AGX_DECLARE_POINTER_TYPES(Window);
  AGX_DECLARE_VECTOR_TYPES(Window);
}

#endif