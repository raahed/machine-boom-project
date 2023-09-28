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

#ifndef AGXPLOT_CURVE_H
#define AGXPLOT_CURVE_H

#include <agxPlot/DataSeries.h>

#include <agx/String.h>
#include <agx/Vec4.h>
#include <agx/Vec2.h>

namespace agxPlot
{
  typedef agx::Vec2 CurvePoint;
  typedef agx::Vector<CurvePoint> CurvePointVector;
  typedef agx::UInt CurveID;

  class System;

  class AGXPHYSICS_EXPORT Curve : public agx::Referenced
  {
    public:
      enum Line
      {
        LINE_SOLID = 0,
        LINE_DASHED = 1,
        LINE_NONE = 2
      };
      enum Symbol
      {
        SYMBOL_CROSS = 0,
        SYMBOL_CIRCLE = 1,
        SYMBOl_DOT = 2,
        SYMBOl_NONE = 3
      };
      Curve(agxPlot::DataSeries* xValues, agxPlot::DataSeries* yValues, const agx::String& name = "");


      /**
      Get all currently generated values from the two DataSeries.
      \return a list of all plot points in the curve
      */
      CurvePointVector getAllValues() const;

      /**
        Get all values from the DataSeries that have been generated since the last update call.
        \return a list of all new plot points in the curve
        */
      CurvePointVector getNewValues() const;

      /**
      Get all values from the DataSeries that have been generated before the last update call.
      \return a list of all old plot points in the curve
      */
      CurvePointVector getOldValues() const;

      /**
       The name/legend of the curve
      */
      void setName(const agx::String& name);
      const agx::String& getName() const;

      /**
       The color of the curve.
      */
      void setColor(const agx::Vec4& color);
      const agx::Vec4& getColor() const;

      /**
       The type of line used.
      */
      void setLineType(Line line);
      Line getLineType() const;

      /**
       The type of symbol used.
      */
      void setSymbol(Symbol symbol);
      Symbol getSymbol() const;

      /**
       The width of the line in pixels.
      */
      void setLineWidth(agx::Real lineWidth);
      agx::Real getLineWidth() const;

      /**
        A generated ID for the curve. No two curves should have the same ID.
      */
      CurveID getCurveID() const;

      /**
       Retrieves the description of the curve

       ["name"] = the name of the curve
       ["id"] = the unique id of the curve
       ["color_r"] = the red value of the line color
       ["color_g"] = the green value of the line color
       ["color_b"] = the blue value of the line color
       ["color_a"] = the alpha value of the line color
       ["lineType"] = an int describing the type of line
       ["lineWidth"] = the width of the line in pixels
       ["symbol"] = an int describing the type of symbol
       ["xAxis"] = the description of the X axis
       ["yAxis"] = the description of the Y axis
      */
      void getDescription(agxJson::Value& root) const;
      /**
       Sets the curve to match that of a description of
       the format from getDescription.
      */
      // Send system to be able to fetch DataSeries by ID
      void setFromDescription(const agxJson::Value& root, System* system);

      // INTERNAL

      agxPlot::DataSeries* getDataSeriesX() const;
      agxPlot::DataSeries* getDataSeriesY() const;

      agx::Event representationChanged;
    private:
      CurvePointVector getValuesFrom(size_t index) const;
      CurvePointVector getValuesTo(size_t index) const;

      void setID(CurveID id);
    private:
      agxPlot::DataSeriesRef m_xValues;
      agxPlot::DataSeriesRef m_yValues;

      agx::Vec4 m_color;
      agx::String m_name;
      Line m_lineType;
      Symbol m_symbol;
      agx::Real m_lineWidth;
      CurveID m_id;

      agx::Event::CallbackType m_dataSeriesRepresentationChangedCallback;
  };
  AGX_DECLARE_POINTER_TYPES(Curve);
  AGX_DECLARE_VECTOR_TYPES(Curve);
}

#endif