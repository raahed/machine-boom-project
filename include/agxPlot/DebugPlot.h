/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXPLOT_DEBUGPLOT_H
#define AGXPLOT_DEBUGPLOT_H

#include <agx/config/AGX_USE_WEBPLOT.h>

#include <agx/Singleton.h>

#include <agxPlot/Curve.h>
#if AGX_USE_WEBPLOT()
#include <agxPlot/WebPlot.h>
#endif


namespace agxSDK
{
  class Simulation;
}


namespace agxPlot
{
  /**
  "Static" helper class to plot values from where e.g., values aren't exposed.
  It's basic so you have to know what your axes labels mean and to which window
  you're pushing values.
  */
  class AGXPHYSICS_EXPORT DebugPlot : public agx::Singleton
  {
    public:
      /**
      Behavior when pushing time data values since several
      calls can be made within same the time step.
      */
      enum TimeDataBehavior
      {
        ADD,       /**< Add previous and current. */
        USE_FIRST, /**< Use first value set - i.e., ignore all the rest of the values during this time step, to this curve. */
        USE_LAST   /**< Use the current value, i.e., write over any previous value written. */
      };

    public:
      /**
      \return true if an instance of this object exists for the given simulation
      */
      static agx::Bool hasInstance( agxSDK::Simulation* simulation );

      /**
      Add y-value to a curve, named \p curve, in a window named \p window, where the x-values are simulation time.
      \param yValue   - y-value
      \param curve    - name of the curve in \p window
      \param window   - name of the window
      \param behavior - behavior if several calls are made during the same time step
      */
      static void addTimeData( agx::Real yValue, const agx::String& curve, const agx::String& window = "", TimeDataBehavior behavior = USE_LAST );

      /**
      Add y-value to a curve, named \p curve, in a window named \p window, where the x-values are integers, i.e., number of samples.
      Use this method, rather than addTimeData, if you know you're writing many values each time step and wants to see them all.
      \param yValue - y-value
      \param curve  - name of the curve in \p window
      \param window - name of the window
      */
      static void add( agx::Real yValue, const agx::String& curve, const agx::String& window = "" );

      /**
      Add (x, y) to a curve with name \p curve in a window named \p window.
      \param xValue - x-value
      \param yValue - y-value
      \param curve  - name of the curve in \p window
      \param window - name of the window
      */
      static void add( agx::Real xValue, agx::Real yValue, const agx::String& curve, const agx::String& window = "" );

    public:
      SINGLETON_CLASSNAME_METHOD();

    protected:
      ~DebugPlot();

      virtual void shutdown() override;

    private:
      struct CurveData : public agx::Referenced
      {
        agxPlot::CurveRef curve;

        protected:
          virtual ~CurveData() {}
      };

      struct TimeCurveData : public CurveData
      {
        TimeCurveData() : m_written( true ), m_value( agx::Real( 0 ) ) {}

        agx::Real getValue() const { m_written = true; return m_value; }

        void set( agx::Real val, TimeDataBehavior behavior )
        {
          if ( !m_written ) {
            if ( behavior == TimeDataBehavior::ADD )
              m_value += val;
            else if ( behavior == TimeDataBehavior::USE_FIRST )
              ;
            else if ( behavior == TimeDataBehavior::USE_LAST )
              m_value = val;
          }
          else {
            m_value = val;
            m_written = false;
          }
        }

        protected:
          virtual ~TimeCurveData() {}

        private:
          mutable agx::Bool m_written;
          agx::Real m_value;
      };
      typedef agx::ref_ptr< TimeCurveData > TimeCurveDataRef;

      struct XYCurveData : public CurveData
      {
        agxPlot::DataSeriesRef xValues;
        agxPlot::DataSeriesRef yValues;

        void add( agx::Real x, agx::Real y )
        {
          xValues->push( x );
          yValues->push( y );
        }

        protected:
          virtual ~XYCurveData() {}
      };
      typedef agx::ref_ptr< XYCurveData > XYCurveDataRef;

      struct YCurveData : public XYCurveData
      {
        YCurveData() : m_counter(0) {}

        void add( agx::Real y )
        {
          XYCurveData::add( (agx::Real)m_counter++, y );
        }

        protected:
          virtual ~YCurveData() {}

        private:
          agx::UInt64 m_counter;
      };
      typedef agx::ref_ptr< YCurveData > YCurveDataRef;

    private:
      typedef agx::HashVector< agx::String, TimeCurveDataRef > TimeCurveTable;
      typedef agx::HashVector< agx::String, XYCurveDataRef > XYCurveTable;
      typedef agx::HashVector< agx::String, YCurveDataRef > YCurveTable;
      typedef agx::HashVector< agx::String, TimeCurveTable > TimeWindowTable;
      typedef agx::HashVector< agx::String, XYCurveTable > XYWindowTable;
      typedef agx::HashVector< agx::String, YCurveTable > YWindowTable;

    private:
      friend class agxSDK::Simulation;
      static DebugPlot* instance( agxSDK::Simulation* simulation );
      static agx::Bool tryInitialize();

    private:
      DebugPlot( agxSDK::Simulation* simulation );
      void clear( agx::Bool removeRefToSimulation );

      TimeCurveData* getOrCreateTimeCurveData( const agx::String& windowName, const agx::String& curveName );
      XYCurveData* getOrCreateXYCurveData( const agx::String& windowName, const agx::String& curveName );
      YCurveData* getOrCreateYCurveData( const agx::String& windowName, const agx::String& curveName );

    private:
#if AGX_USE_WEBPLOT()
      agxPlot::WebPlotRef m_webPlot;
#endif
      TimeWindowTable m_timeCurveWindows;
      XYWindowTable m_xyCurveWindows;
      YWindowTable m_yCurveWindows;
      agxSDK::Simulation* m_simulation;

    private:
      static DebugPlot* s_instance;
  };
}

#endif
