/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#ifndef AGXOSG_PRESSURE_TO_COLOR_CONVERTER_H
#define AGXOSG_PRESSURE_TO_COLOR_CONVERTER_H


#include <agx/macros.h>
#include <agx/Referenced.h>

#include <agxOSG/export.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Image>
#include <osgSim/ColorRange>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.
namespace agxOSG
{

  class PressureAtlas;

  AGX_DECLARE_POINTER_TYPES(PressureToColorConverter);
  AGX_DECLARE_VECTOR_TYPES(PressureToColorConverter);

  class AGXOSG_EXPORT PressureToColorConverter : public agx::Referenced
  {
  public:
    PressureToColorConverter(agxOSG::PressureAtlas* atlas);
    virtual void convert(bool dirty) = 0;

    osg::Image* getPressureMap();
    osg::Image* getColorMap();

  protected:
    virtual ~PressureToColorConverter() {};

  protected:
    osg::ref_ptr<osg::Image> m_pressureMap;
    osg::ref_ptr<osg::Image> m_colorMap;
  };


  AGX_DECLARE_POINTER_TYPES(PressureToRgbaConverter);

  class AGXOSG_EXPORT PressureToRgbaConverter : public PressureToColorConverter
  {
  public:
    PressureToRgbaConverter(PressureAtlas* atlas, osgSim::ColorRange* colorRange);
    virtual void convert(bool dirty) override;

  protected:
    virtual ~PressureToRgbaConverter() {}

  private:
    osg::ref_ptr<osgSim::ColorRange> m_colorRange;
  };


  AGX_DECLARE_POINTER_TYPES(GatherMaxPressure);

  class AGXOSG_EXPORT GatherMaxPressure : public PressureToRgbaConverter
  {
  public:
    GatherMaxPressure(PressureAtlas* atlas, osgSim::ColorRange* colorRange);
    virtual void convert(bool) override;

    void convertToColorImage();

    void clear();

  protected:
    virtual ~GatherMaxPressure() {}

  private:
    osg::ref_ptr<osg::Image> m_pressureMap;
    osg::ref_ptr<osg::Image> m_maxPressure;
  };


  AGX_DECLARE_POINTER_TYPES(GatherAvgPressure);

  class AGXOSG_EXPORT GatherAvgPressure : public PressureToRgbaConverter
  {
  public:
    GatherAvgPressure(PressureAtlas* atlas, osgSim::ColorRange* colorRange);
    virtual void convert(bool dirty) override;

    void convertToColorImage();

    void clear();

  protected:
    virtual ~GatherAvgPressure();

  private:
    osg::ref_ptr<osg::Image> m_pressureMap;
    double* m_totalPressure;
    osg::ref_ptr<osg::Image> m_avgPressure;
    size_t m_numSamples;
  };

}
#endif
