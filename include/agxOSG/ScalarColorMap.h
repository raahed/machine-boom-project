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

#pragma once

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <osg/ref_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agxRender/Color.h>
#include <agx/Referenced.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES( ScalarColorMap );
  /**
  Class that represents a linear scalar color map that converts a scalar interval to
  a specified color range. This wraps the osgSim::ColorRange object with a layer of
  convenience functionality.
  */
  class AGXOSG_EXPORT ScalarColorMap : public agx::Referenced
  {
  public:

    /**
    Enum representation of predefined color ranges than can be set in the class.
    */
    enum ColorRangePreset
    {
      RAINBOW,
      COOLWARM,
      VIRIDIS,
      INFERNO,
      PLASMA,
      GRAYSCALE,
      BLUE_WHITE_RED,
      CIVIDIS,
      BONE,
      COPPER,
      AFMHOT
    };

    /**
    Default constructor
    */
    ScalarColorMap();

    /**
    Constructor
    \param colorVector - the specific color range to be used in scalar map, which is mapped to the
                         scalar interval.
    \param minScalar - The minimum value of the scalar interval that is mapped to the color range.
    \param maxScalar - The maximum value of the scalar interval that is mapped to the color range.
    \note - if the specified minimum scalar larger than the current maximum value or
            vice versa, the values will be switched.
    */
    ScalarColorMap( const agxRender::ColorVector& colorVector,
                    agx::Real minScalar,
                    agx::Real maxScalar );

    /**
    Constructor
    \param preset - preset enum for one of the predefined color ranges that exist in the class that can
                    be chosen to initialize the colors in the map.
    \param minScalar - The minimum value of the scalar interval that is mapped to the color range.
    \param maxScalar - The maximum value of the scalar interval that is mapped to the color range.
    \note - if the specified minimum scalar larger than the current maximum value or
            vice versa, the values will be switched.
    */
    ScalarColorMap( ColorRangePreset preset,
                    agx::Real minScalar,
                    agx::Real maxScalar );

    /**
    Set a range of colors from an available color range preset in the class.
    \param colorRangePreset - the specified preset enum that is used to construct the color
                              range.
    */
    void setColorsFromPreset( ColorRangePreset colorRangePreset );

    /**
    Get an osg representation of a color value in the map given a scalar inside the interval.
    \param scalar - a specified scalar that is used to interpolate a color in the given range.
    \param modifyAlpha - true if the alpha value of the color should be modified, false otherwise.
    \return an osg::Vec4 color representation given the specified scalar value.
    */
    osg::Vec4 getColorOSG( agx::Real scalar, bool modifyAlpha = false ) const;

    /**
    Get an agx representation of a color value in the map given a scalar inside the interval.
    \param scalar - a specified scalar that is used to interpolate a color in the given range.
    \param modifyAlpha - true if the alpha value of the color should be modified, false otherwise.
    \return an agx::Vec4f color representation given the specified scalar value.
    */
    agx::Vec4f getColor( agx::Real scalar, bool modifyAlpha = false ) const;

    /**
    Set the min scalar of the interval.
    \param minScalar - the minimum scalar value in the interval.
    \note - the specified minimum scalar will be rejected if the value is
            larger than the current maximum value.
    */
    void setMinScalar( agx::Real minScalar );

    /**
    Set the maximum scalar of the interval.
    \param maxScalar - the maximum scalar value in the interval.
    \note - the specified maximum scalar will be rejected if the value is
            less than the current minimum value.
    */
    void setMaxScalar( agx::Real maxScalar );

    /**
    Set the scalars of the interval used in the color interpolation.
    \param minScalar - the minimum scalar value in the interval.
    \param maxScalar - the maximum scalar value in the interval.
    \note - if the specified minimum scalar larger than the current maximum value or
            vice versa, the values will be switched.
    */
    void setScalarInterval( agx::Real minScalar, agx::Real maxScalar );

    /*
    Manually set the colors used in the interpolation range.
    \param colorVector - a vector containing the specified colors of the interval.
    */
    void setColors( const agxRender::ColorVector& colorVector );

    /**
    \return the minimum scalar value of the interval.
    */
    agx::Real getMinScalar() const;

    /**
    \return the maximum scalar value of the interval.
    */
    agx::Real getMaxScalar() const;

    /**
    \return vector containing colors used the current interval.
    */
    agxRender::ColorVector getColors() const;

    /**
    \returns a pointer to the internal osgSim::ColorRange object.
    */
    osg::ref_ptr < osgSim::ColorRange > getOsgColorRange() const;

  protected:
    virtual ~ScalarColorMap();
    void initColorRange( const std::vector<osg::Vec4>& osgColors );
    void updateColorRange();
    void correctScalarInterval();
    agxRender::ColorVector getColorsFromPreset( ColorRangePreset preset );

  protected:
    agx::Real                          m_minScalar;
    agx::Real                          m_maxScalar;
    float                              m_rScaling;
    osg::ref_ptr< osgSim::ColorRange > m_colorRange;
  };
}
