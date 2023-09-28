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

#ifndef AGXIO_IMAGEVALUEINTERPRETER_H
#define AGXIO_IMAGEVALUEINTERPRETER_H

#include <agxIO/Image.h>
#include <agx/Vec4.h>

namespace agxIO
{
  typedef float (*GetValueFunction)( float val );
  typedef float (*ReadValue01Function)( agxIO::Image* image, size_t x, size_t y );

  inline float linearValueFunction( float val )
  {
    return val;
  }

  inline float quadraticValueFunction( float val )
  {
    return val*val;
  }

  inline float cubicValueFunction( float val )
  {
    return val*val*val;
  }

  inline float readOneOrTwoChannelValue01( agxIO::Image* image, size_t x, size_t y )
  {
    size_t pos = (size_t)image->getWidth() * y + x;
    pos *= (size_t)image->getChannels();
    return 0.00392156863f * image->getData()[ pos ]; // unsigned char / 255.0.
  }

  inline float readThreeChannelMergedValue01( agxIO::Image* image, size_t x, size_t y )
  {
    const unsigned char* data = image->getData();
    size_t pos = (size_t)image->getWidth() * y + x;
    pos *= 3;
    return 0.00392156863f * float(( 77 * data[ pos ] + 150 * data[ pos + 1 ] + 29 * data[ pos + 2 ] ) >> 8);
  }

  inline float readFourChannelMergedValue01( agxIO::Image* image, size_t x, size_t y )
  {
    const unsigned char* data = image->getData();
    size_t pos = (size_t)image->getWidth() * y + x;
    pos *= 4;
    return 0.00392156863f * float(( 77 * data[ pos ] + 150 * data[ pos + 1 ] + 29 * data[ pos + 2 ] ) >> 8);
  }

  inline float readSingleChannelValue01( agxIO::Image* image, size_t x, size_t y, size_t channel )
  {
    const unsigned char* data = image->getData();
    size_t pos = (size_t)image->getWidth() * y + x;
    pos *= (size_t)image->getChannels();
    return 0.00392156863f * data[ pos + channel ];
  }

  inline void writeSingleChannelValue01( unsigned char val, agxIO::Image* image, size_t x, size_t y, size_t channel )
  {
    unsigned char* data = image->getData();
    size_t pos = (size_t)image->getWidth() * y + x;
    pos *= (size_t)image->getChannels();
    data[ pos + channel ] = val;
  }

  class AGXPHYSICS_EXPORT PNGImageValueInterpreter : public agx::Referenced
  {
    public:
      /**
      Default constructor. Minimum value set to 0 and maximum value to 1.
      */
      PNGImageValueInterpreter();

      /**
      Construct given minimum- and maximum value.
      \param minValue - minimum value when image channel value is zero
      \param maxValue - maximum value when image channel value is 255
      */
      PNGImageValueInterpreter( float minValue, float maxValue );

      /**
      Read PNG image. Searches agx paths.
      \return true if load was successful - otherwise false (file not found or unsupported PNG)
      */
      bool read( const std::string& filename );

      /**
      \return true if the image is valid
      */
      bool valid() const;

      /**
      Set new maximum value for the accessors methods (i.e., channel value = 255 => maximum value).
      \param maxValue - maximum value
      */
      void setMaxValue( float maxValue );

      /**
      Set new minimum value for the accessors methods (i.e., channel value = 0 => minimum value).
      \param minValue - minimum value
      */
      void setMinValue( float minValue );

      /**
      \return the maximum value used by the accessors methods (when channel value = 255)
      */
      float getMaxValue() const;

      /**
      \return the minimum value used by the accessors methods (when channel value = 0)
      */
      float getMinValue() const;

      /**
      \return the PNG image if initialized
      */
      agxIO::Image* getImage() const;

      /**
      \return the height of the image (number of pixels)
      */
      size_t getImageHeight() const;

      /**
      \return the width of the image (number of pixels)
      */
      size_t getImageWidth() const;

      /**
      Reads image at pixel (x,y) and returns value between given minimum and maximum. Alpha channel is
      ignored for any PNG with alpha (i.e., 2 or 4). For 3 and 4 channels the RGB channels are merged
      to a gray scale single value and passed to GetValueFunction \p func.
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return value between given minimum and maximum
      */
      float getValue( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Reads single "Red" channel at pixel (x,y) and returns value between given minimum and maximum.
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return value between given minimum and maximum
      */
      float getValueR( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Reads single "Green" channel at pixel (x,y) and returns value between given minimum and maximum. If the PNG has two
      channels, the alpha channel will be used here (i.e., same result as calling getValueA).
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return value between given minimum and maximum
      */
      float getValueG( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Reads single "Blue" channel at pixel (x,y) and returns value between given minimum and maximum. The PNG has to have at least
      three channels for valid calls to this method.
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return value between given minimum and maximum
      */
      float getValueB( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Reads single "Alpha" channel at pixel (x,y) and returns value between given minimum and maximum. The PNG has to have either two
      or four channels for the value to be valid. E.g., if three channels, value of G will be calculated (offset = numChannels - 1).
      Asserts if number of channels is 1 or 3, but make sure you have the correct number of channels when calling this method.
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return value between given minimum and maximum
      */
      float getValueA( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Reads R, G, and B channels and returns Vec3f with result at pixel (x,y).
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return Vec3f with results
      */
      agx::Vec3f getValueRGB( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Reads R, G, B and A channels and returns Vec4f with result at pixel (x,y). I.e., the PNG has to have four channels for the values to
      be valid.
      \param x - pixel at column x
      \param y - pixel at row y
      \param func - function taking a value from 0 to 1 following a given mathematical function (e.g., linear, quadratic, cubic etc)
      \return Vec4f with results
      */
      agx::Vec4f getValueRGBA( size_t x, size_t y, GetValueFunction func = linearValueFunction ) const;

      /**
      Set current value to the "Red" channel.
      \note This value isn't stored in the actual image on disc - it's only stored in a local data storage.
      */
      void setValueR( unsigned char val, size_t x, size_t y );

      /**
      Set current value to the "Green" channel.
      \note This value isn't stored in the actual image on disc - it's only stored in a local data storage.
      */
      void setValueG( unsigned char val, size_t x, size_t y );

      /**
      Set current value to the "Blue" channel.
      \note This value isn't stored in the actual image on disc - it's only stored in a local data storage.
      */
      void setValueB( unsigned char val, size_t x, size_t y );

      /**
      Set current value to the "Alpha" channel.
      \note This value isn't stored in the actual image on disc - it's only stored in a local data storage.
      */
      void setValueA( unsigned char val, size_t x, size_t y );

    protected:
      /**
      Reference counted object. Protected destructor.
      */
      virtual ~PNGImageValueInterpreter();

    protected:
      agxIO::ImageRef m_image;
      float m_minValue;
      float m_maxValue;
      ReadValue01Function m_readFunction;
  };

  typedef agx::ref_ptr< PNGImageValueInterpreter > PNGImageValueInterpreterRef;
}

#endif
