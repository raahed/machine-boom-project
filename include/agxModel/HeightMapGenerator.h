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

#ifndef AGXMODEL_HEIGHT_MAP_GENERATOR_H
#define AGXMODEL_HEIGHT_MAP_GENERATOR_H

#include <agxModel/export.h>
#include <agx/Referenced.h>
#include <agx/Real.h>
#include <agx/Vector.h>
#include <agx/agx_vector_types.h>
#include <agxIO/Image.h>
#include <agxStream/Serializable.h>


namespace agx
{
  typedef agx::Vector<agx::RealVector> RealMatrix;
}


namespace agxCollide
{
  class Trimesh;
}

namespace agxModel
{
  /**
  * A base class used for generating textures
  */
  class AGXMODEL_EXPORT HeightMapGenerator : public agx::Referenced
  {
  public:

    HeightMapGenerator();
    HeightMapGenerator(const size_t& width, const size_t& height);

    /**
    * Gets the width in pixels of the produced texture
    */
    size_t getWidth();

    /**
    * Sets the height in pixels of the produced texture
    */
    void setWidth(const size_t& width);

    /**
     * Gets the height in pixels of the produced texture
    */
    size_t getHeight();

    /**
    * Sets the height in pixels of the produced texture
    */
    void setHeigth(const size_t& height);

  protected:

    /// Default destructor
    virtual ~HeightMapGenerator();

  protected:
    size_t m_width;
    size_t m_height;
  };

  AGX_DECLARE_POINTER_TYPES(GaussianHeightMapGenerator);

  /**
  * This struct holds parameters for the creation of a height texture that is a composite bump created by Gaussian height functions
  */
  class AGXMODEL_EXPORT GaussianFunctionSettings
  {
  public:
    GaussianFunctionSettings(const agx::Vec2& size, const agx::Vec2& widthInter,
      const agx::Vec2& heightInter, size_t num, bool periodic_x = false, bool periodic_y = false, agx::Real max_Height = 0);

    GaussianFunctionSettings();

    agx::Vec2 areaSize;           // Specifying the area size that will be mapped to the height texture
    agx::Vec2 widthInterval;      // Specifying the width interval where bumps will be present
    agx::Vec2 heightInterval;     // Specifying the height interval where bumps will be present
    size_t numDimples;            // Specifying the number of Gaussian bumps that is used in the texture
    bool periodic_X;              // True if the bumps should be periodic along the x-axis
    bool periodic_Y;              // True if the bumps should be periodic along the Y-axis
    agx::Real maxHeight;          // Max height of any point in the height texture
  };

  /// A height map generator using a Gaussian function.
  class AGXMODEL_EXPORT GaussianHeightMapGenerator : public HeightMapGenerator
  {

  public:

    /// Default constructor
    GaussianHeightMapGenerator();

    /**
     *  Constructor that takes the image width, image hegiht and settings as arguments.
     */
    GaussianHeightMapGenerator(const size_t& width, const size_t& height,
      const GaussianFunctionSettings& settings);

    /**
    * Sets the settings that will be used when creating the Gaussian texture
    */
    void setSettings(const GaussianFunctionSettings& settings);

    /**
    * Gets the settings used in the simulation
    */
    GaussianFunctionSettings getSettings();

    /**
    * Sets the seed that it used by the random number generator when creating a bump texture.
    */
    void setSeed(agx::UInt seed);

    /**
     *  Generates and returns height data in matrix form.
     */
    agx::RealMatrix generateHeightData();

  protected:

    void init();

    /// Default destructor
    virtual ~GaussianHeightMapGenerator();

    /// Creates a matrix that holds the heights of the Gaussian dimples
    agx::RealMatrix createGaussianHeightMatrix(agx::Vec2 X_bounds, size_t X_resolution,
      agx::Vec2 Y_bounds, size_t Y_resolution,
      const agx::RealMatrix& pos, const agx::RealMatrix& spread, const agx::RealVector& amplitude,
      bool limitToMaxHeight=false, agx::Real maxHeight=0);

  protected:

    GaussianFunctionSettings m_settings;
    agx::UInt m_seed;
  };

  /*Implementation*/

  AGX_FORCE_INLINE size_t HeightMapGenerator::getWidth()
  {
    return m_width;
  }

  AGX_FORCE_INLINE size_t HeightMapGenerator::getHeight()
  {
    return m_height;
  }

  AGX_FORCE_INLINE void GaussianHeightMapGenerator::setSeed(agx::UInt seed)
  {
    m_seed = seed;
  }

  /**
  * This namespace contains utility functions for modifying different type of agx shapes from height data.
  */
  namespace MeshModifierUtils
  {
    AGXMODEL_EXPORT agxCollide::Trimesh* bumpCylindricalMeshAroundAnAxisFromHeightData(
      agxCollide::Trimesh *trimesh,
      agx::RealMatrix *heightData,
      agx::Vec3 axis,
      agx::Vec3 perpendicularBaseVector,
      agx::Real cylinderLength,
      agx::Vec3 cylinderCenter);
  }
}

#endif /*AGXMODEL_HEIGHT_MAP_GENERATOR_H*/
