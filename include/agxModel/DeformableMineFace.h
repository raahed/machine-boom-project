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

#ifndef AGXMODEL_DEFORMABLE_MINE_FACE_H
#define AGXMODEL_DEFORMABLE_MINE_FACE_H

#include <agxModel/export.h>

#include <agxModel/HeightFieldDeformer.h>
#include <agx/Physics/GranularBodySystem.h>
#include <agxData/Buffer.h>

namespace agx
{
  class Journal;
}

namespace agxModel
{

  AGX_DECLARE_POINTER_TYPES(DeformableMineFace);
  AGX_DECLARE_VECTOR_TYPES(DeformableMineFace);
  /**
  DeformableMineFace contains a agxCollide::HeightField.
  This height field can be deformed by simple geometrical overlap
  with a deformer shape.
  Right now, the supported shape types for deformer shapes are:
    Cylinder
  */
  class AGXMODEL_EXPORT DeformableMineFace : public agxModel::HeightFieldDeformer
  {
    public:

      /**
      Constructor. Create a height field deformer given a specified height field.
      */
      DeformableMineFace( agxCollide::HeightField* heightField, agxCollide::Shape* deformerShape );

      /// Called each time step. Overriding inherited method from HeightFieldDeformer.
      virtual void preCollide(const agx::TimeStamp& t) override;

      /// \return: The amount of deformed volume.
      agx::Real getDeformedVolume() const;

      /// Sets the deformed volume to zero.
      void resetDeformedVolume();

      /// \return: The deepest depth reached.
      agx::Real getDeepestDepthReached() const;

      /// Sets the deepest depth reached to zero.
      void resetDeepestDepthReached();

      /// Sets the minimum indices which should be deformed (default: 0, 0).
      void setMinDeformableIndices(size_t x, size_t y);

      /// Sets the maximum indices which should be deformed (default: HeightField::getResolutionX - 1, HeightField::getResolutionY - 1).
      void setMaxDeformableIndices(size_t x, size_t y);

      /// Returns the minimum indices which should be deformed (default: 0, 0).
      agx::Vec2i getMinDeformableIndices() const;

      /// Returns the maximum indices which should be deformed (default: HeightField::getResolutionX - 1, HeightField::getResolutionY - 1).
      agx::Vec2i getMaxDeformableIndices() const;

      /**
      Deforms the DeformableMineFace in the overlap with a specified geometry with the option
      to either register the deformation volume data as done with the coupled deformer shape.
      \param - the geometry to deform the DeformableMineFace with in the overlap
      \param recordDeformStats - true if the deformation volume data should be recorded as a
                                 regular deformation as with the coupled deformer shape, false
                                 if not. ( default: true )
      */
      void deformHeightFieldFromGeometry( agxCollide::Geometry* geometry, bool recordDeformStats=true );

      /**
      \return a reference to a vector containing all volume differences for modified indices (new - old).
      Same size and order as the vector in getModifiedIndices.
      */
      const agx::RealVector& getVolumeDifferences() const;

      /**
      Get points encapsulating the created volume from the
      height differences in the last modified height field vertices. These points
      should create a convex hull.
      \return true if there are any volume differences from the last modified heights.
      */
      bool getConvexHullFromRemovedVolume( agx::Vec3Vector& convexHull ) const;

      /**
      \return the coupled deformer shape in the DeformableMineFace.
      */
      agxCollide::Shape* getDeformerShape() const;

      /**
      Set height field indices that should be forbidden from deformation.
      \param forbiddenIndices - vector containing forbidden indices
      */
      void setForbiddenIndices( const agx::Vec2iVector& forbiddenIndices );

      /**
      Add a height field index that should be forbidden from deformation.
      \param x - the x index of the height field index that should not be allowed for deformation.
      \param y - the y index of the height field index that should not be allowed for deformation.
      */
      void addForbiddenIndex( size_t x, size_t y );

      /**
      \return the current forbidden indices that are forbidden from deformation.
      */
      agx::Vector< agx::Vec2i > getForbiddenIndices() const;

      /**
      Clears the current stored forbidden indices from the DeformableMineFace.
      */
      void clearForbiddenIndices();

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::DeformableMineFace);

  protected:
    virtual ~DeformableMineFace();
    DeformableMineFace();

    void deformHeightFieldFromShape( agxCollide::Shape* shape,
                                     bool recordDeformStats );

    bool isAllowedIndex( size_t x, size_t y ) const;

  protected:
    agxCollide::ShapeRef m_deformerShape;
    agx::Real m_deformedVolume;
    agx::Real m_deepestDepthReached;
    agx::Real m_deepestDepthReachedOffset;
    agx::Vec2i m_minDeformableIndices;
    agx::Vec2i m_maxDeformableIndices;
    agx::RealVector m_volumeDifferences;
    agx::SetVector<agx::Vec2i> m_forbiddenIndices;
  };

  typedef agx::ref_ptr<DeformableMineFace> DeformableMineFaceRef;

  // Implementations
  AGX_FORCE_INLINE DeformableMineFace::~DeformableMineFace()
  {
  }


  AGX_FORCE_INLINE agx::Real DeformableMineFace::getDeformedVolume() const
  {
    return m_deformedVolume;
  }


  AGX_FORCE_INLINE void DeformableMineFace::resetDeformedVolume()
  {
    m_deformedVolume = 0;
  }


  AGX_FORCE_INLINE agx::Real DeformableMineFace::getDeepestDepthReached() const
  {
    return m_deepestDepthReached;
  }


  AGX_FORCE_INLINE void DeformableMineFace::resetDeepestDepthReached()
  {
    m_deepestDepthReachedOffset += m_deepestDepthReached;
    m_deepestDepthReached = 0;
  }


  AGX_FORCE_INLINE void DeformableMineFace::setMinDeformableIndices(size_t x, size_t y)
  {
    m_minDeformableIndices = agx::Vec2i(x,y);
  }


  AGX_FORCE_INLINE void DeformableMineFace::setMaxDeformableIndices(size_t x, size_t y)
  {
    m_maxDeformableIndices = agx::Vec2i(x, y);
  }


  AGX_FORCE_INLINE agx::Vec2i DeformableMineFace::getMinDeformableIndices() const
  {
    return m_minDeformableIndices;
  }


  AGX_FORCE_INLINE agx::Vec2i DeformableMineFace::getMaxDeformableIndices() const
  {
    return m_maxDeformableIndices;
  }

  AGX_FORCE_INLINE const agx::RealVector& DeformableMineFace::getVolumeDifferences() const
  {
    return m_volumeDifferences;
  }

  /**
  DeformableParticleCreator is a class used to generate granular material given the deformed volume in the specified
  DeformableMineFace. A Swell Factor can be added to modify the volume created.
  */
  class AGXMODEL_EXPORT DeformableParticleCreator : public agxSDK::StepEventListener
  {
  public:
    /**
    Constructor of the DeformableParticleCreator
    \param mineFace The specified mine face that the creator will couple to
    \param system The granular body system that the generated granular particles will belong to.
    \param table Distribution table containing the size distribution used to create the particles from the deformed volume
    */
    DeformableParticleCreator(agxModel::DeformableMineFace* mineFace,
      agx::Physics::GranularBodySystem* system,
      agx::ParticleEmitter::DistributionTable* table);

    /// Called each time step.
    virtual void preCollide(const agx::TimeStamp&) override;

    /**
    Returns the total volume of the created particles from the deformed volume.
    */
    agx::Real getTotalCreatedVolume() const;

    /**
    Returns the registered summed deformed volume in the deformable mine face registered by the creator
    */
    agx::Real getSummedUpDeformedVolume() const;

    /**
    Set the Swell Factor of the creator. The Swell Factor will modify the volume created by the following formula:
    Created Volume = Swell Factor * Deformed Volume
    Default value is 1.0.[min=0.0, max=infinity]
    */
    void setSwellFactor(agx::Real swellFactor);

    /**
    Get the Swell Factor of the creator. The Swell Factor will modify the volume created by the following formula:
    Created Volume = Swell Factor * Deformed Volume
    */
    agx::Real getSwellFactor() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::DeformableParticleCreator);
  protected:
    virtual ~DeformableParticleCreator();

  private:
    /// Default constructor
    DeformableParticleCreator();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxModel::DeformableMineFaceRef m_deformableMineFace;
    agx::Physics::GranularBodySystemRef m_system;
    agx::ParticleEmitter::DistributionTableRef m_table;
    agx::Real m_swellFactor;
    agx::Real m_summedUpDeformedVolume;
    agx::Real m_totalCreatedVolume;
    agx::UInt m_numberOfBodiesCreated;
  };
}

#endif
