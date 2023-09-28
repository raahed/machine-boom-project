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

#ifndef AGXMODEL_HEIGHT_FIELD_DEFORMER_H
#define AGXMODEL_HEIGHT_FIELD_DEFORMER_H

#include <agxModel/export.h>

#include <agxSDK/StepEventListener.h>
#include <agxSDK/Simulation.h>
#include <agxCollide/HeightField.h>
#include <agx/Journal.h>

namespace agxModel
{
  class HeightFieldDeformer;

  typedef agx::ref_ptr<HeightFieldDeformer> HeightFieldDeformerRef;
  typedef agx::VectorPOD<HeightFieldDeformer*> HeightFieldDeformerPtrVector;
  /**
  HeightFieldDeformer contains a agxCollide::HeightField.
  */
  class AGXMODEL_EXPORT HeightFieldDeformer : public agxSDK::StepEventListener
  {
    friend class HeightFieldDeformerJournalRecorder;
    public:

      /**
      Constructor. Create a height field deformer given a specified height field.
      */
      HeightFieldDeformer( agxCollide::HeightField* heightField, agxSDK::StepEventListener::ActivationMask mask);

      /**
      Return the HeightFieldDeformer geometry (which is created in HeightFieldDeformer constructor)
      */
      agxCollide::HeightField* getHeightField() const;

      /// Is the deformer valid? If not, no actions will be taken.
      bool isValid() const;

      /**
      \return a reference to a vector containing all indices in the height field which have been changed.
      */
      const agx::Vec2iVector& getModifiedIndices() const;

      /**
      \return a reference to a vector containing all height differences for modified indices (new - old).
      Same size and order as the vector in getModifiedIndices.
      */
      const agx::RealVector& getHeightDifferences() const;

      /**
      Utility function to extract all HeightFieldDeformers from an simulation (and its sub-simulation).
      \param deformers - The vector where all the deformables found in the simulation will be placed
      \param simulation - The simulation where to find all deformables.
      \return true if deformables are added to the vector, otherwise false.
      */
      static bool extractHeightFieldDeformers(agxModel::HeightFieldDeformerPtrVector& deformers, agxSDK::Simulation* simulation);

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::HeightFieldDeformer);

  protected:
    virtual ~HeightFieldDeformer();
    HeightFieldDeformer();

  protected:
    agxCollide::HeightFieldRef m_heightField;
    bool m_valid;
    agx::Vec2iVector m_modifiedIndices;
    agx::RealVector m_heightDifferences;


  };

  typedef agx::ref_ptr<HeightFieldDeformer> HeightFieldDeformerRef;

  // Implementations

  AGX_FORCE_INLINE agxCollide::HeightField* HeightFieldDeformer::getHeightField() const
  {
    return m_heightField;
  }


  AGX_FORCE_INLINE bool HeightFieldDeformer::isValid() const
  {
    return m_valid;
  }


  AGX_FORCE_INLINE const agx::Vec2iVector& HeightFieldDeformer::getModifiedIndices() const
  {
    return m_modifiedIndices;
  }


  AGX_FORCE_INLINE const agx::RealVector& HeightFieldDeformer::getHeightDifferences() const
  {
    return m_heightDifferences;
  }


  AGX_FORCE_INLINE HeightFieldDeformer::~HeightFieldDeformer()
  {
  }

  /**
  Utility class used to record HeightFieldDeformer in an agx::Journal. Will restore the heightfield values upon journal playback.
  */
  class AGXMODEL_EXPORT HeightFieldDeformerJournalRecorder : public agxSDK::StepEventListener
  {
  public:
    /**
    Constructor for the HeightFieldDeformerJournalRecorder.
    \param deformableHeightField The HeigthtFieldDeformer to store in the journal
    */
    HeightFieldDeformerJournalRecorder(HeightFieldDeformer* deformableHeightField);

    /**
    Called before the collision detection step is taken in the simulation
    \param time - the current simulation time
    */
    virtual void preCollide(const agx::TimeStamp& time) override;

    /**
    Called before a step is taken in the simulation
    \param time - the current simulation time
    */
    virtual void pre(const agx::TimeStamp& time) override;

    /**
    Called after a step is taken in the simulation
    \param time - the current simulation time
    */
    virtual void post(const agx::TimeStamp& time) override;

    /// Called when this object is added to a simulation
    virtual void addNotification() override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::HeightFieldDeformerJournalRecorder);

  protected:

    bool heigthBufferIsValid();

    void storeHeights();

    void storeChangedVertices();

    void restoreHeights();

    void initHeightFieldStorage();

    void createJournalBindings(agx::Journal* journal);

    void journalAttachedCallback(agxSDK::Simulation* simulation, agx::Journal* journal);

    agxData::Buffer* getHeightBuffer();

    virtual ~HeightFieldDeformerJournalRecorder();

  private:
    HeightFieldDeformerJournalRecorder();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxSDK::Simulation::JournalAttachEvent::CallbackType m_journalAttachedCallback;
    HeightFieldDeformerRef m_deformableHeightField;
    agx::RealVector        m_heightIndices;
    agxData::BufferRef     m_heigthBuffer;
    agx::ComponentRef      m_dataRoot;
    agx::Uuid              m_dataRootUUID;
    agx::JournalObserver   m_journal;
  };
}

#endif
