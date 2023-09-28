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

#include <agxSDK/MergeSplitThresholds.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( WireMergeSplitThresholds );

  /**
  Specific thresholds used to split and merge objects given wires.
  */
  class AGXPHYSICS_EXPORT WireMergeSplitThresholds : public MergeSplitThresholds
  {
    public:
      enum MergeThresholds : agx::UInt
      {
        NUM_MERGE_THRESHOLDS
      };

      enum SplitThresholds : agx::UInt
      {
        MERGE_TENSION_SCALE = NUM_MERGE_THRESHOLDS, /**< When a node is merged the tension is stored and monitored
                                                         to perform split. This threshold scales the merge tension
                                                         making split more likely when < 1 and less likely > 1. When
                                                         this scale is 0 the only force keeping the node merged is
                                                         from the contact - which in most cases isn't enough. */
        FORCE_PROPAGATION_DECAY_SCALE,              /**< When external forces are acting on a partially merged wire,
                                                         the force will propagate and split several nodes at once.
                                                         This threshold controls the amout of force (of the total
                                                         external force) that is used to split each node. If this
                                                         value is high (> 1), the force will not propagate 'too'
                                                         long, keeping the wire merged. Default: 1.0 */
        NUM_THRESHOLDS
      };

    public:
      /**
      Default constructor, all thresholds will be set to default.
      */
      WireMergeSplitThresholds();

      /**
      When a node is merged the tension is stored and monitored
      to perform split. This threshold scales the merge tension
      making split more likely when < 1 and less likely > 1. When
      this scale is 0 the only force keeping the node merged is
      from the contact - which in most cases isn't enough.
      Default: 1.0
      \param mergeTensionScale - the merge tension scale
      */
      void setMergeTensionScale( agx::Real mergeTensionScale );

      /**
      \return the current merge tension scale
      */
      agx::Real getMergeTensionScale() const;

      /**
      When external forces are acting on a partially merged wire,
      the force will propagate and split several nodes at once.
      This threshold controls the amout of force (of the total
      external force) that is used to split each node. If this
      value is high (> 1), the force will not propagate 'too'
      long, keeping the wire merged. Default: 1.0.
      \param forcePropagationDecayScale - the force propagation decay scale
      */
      void setForcePropagationDecayScale( agx::Real forcePropagationDecayScale );

      /**
      \return the current force propagation decay scale
      */
      agx::Real getForcePropagationDecayScale() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::WireMergeSplitThresholds );

      /**
      Storing data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restoring data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

      /**
      Creates and returns a clone of this object.
      */
      virtual agx::ICloneableRef clone( agx::ICloneable* child = nullptr ) override;

      /**
      Resets all values to default.
      */
      virtual void resetToDefault() override;

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~WireMergeSplitThresholds();
  };

  inline agx::Real WireMergeSplitThresholds::getMergeTensionScale() const
  {
    return get( MERGE_TENSION_SCALE );
  }

  inline agx::Real WireMergeSplitThresholds::getForcePropagationDecayScale() const
  {
    return get( FORCE_PROPAGATION_DECAY_SCALE );
  }
}
