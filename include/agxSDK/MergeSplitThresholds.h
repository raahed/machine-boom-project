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

#include <agx/ICloneable.h>
#include <agx/BitState.h>
#include <agx/agx_vector_types.h>

#include <agxStream/Serializable.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( MergeSplitThresholds );

  /**
  Base class for thresholds/constants/values used in agxSDK::MergeSplitAlgorithm. This object
  hold real values and handles store/restore of these.
  */
  class AGXPHYSICS_EXPORT MergeSplitThresholds : public agx::ICloneable, public agxStream::Serializable
  {
    public:
      /**
      Mixing two values function signature given two values and the return value is the mix.
      */
      using MixFunction = std::function<agx::Real( agx::Real, agx::Real )>;

    public:
      /**
      Assign value given index and value. Assuming index < getNumValues().
      */
      void set( const agx::UInt index, agx::Real value );

      /**
      \return value given index assuming index < getNumValues()
      */
      agx::Real get( const agx::UInt index ) const;

      /**
      \return the number of values handled by this object
      */
      agx::UInt getNumValues() const;

      /**
      \return true if these thresholds are global, i.e., created and handled by the MergeSplitHandler object
      */
      agx::Bool isGlobal() const;

      /**
      Find and mix values of two instances. Explicit instances, i.e., non-global (isGlobal() == false) are
      favored and if both are non-global (and different instances), the mixFunc will be called.
      */
      agx::Real get( const agx::UInt index, const MergeSplitThresholds* other, MergeSplitThresholds::MixFunction mixFunc ) const;

    public:
      /**
      Restore all thresholds to default.
      */
      virtual void resetToDefault() = 0;

      DOXYGEN_START_INTERNAL_BLOCK()
      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxSDK::MergeSplitThresholds );

      /**
      Stores all thresholds.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restores all thresholds.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      /**
      Saves internal data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data to stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Construct given number of thresholds.
      */
      MergeSplitThresholds( const agx::UInt numValues );

      /**
      Reference counted object, protected destructor.
      */
      virtual ~MergeSplitThresholds();

      /**
      Resizes and copies values to \p target.
      \param target - clone
      */
      void cloneValuesTo( MergeSplitThresholds& target ) const;

    private:
      friend class MergeSplitHandler;

      enum BitOptions
      {
        GLOBAL_THRESHOLDS = 1 << 0
      };

      using Options = agx::BitState<BitOptions, agx::Int32>;

      Options& getOptions();
      const Options& getOptions() const;

    private:
      agx::RealVector m_values;
      Options m_options;
  };

  AGX_FORCE_INLINE agx::Real MergeSplitThresholds::get( const agx::UInt index ) const
  {
    agxAssert( index < m_values.size() );
    return m_values[ index ];
  }

  inline agx::UInt MergeSplitThresholds::getNumValues() const
  {
    return (agx::UInt)m_values.size();
  }

  inline agx::Bool MergeSplitThresholds::isGlobal() const
  {
    return m_options.Is( GLOBAL_THRESHOLDS );
  }

  inline MergeSplitThresholds::Options& MergeSplitThresholds::getOptions()
  {
    return m_options;
  }

  inline const MergeSplitThresholds::Options& MergeSplitThresholds::getOptions() const
  {
    return m_options;
  }

  inline agx::Real MergeSplitThresholds::get( const agx::UInt index, const MergeSplitThresholds* other, MergeSplitThresholds::MixFunction mixFunc ) const
  {
           // Value from this if:
           //   1. other is not given
           //   2. this is identical to other
           //   3. other is global - i.e., this may contain unique values which overrides global
    return other == nullptr || this == other || other->isGlobal() ?
             get( index ) :
           // Value from other if:
           //   1. other is given (implicit)
           //   2. this is global
           this->isGlobal() ?
             other->get( index ) :
           // Mix values if both are given and none is global.
             mixFunc( get( index ), other->get( index ) );
  }
}
