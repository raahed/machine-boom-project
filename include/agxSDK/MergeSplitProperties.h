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

#include <agxSDK/MergeSplitAlgorithm.h>
#include <agxSDK/GeometryContactMergeSplitThresholds.h>
#include <agxSDK/ConstraintMergeSplitThresholds.h>
#include <agxSDK/WireMergeSplitThresholds.h>
#include <agxSDK/MergeIgnoreFilter.h>
#include <agxSDK/MergeIgnoreGroups.h>

#include <agx/ICloneable.h>
#include <agx/BitState.h>


namespace agxSDK
{
  class GeometryContactMergeSplitThresholds;
  class ConstraintMergeSplitThresholds;
  class WireMergeSplitThresholds;

  AGX_DECLARE_POINTER_TYPES( MergeSplitProperties );

  class AGXPHYSICS_EXPORT MergeSplitProperties : public agx::ICloneable
  {
    public:
      /**
      Maps given callback to an index (0, 1, 2, ...).
      */
      static agx::UInt findCallbackIndex( agxSDK::MergeSplitAlgorithm::Callback callback );

    public:
      /**
      Default constructor, sets default values to all thresholds and
      merge + split disabled.
      */
      MergeSplitProperties();

      /**
      Enable/disable both merge and split for the object.
      \param enable - true to enable, false to disable
      */
      void setEnableMergeSplit( agx::Bool enable );

      /**
      \return true when both merge and split are enabled
      */
      agx::Bool getEnableMergeSplit() const;

      /**
      Enable/disable merge for this object.
      \param enable - true to enable, false to disable
      */
      void setEnableMerge( agx::Bool enable );

      /**
      \return true if the object may merge
      */
      agx::Bool getEnableMerge() const;

      /**
      Enable/disable split for this object.
      \param enable - true to enable, false to disable
      */
      void setEnableSplit( agx::Bool enable );

      /**
      \return true if the object may split
      */
      agx::Bool getEnableSplit() const;

      /**
      Add all rigid bodies with this MergeSplitProperty to the given merge
      ignore group. Merge can be disabled group-wise using
      MergeSplitHandler::setEnableMergePair.

      The group name will be converted to a numeric group ID larger than
      MergeIgnoreFilter::MAX_UNAMED_GROUP_ID

      \param group The group name this MergeSplitProperty should be added to.
      */
      void addGroup(const agx::Name& group);

      /**
      Add all rigid bodies with this MergeSplitProperty to the given merge
      ignore group. Merge can be disabled group-wise using
      MergeSplitHandler::setEnableMergePair.

      Group IDs larger than MergeIgnoreFilter::MAX_UNAMED_GROUP_ID are reserved
      for named merge ignore groups.

      \param group The group ID this MergeSplitProperty should be added to.
      */
      void addGroup(agx::UInt32 group);

      /**
      Remove the given merge ignore group from this MergeSplitProperty.
      \param group - The name of the group to remove.
      */
      void removeGroup(const agx::Name& group);

      /**
      Remove the given merge ignore group from this MergeSplitProperty.
      \param group - The ID of the group to remove.
      */
      void removeGroup(agx::UInt32 group);

      /**
      \return Return group IDs for all merge ignore groups, including named.
      */
      const agx::UInt32Vector& getGroupIds() const;

      /**
      \return All group names added to this MergeSplitProperty.
      */
      const agx::NameVector& getGroupNames() const;

      /**
      \return explicitly assigned contact thresholds, nullptr if not assigned (default MergeSplitHandler::getContactThresholds() will be used)
      */
      agxSDK::GeometryContactMergeSplitThresholds* getContactThresholds() const;

      /**
      \return explicitly assigned/create contact thresholds, and in case of no assigned instance, a new (with default values) will be created
      */
      agxSDK::GeometryContactMergeSplitThresholds* getOrCreateContactThresholds();

      /**
      Assign contact thresholds. If nullptr, the default MergeSplitHandler contact thresholds will be used.
      */
      void setContactThresholds( agxSDK::GeometryContactMergeSplitThresholds* contactThresholds );

      /**
      \return explicitly assigned constraint thresholds, nullptr if not assigned (default MergeSplitHandler::getConstraintThresholds() will be used)
      */
      agxSDK::ConstraintMergeSplitThresholds* getConstraintThresholds() const;

      /**
      \return explicitly assigned/create constraint thresholds, and in case of no assigned instance, a new (with default values) will be created
      */
      agxSDK::ConstraintMergeSplitThresholds* getOrCreateConstraintThresholds();

      /**
      Assign constraint thresholds. If nullptr, the default MergeSplitHandler constraint thresholds will be used.
      */
      void setConstraintThresholds( agxSDK::ConstraintMergeSplitThresholds* constraintThresholds );

      /**
      \return explicitly assigned wire thresholds, nullptr if not assigned (default MergeSplitHandler::getWireThresholds() will be used)
      */
      agxSDK::WireMergeSplitThresholds* getWireThresholds() const;

      /**
      \return explicitly assigned/create wire thresholds, and in case of no assigned instance, a new (with default values) will be created
      */
      agxSDK::WireMergeSplitThresholds* getOrCreateWireThresholds();

      /**
      Assign wire thresholds. If nullptr, the default MergeSplitHandler wire thresholds will be used.
      */
      void setWireThresholds( agxSDK::WireMergeSplitThresholds* wireThresholds );

      /**
      Enable/disable native serialization of this object. Enabled by default.
      */
      void setEnableSerialization( agx::Bool flag );

      /**
      \return true if native serialization is enabled - otherwise false
      */
      agx::Bool getEnableSerialization() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agx::ICloneableRef clone( agx::ICloneable* child = nullptr ) override;
      void store( agxStream::OutputArchive& out ) const;
      void restore( agxStream::InputArchive& in );
      void storeLightData( agxStream::StorageStream& str ) const;
      void restoreLightData( agxStream::StorageStream& str );

      template<typename T>
      T* getMergeSplitThresholds( const agx::UInt index ) const;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      virtual ~MergeSplitProperties();

    private:
      enum EState
      {
        MERGE_ENABLED         = 1 << 0,
        SPLIT_ENABLED         = 1 << 1,
        SERIALIZATION_ENABLED = 1 << 2
      };

    private:
      typedef agx::BitState< EState, agx::Int32 > State;
      typedef agx::Vector<MergeSplitThresholdsRef> ThresholdsContainer;

    private:
      template<typename T>
      T* getOrCreateMergeSplitThresholds( const agx::UInt index );

      template<typename T>
      T* setMergeSplitThresholds( const agx::UInt index, agx::ref_ptr<T> thresholds );

    private:
      State m_state;
      ThresholdsContainer m_thresholds;
      agxSDK::MergeIgnoreGroups m_mergeIgnoreGroups;
  };

  AGX_FORCE_INLINE agx::Bool MergeSplitProperties::getEnableMergeSplit() const
  {
    return getEnableMerge() && getEnableSplit();
  }

  AGX_FORCE_INLINE agx::Bool MergeSplitProperties::getEnableMerge() const
  {
    return m_state.Is( MERGE_ENABLED );
  }

  AGX_FORCE_INLINE agx::Bool MergeSplitProperties::getEnableSplit() const
  {
    return m_state.Is( SPLIT_ENABLED );
  }

  inline GeometryContactMergeSplitThresholds* MergeSplitProperties::getContactThresholds() const
  {
    return getMergeSplitThresholds<GeometryContactMergeSplitThresholds>( MergeSplitProperties::findCallbackIndex( MergeSplitAlgorithm::CONTACTS ) );
  }

  inline ConstraintMergeSplitThresholds* MergeSplitProperties::getConstraintThresholds() const
  {
    return getMergeSplitThresholds<ConstraintMergeSplitThresholds>( MergeSplitProperties::findCallbackIndex( MergeSplitAlgorithm::CONSTRAINTS ) );
  }

  inline WireMergeSplitThresholds* MergeSplitProperties::getWireThresholds() const
  {
    return getMergeSplitThresholds<WireMergeSplitThresholds>( MergeSplitProperties::findCallbackIndex( MergeSplitAlgorithm::WIRES ) );
  }

  inline agx::UInt MergeSplitProperties::findCallbackIndex( agxSDK::MergeSplitAlgorithm::Callback callback )
  {
    agxAssert( callback != MergeSplitAlgorithm::NONE );
    return agx::pow2Exponent( callback );
  }

  template<typename T>
  inline T* MergeSplitProperties::getMergeSplitThresholds( const agx::UInt index ) const
  {
    agxAssert( index >= m_thresholds.size() || m_thresholds[ index ] == nullptr || m_thresholds[ index ]->is<T>() );

    return index < m_thresholds.size() ? (T*)m_thresholds[ index ].get() : nullptr;
  }

  template<typename T>
  T* MergeSplitProperties::getOrCreateMergeSplitThresholds( const agx::UInt index )
  {
    if ( index < m_thresholds.size() && m_thresholds[ index ] != nullptr ) {
      agxAssert( m_thresholds[ index ]->is<T>() );
      return (T*)m_thresholds[ index ].get();
    }

    return setMergeSplitThresholds<T>( index, new T() );
  }

  template<typename T>
  T* MergeSplitProperties::setMergeSplitThresholds( const agx::UInt index, agx::ref_ptr<T> thresholds )
  {
    if ( index >= m_thresholds.size() )
      m_thresholds.resize( index + 1 );

    return (T*)( m_thresholds[ index ] = thresholds ).get();
  }
}
