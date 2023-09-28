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

#include <agxWire/Node.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxWire
{
  class WireDistanceCompositeConstraint;

  typedef agx::RangeReal WireNodeRange;

  struct AGXPHYSICS_EXPORT HighResData
  {
    HighResData() : range( agx::Real( 0 ), agx::Real( 0 ) ), resolutionPerUnitLength( agx::Real( 0 ) ) {}

    HighResData( agx::RangeReal rangeRelWire, agx::Real resPerUnitLength )
      : range( rangeRelWire ), resolutionPerUnitLength( resPerUnitLength ) {}

    agx::Bool overlaps( agx::RangeReal otherRange ) const
    {
      return agx::leq(range.lower(), otherRange.upper() ) &&
             agx::geq(range.upper(), otherRange.lower() );
    }

    agx::Bool overlaps( agx::Real val, agx::Real epsilon )
    {
      return overlaps( agx::RangeReal( val - epsilon, val + epsilon ) );
    }

    HighResData reversed() const
    {
      HighResData ret = *this;
      ret.range.lower() *= agx::Real( -1 );
      ret.range.upper() *= agx::Real( -1 );
      std::swap( ret.range.lower(), ret.range.upper() );
      return ret;
    }

    agx::RangeReal range;
    agx::Real resolutionPerUnitLength;
  };
  typedef agx::Vector< HighResData > HighResDataContainer;
  typedef agx::List< HighResData > HighResDataList;

  class AGXPHYSICS_EXPORT WireNodeDensityController : public agx::Referenced
  {
    public:
      WireNodeDensityController() {}

      void update( agxWire::WireDistanceCompositeConstraint* wire );

      void clear();

      /**
      \note A range in this structure can be < 0 and > wire rest length.
      \sa getRangeExcessData
      \return high resolution data for this wire containing start and end points and new resolution
      */
      const HighResDataContainer& getHighResolutionData() const;

      /**
      Non-const version only. Are you interested in looking at this data you should
      also be interested in deleting it's content when done.
      \return the high resolution range excess data, i.e., the ranges not part of this wire
      */
      HighResDataList& getRangeExcessData();

      void addResolutionRange( agxWire::Node* node, agx::Real distanceAlongWire );
      void addResolutionRange( agx::RangeReal range, agx::Real resolutionPerUnitLength );
      void addResolutionRange( agxWire::HighResData hresData );

      void addExternalHighResolutionRange( Node* node, agx::Real distanceOnSegment );

      /**
      find if lumped node position is ok for the local resolution of the line
      */
      bool resolutionHighEnoughForPosition(const agx::Real restlengthToBfn,const agx::Real minDistanceToLump, const agx::Real eps = agx::Real(1E-6) ) const;

      /**
      \param distanceOnWireToStart - The start of the part of the wire we consider
      \param localLength  - The distance of the part of the line we consider
      \param foundRanges  - the ranges within local segment of length "localLength" where there could appear nodes
      */
      void findLocalHighResolutionRanges( const agx::Real distanceOnWireToStart, const agx::Real localLength, agx::Vector< agxWire::WireNodeRange >& foundRanges );

    private:
      HighResDataContainer m_highResolutionNodesDistanceOnWire;
      HighResDataList m_rangeExcess;

  };

}

DOXYGEN_END_INTERNAL_BLOCK()
