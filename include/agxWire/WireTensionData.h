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

namespace agxWire
{
  /**
  Structure containing relevant tension data for a wire element/segment.
  */
  struct AGXPHYSICS_EXPORT WireSegmentTensionData
  {
    WireSegmentTensionData();
    WireSegmentTensionData( agx::Real rawTension, agx::Real smoothedTension );
    explicit WireSegmentTensionData( const agxWire::Node::Tension* tension );

    agx::Real raw;      /**< Raw tension value. */
    agx::Real smoothed; /**< Smoothed tension value given the current smoothing filter. */
  };

  /**
  Structure containing relevant tension data for a node. A node is connected to
  two wire segments, one defined to be before (prev), the other after (next).
  */
  struct AGXPHYSICS_EXPORT WireNodeTensionData
  {
    /**
    Default constructor - all values default.
    */
    WireNodeTensionData();

    /**
    Construct given node tension data for current node (for prev segment) and next node in (for next segment).
    */
    WireNodeTensionData( const agxWire::Node::Tension* currNodeTension, const agxWire::Node::Tension* nextNodeTension );

    /**
    \return the average raw tension value over prev and next segments
    */
    agx::Real getAverageRaw() const;

    /**
    \return the average smoothed tension value over prev and next segments
    */
    agx::Real getAverageSmoothed() const;

    /**
    \return the raw tension value from previous segment (by definition the end node on a segment carries the tension value)
    */
    agx::Real getRaw() const;

    /**
    \return the smoothed tension value from previous segment (by definition the end node on a segment carries the tension value)
    */
    agx::Real getSmoothed() const;

    WireSegmentTensionData prev; /**< Segment before node (this node is end node to that segment) tension data. */
    WireSegmentTensionData next; /**< Segment after node (this node is begin node to that segment) tension data. */
  };

  DOXYGEN_START_INTERNAL_BLOCK()


  inline WireSegmentTensionData::WireSegmentTensionData()
    : raw( agx::Real( 0 ) ), smoothed( agx::Real( 0 ) )
  {
  }

  inline WireSegmentTensionData::WireSegmentTensionData( agx::Real rawTension, agx::Real smoothedTension )
    : raw( rawTension ), smoothed( smoothedTension )
  {
  }

  inline WireSegmentTensionData::WireSegmentTensionData( const Node::Tension* tension )
    : raw( tension != nullptr ? tension->raw : agx::Real( 0 ) ), smoothed( tension != nullptr ? tension->smoothed : agx::Real( 0 ) )
  {
  }

  inline WireNodeTensionData::WireNodeTensionData()
    : prev(), next()
  {
  }

  inline WireNodeTensionData::WireNodeTensionData( const Node::Tension* currNodeTension, const Node::Tension* nextNodeTension )
    : prev( currNodeTension ), next( nextNodeTension )
  {
  }

  inline agx::Real WireNodeTensionData::getAverageRaw() const
  {
    return agx::Real( 0.5 ) * ( prev.raw + next.raw );
  }

  inline agx::Real WireNodeTensionData::getAverageSmoothed() const
  {
    return agx::Real( 0.5 ) * ( prev.smoothed + next.smoothed );
  }

  inline agx::Real WireNodeTensionData::getRaw() const
  {
    return prev.raw;
  }

  inline agx::Real WireNodeTensionData::getSmoothed() const
  {
    return prev.smoothed;
  }

  DOXYGEN_END_INTERNAL_BLOCK()

}

