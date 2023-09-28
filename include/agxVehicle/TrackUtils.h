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

#include <agxVehicle/Track.h>

#include <agxCollide/Box.h>

namespace agxVehicle
{
  namespace utils
  {
    inline void createGeometryBox( const TrackNode& node )
    {
      agxCollide::GeometryRef geometry = new agxCollide::Geometry( new agxCollide::Box( node.getHalfExtents() ) );
      agx::AffineMatrix4x4 transform = agx::AffineMatrix4x4::translate( 0, 0, node.getHalfExtents().z() );
      geometry->setLocalTransform( transform );
      node.getRigidBody()->add( geometry );
    }

    /**
    Given a sorted container this function will split the content
    into ranges where the elements are consecutive.
    \param container - sorted container with values
    \return ranges where each range holds elements in consecutive order
    */
    template<typename ContainerT>
    IndexRangeContainer splitNonconsecutive( const ContainerT& container )
    {
      if ( container.size() < 2 )
        return IndexRangeContainer{ IndexRange( 0, container.size() ) };

      agxAssert( std::is_sorted( container.begin(), container.end() ) );

      IndexRangeContainer result{ IndexRange( 0, 1 ) };
      for ( agx::UInt i = 1; i < container.size(); ++i ) {
        if ( container[ i ] == container[ i - 1 ] + typename ContainerT::value_type( 1 ) )
          ++result.back().end();
        else
          result.push_back( agx::IndexRange( i, i + 1 ) );
      }

      return result;
    }

    /**
    Splits a sorted container into ranges where each split happens when
    container[ i ] - container[ i - 1 ] > 1. The ranges are transformed
    back to the track node indices and if the last node is included
    the first range will include the last nodes. E.g., three ranges
    with number of nodes = 60:
    [0 1 2][45 46][57 58 59] will at the end be two ranges:
    [57 58 59 0 1 2][45 46].
    */
    template<typename ContainerT>
    IndexRangeContainer splitNonconsecutive( const Track& track, const ContainerT& container )
    {
      auto result = splitNonconsecutive( container );

      agxAssert( !result.empty() );
      if ( result.size() == 1 && result.front().empty() )
        return IndexRangeContainer{};

      // Transform ranges from indices to container values. E.g.,
      // 2 4 7 [9 10 11 12] 18
      // index: 3  ...  6
      // range original:    begin = 3, end = 7
      // range transformed: begin = 9, end = 13
      for ( agx::UInt i = 0; i < result.size(); ++i ) {
        auto& range     = result[ i ];
        const auto size = range.size();
        range.begin()   = container[ range.begin() ];
        range.end()     = range.begin() + size;
      }

      if ( result.size() < 2 )
        return result;

      // Including the last range in the first if the first node belongs
      // to the first range and the last node to the last.
      const auto numNodes = track.nodes().size();
      if ( result.front().begin() == agx::UInt( 0 ) && result.back().end() == numNodes ) {
        result.front().begin() = result.back().begin();
        result.pop_back();
      }

      return result;
    }

    /**
    \return the previous index so if index == 0 the result will be size - 1
    */
    inline agx::UInt prevIndex( agx::UInt index, agx::UInt size )
    {
      agxAssert( size > 0 && index != agx::InvalidIndex );
      return index == 0 ? size - 1 : index - 1;
    }

    /**
    \return the next index so if index == size - 1 the result will be 0
    */
    inline agx::UInt nextIndex( agx::UInt index, agx::UInt size )
    {
      agxAssert( size > 0 && index != agx::InvalidIndex );
      return ( index + 1 ) % size;
    }

    /**
    Creates a node range where nodeIndex is excluded and number of nodes
    to include is signed. E.g., given [0 1 2 3 4 5]:
      nodeIndex = 3, numNodes = +3 -> [4 5 0]
      nodeIndex = 3, numNodes = -4 -> [5 0 1 2].
    */
    inline TrackNodeRange createRangeExcludingNodeIndex( const Track& track, agx::UInt nodeIndex, agx::Int signedNumNodes )
    {
      const auto totalNumNodes = track.nodes().size();
      if ( totalNumNodes < 1 )
        return track.nodes();

      const agx::UInt numNodes = signedNumNodes < 0 ?
                                   (agx::UInt)-signedNumNodes :
                                   (agx::UInt)signedNumNodes;
      agx::UInt beginIndex = agx::InvalidIndex;
      agx::UInt endIndex   = agx::InvalidIndex;
      if ( signedNumNodes < 0 ) {
        beginIndex = nodeIndex > numNodes ?
                       nodeIndex - numNodes :
                       nodeIndex + totalNumNodes - numNodes;
        endIndex   = nodeIndex;
      }
      else {
        beginIndex = ( nodeIndex + 1 ) % totalNumNodes;
        endIndex = ( nodeIndex + numNodes + 1 ) % totalNumNodes;
      }

      return TrackNodeRange( track.getIterator( beginIndex ), track.getIterator( endIndex ) );
    }
  }
}
