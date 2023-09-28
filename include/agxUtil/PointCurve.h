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

#include <agx/agx_vector_types.h>
#include <agx/Logger.h>

#include <algorithm>

namespace agxUtil
{
  enum class SegmentType
  {
    FIRST,
    INTERMEDIATE,
    LAST
  };

  /**
  Utility class curve defined by a set of points. It's possible
  to iterate this curve point and segment wise. Main purpose is
  to be able to traverse this curve with a given segment length.
  It's also possible to find optimal segment length given desired
  number of segments.
  */
  template<typename T>
  class PointCurve
  {
    public:
      using value_type = T;
      using container_type = agx::Vector<value_type>;

    public:
      /**
      Segment data with begin and end point.
      */
      struct Segment
      {
        value_type begin;
        value_type end;
      };

      /**
      Segment point with current curve segment and a point on
      that segment with local and global time. Result of
      PointCurve::evaluate( globalTime ).
      */
      struct SegmentPoint : public Segment
      {
        SegmentPoint() : point(), time( -agx::Infinity ), localTime( -1 )
        {
        }

        agx::Vec3 point;      /**< Point on segment. */
        agx::Real time;       /**< The point's global time on the curve. */
        agx::Real localTime;  /**< The point's local time on the curve. */

        /**
        \return true if this segment point is valid - otherwise false
        */
        inline agx::Bool isValid() const { return localTime >= agx::Real( 0 ); }
      };

      /**
      Result data of segmentation, PointCurve::findSegmentLength.
      */
      struct SegmentationResult
      {
        SegmentationResult( agx::UInt numSegments )
          : error( agx::Infinity ), segmentLength( agx::Real( -1 ) ), numSegments( numSegments ), numIterations( 0 )
        {
        }

        agx::Real error;
        agx::Real segmentLength;
        agx::UInt numSegments;
        agx::UInt numIterations;
      };

    public:
      using TransformCallback = std::function<void( value_type& )>;
      using SegmentCallback = std::function<void( const Segment&, SegmentType )>;
      using SegmentPointCallback = std::function<void( const SegmentPoint&, const SegmentPoint&, SegmentType )>;
      using SegmentationErrorCallback = std::function<agx::Real( const PointCurve&,
                                                                 const SegmentPoint&,
                                                                 const SegmentPoint&,
                                                                 SegmentType )>;

    public:
      /**
      Default constructor.
      */
      PointCurve();

      /**
      Construct given any container. The ContainerT::value_type must support
      cast to agx::Vec3.
      \param container - container with points
      */
      template<typename ContainerT>
      PointCurve( const ContainerT& container );

      /**
      Construct given any container and a transform function. The transform
      function transforms each ContainerT::value_type to this::value_type.
      \param container - container with points
      \param func - transform function
      */
      template<typename ContainerT, typename TransformFunc>
      PointCurve( const ContainerT& container, TransformFunc func );

      /**
      When all points has been added this method has to be called to collect
      curve data for efficient manipulation/evaluation.
      \return true if successfully initialized
      */
      agx::Bool finalize();

      /**
      Add new point to this curve.
      \note PointCurve::finalize has to be called when all additional points
            has been added.
      \param point - point to add
      */
      template<typename T2>
      void add( const T2& point );

      /**
      Evaluate at given time. Time = 0 is the first point and time = 1 the last.
      If time < 0 or time > 1 the point is interpolated from the first or the
      last segment.
      \param time - time along this curve
      \return segment point data
      */
      SegmentPoint evaluate( agx::Real time ) const;

      /**
      \return the total length of this curve
      */
      agx::Real calculateLength() const;

      /**
      Transform points in this curve. Finalize will be executed
      after all points has been transformed.
      */
      void transform( TransformCallback callback );

      /**
      Traverse all segments.
      \param callback - function to call for each segment
      */
      void traverse( SegmentCallback callback ) const;

      /**
      Traverses this curve and invokes callback with a segment of a given segment length.
      Note that the curve might be longer or shorter at the end.
      \param callback - callback with two SegmentPoint defining a segment of length \p segmentLength
      \param segmentLength - Length of the segment
      \param tolerance - segment length tolerance
      */
      void traverse( SegmentPointCallback callback, agx::Real segmentLength, agx::Real tolerance = agx::Real( 1.0E-6 ) ) const;

      /**
      Uses Newton Raphson to minimize the error while dividing this curve into segments.
      \param numSegments - number of segments to divide this curve into
      \param errorFunction - function which returns the error of each segment
      \param maxError - algorithm is successful when the absolute value of the sum over
                        errorFunction of all segments is less than this value
      \param segmentLengthTolerance - segment length tolerance
      \param maxNumIterations - maximum number of iterations before exiting
      \return the resulting segment length with minimum error
      */
      SegmentationResult findSegmentLength( agx::UInt numSegments,
                                            SegmentationErrorCallback errorFunction,
                                            agx::Real maxError,
                                            agx::Real segmentLengthTolerance = agx::Real( 1.0E-6 ),
                                            agx::UInt maxNumIterations = 100ul ) const;

    private:
      /**
      Finds index in m_points and m_time given global time.
      */
      agx::UInt findIndex( agx::Real time ) const;

    private:
      container_type m_points;
      agx::RealVector m_time;
  };

  template<typename T>
  PointCurve<T>::PointCurve()
  {
  }

  template<typename T>
  template<typename ContainerT>
  PointCurve<T>::PointCurve( const ContainerT& container )
  {
    m_points.resize( container.size() );
    for ( agx::UInt i = 0; i < m_points.size(); ++i )
      m_points[ i ] = (value_type)container[ i ];
  }

  template<typename T>
  template<typename ContainerT, typename TransformFunc>
  PointCurve<T>::PointCurve( const ContainerT& container, TransformFunc func )
  {
    m_points.resize( container.size() );
    std::transform( container.begin(), container.end(), m_points.begin(), func );
  }

  template<typename T>
  template<typename T2>
  void PointCurve<T>::add( const T2& point )
  {
    if ( !m_time.empty() )
      m_time.clear();

    m_points.push_back( (value_type)point );
  }

  template<typename T>
  typename PointCurve<T>::SegmentPoint PointCurve<T>::evaluate( agx::Real time ) const
  {
    if ( m_points.size() < 2 ) {
      LOGGER_WARNING() << "PointCurve::evaluate called with an undefined curve - number of points < 2." << LOGGER_ENDL();
      return SegmentPoint();
    }

    if ( m_points.size() != m_time.size() ) {
      LOGGER_WARNING() << "PointCurve::finalize must be executed before PointCurve::evaluate" << LOGGER_ENDL();
      return SegmentPoint();
    }

    auto segment = SegmentPoint();
    auto index = findIndex( time );

    if ( index + 1 == m_points.size() ) {
      // If time >= 1 roll back the index from the last point
      // to enable time > 1 to be calculated beyond last point.
      --index;
    }

    segment.begin     = m_points[ index ];
    segment.end       = m_points[ index + 1 ];
    segment.time      = time;
    segment.localTime = ( time - m_time[ index ] ) / ( m_time[ index + 1 ] - m_time[ index ] );
    segment.point     = (agx::Vec3)segment.begin + segment.localTime * ( (agx::Vec3)segment.end - (agx::Vec3)segment.begin );

    return segment;
  }

  template<typename T>
  agx::Real PointCurve<T>::calculateLength() const
  {
    agx::Real length = agx::Real( 0 );
    for ( agx::UInt i = 1; i < m_points.size(); ++i )
      length += ((agx::Vec3)m_points[ i - 1 ]).distance( (agx::Vec3)m_points[ i ] );

    return length;
  }

  template<typename T>
  agx::Bool PointCurve<T>::finalize()
  {
    m_time.resize( m_points.size(), agx::Real( 0 ) );

    const auto totalLength = calculateLength();
    if ( totalLength < agx::RealEpsilon )
      return false;

    agx::Real accumulatedTime = agx::Real( 0 );
    m_time.front() = accumulatedTime;
    for ( agx::UInt i = 1; i < m_points.size(); ++i ) {
      accumulatedTime += ((agx::Vec3)m_points[ i - 1 ]).distance( (agx::Vec3)m_points[ i ] ) / totalLength;
      m_time[ i ] = accumulatedTime;
    }
    m_time.back() = agx::Real( 1 );

    return m_time.size() > 1;
  }

  template<typename T>
  void PointCurve<T>::transform( typename PointCurve<T>::TransformCallback callback )
  {
    for ( agx::UInt i = 0; i < m_points.size(); ++i )
      callback( m_points[ i ] );

    this->finalize();
  }

  template<typename T>
  void PointCurve<T>::traverse( typename PointCurve<T>::SegmentCallback callback ) const
  {
    for ( agx::UInt i = 1; i < m_points.size(); ++i )
      callback( Segment{ m_points[ i - 1 ], m_points[ i ] },
                i == 1 ?
                  SegmentType::FIRST :
                i + 1 == m_points.size() ?
                  SegmentType::LAST :
                  SegmentType::INTERMEDIATE );
  }

  template<typename T>
  void PointCurve<T>::traverse( typename PointCurve<T>::SegmentPointCallback callback,
                                agx::Real segmentLength,
                                agx::Real tolerance /*= agx::Real( 1.0E-6 )*/ ) const
  {
    const auto totalLength = calculateLength();
    if ( totalLength < agx::RealEpsilon )
      return;

    const agx::Real dt = segmentLength / totalLength;
    agx::Bool done     = false;
    agx::Real prevT    = agx::Real( 0 );
    auto prev          = evaluate( prevT );
    SegmentType type = SegmentType::FIRST;
    while ( !done ) {
      agx::Real currT          = prevT + dt;
      auto curr                = evaluate( currT );
      agx::Real prevToCurrDist = prev.point.distance( curr.point );
      while ( !agx::equivalent( prevToCurrDist, segmentLength, tolerance ) ) {
        const agx::Real overshoot = prevToCurrDist - segmentLength;
        currT                    -= overshoot / totalLength;
        curr                      = evaluate( currT );
        prevToCurrDist            = prev.point.distance( curr.point );
      }

      done = currT > agx::Real( 1 ) + dt;
      done = done ||
             ( currT + agx::Real( 0.5 ) * dt >= agx::Real( 1 ) &&
               curr.point.distance( (agx::Vec3)m_points.back() ) < agx::Real( 0.5 ) * segmentLength );

      if ( done )
        type = SegmentType::LAST;

      callback( prev, curr, type );

      if ( type == SegmentType::FIRST )
        type = SegmentType::INTERMEDIATE;

      prevT = currT;
      prev  = curr;
    }
  }

  template<typename T>
  typename
  PointCurve<T>::SegmentationResult PointCurve<T>::findSegmentLength( agx::UInt numSegments,
                                                                      typename PointCurve<T>::SegmentationErrorCallback errorFunction,
                                                                      agx::Real maxError,
                                                                      agx::Real segmentLengthTolerance /*= agx::Real( 1.0E-6 )*/,
                                                                      agx::UInt maxNumIterations /*= 100ul*/ ) const
  {
    SegmentationResult result( numSegments );
    if ( numSegments < 1 )
      return result;

    const auto totalLength = calculateLength();
    if ( totalLength < agx::RealEpsilon )
      return result;

    result.segmentLength = totalLength / agx::Real( result.numSegments );

    const agx::Real dl = agx::Real( 1.0E-3 ) / agx::Real( result.numSegments );
    agx::Bool done = false;
    SegmentationResult bestResult( numSegments );
    const auto& self = *this;
    while ( !done && result.numIterations < maxNumIterations ) {
      ++result.numIterations;

      agx::Real ePrev = agx::Real( 0 );
      agx::Real eCurr = agx::Real( 0 );
      agx::Real eNext = agx::Real( 0 );

      this->traverse( [&self, &errorFunction, &ePrev]( const SegmentPoint& p1, const SegmentPoint& p2, SegmentType type )
      {
        ePrev += errorFunction( self, p1, p2, type );
      }, result.segmentLength - dl, segmentLengthTolerance );

      this->traverse( [&self, &errorFunction, &eCurr]( const SegmentPoint& p1, const SegmentPoint& p2, SegmentType type )
      {
        eCurr += errorFunction( self, p1, p2, type );
      }, result.segmentLength, segmentLengthTolerance );

      this->traverse( [&self, &errorFunction, &eNext]( const SegmentPoint& p1, const SegmentPoint& p2, SegmentType type )
      {
        eNext += errorFunction( self, p1, p2, type );
      }, result.segmentLength + dl, segmentLengthTolerance );

      if ( eCurr < bestResult.error ) {
        bestResult = result;
        bestResult.error = eCurr;
      }

      const agx::Real dr = -agx::Real( 2 ) * dl * eCurr / ( eNext - ePrev );
      result.segmentLength += dr;

      // No solution if we're approaching negative length or reach max iterations.
      if ( result.numIterations == maxNumIterations ||
           result.segmentLength - dl < agx::REAL_SQRT_EPSILON )
        return bestResult;

      result.error = agx::Real( 0 );
      this->traverse( [&self, &result, &errorFunction]( const SegmentPoint& p1, const SegmentPoint& p2, SegmentType type )
      {
        result.error += errorFunction( self, p1, p2, type );
      }, result.segmentLength, segmentLengthTolerance );

      done = std::abs( result.error ) <= maxError;
    }

    return result;
  }

  template<typename T>
  agx::UInt PointCurve<T>::findIndex( agx::Real time ) const
  {
    if ( agx::leq( time, agx::Real( 0 ) ) )
      return 0ul;
    else if ( agx::geq( time, agx::Real( 1 ) ) )
      return std::max( m_points.size(), (typename container_type::size_type)1 ) - 1;

    auto it = std::lower_bound( m_time.begin(), m_time.end(), time );
    agxAssert( it != m_time.begin() );
    agxAssert( it != m_time.end() );

    return it - m_time.begin() - 1;
  }
}
