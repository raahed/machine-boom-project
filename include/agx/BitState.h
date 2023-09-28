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

#ifndef AGX_BITSTATE_H
#define AGX_BITSTATE_H

#include <agx/Integer.h>

#include <agxStream/OutputArchive.h>
#include <agxStream/InputArchive.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  /**
  Enumeration state, mainly used in agx::MergedBody, merge-split implementation.

  NOTE: Avoid using 0 as an invalid state, e.g., NONE, INVALID or NOTHING, use
        empty() instead, because state.Is(0) will always be false.

  enum States
  {
    THIS_HAS_BEEN_DONE = 1 << 0,
    DO_THAT = 1 << 1,
    MOVE_OVER_HERE = 1 <<2
  };

  BitState<States> state;
  state.add( DO_THAT );
  state.remove( THIS_HAS_BEEN_DONE );
  state.update( MOVE_OVER_HERE, shouldMoveOvereThere );
  */
  template< typename EnumT, typename BitT = agx::Int64 >
  class BitState
  {
    public:
      typedef BitT BitType;
      typedef EnumT EnumType;

    public:
      BitState()
        : m_bitState( BitType( 0 ) )
      {
      }

      explicit BitState( BitType initialState )
        : m_bitState( initialState )
      {
      }

      ~BitState() = default;

      agx::Bool operator== ( EnumType state ) const
      {
        return m_bitState == BitType( state );
      }

      agx::Bool operator!= ( EnumType state ) const
      {
        return !( *this == state );
      }

      agx::Bool empty() const
      {
        return m_bitState == BitType( 0 );
      }

      BitT get() const
      {
        return m_bitState;
      }

      BitState& set( BitType state )
      {
        m_bitState = state;
        return *this;
      }

      BitState& update( EnumType state, agx::Bool enable )
      {
        if ( enable ) return add( state );
        else return remove( state );
      }

      BitState& add( EnumType state )
      {
        m_bitState |= (BitType)state;
        return *this;
      }

      BitState& remove( EnumType state )
      {
        m_bitState = (BitType)( m_bitState & ~state );
        return *this;
      }

      BitState& toggle( EnumType state )
      {
        return update( state, !Is( state ) );
      }

      agx::Bool Is( BitType state ) const
      {
        return ( m_bitState & state ) != 0;
      }

      agx::Bool Not( BitType state ) const
      {
        return ( m_bitState & state ) == 0;
      }

      void store( agxStream::OutputArchive& out ) const { out << agxStream::out( "bitState", m_bitState ); }
      void restore( agxStream::InputArchive& in ) { in >> agxStream::in( "bitState", m_bitState ); }

    private:
      BitType m_bitState;
  };
}

DOXYGEN_END_INTERNAL_BLOCK()

#endif
