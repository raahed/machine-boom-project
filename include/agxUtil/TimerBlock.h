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

#include <agx/Timer.h>
#include <agx/String.h>

namespace agxUtil
{
  /**
  Utility class to time a scope. The title and time will be printed
  while this object is being destructed.
  */
  class TimerBlock
  {
    public:
      using OnDeleteCallback = std::function<void( agx::UInt64, agx::UInt64 )>;

    public:
      /**
      Create timer block given format string.

      Example:
      for ( ... ) {
        auto tb = agxUtil::TimerBlock::create( "Iteration %d", i + 1 );
      }

      Will print:
      {Timer Block}: Iteration 1
          - Execute time: 0.000149955 ms.
      {Timer Block}: Iteration 2
          - Execute time: 0.002132455 ms.
      {Timer Block}: Iteration 3
          - Execute time: 0.000093941 ms.
      etc.
      */
      template<typename... ArgsT>
      static TimerBlock create( const agx::String& format, ArgsT&&... args )
      {
        agx::String printout = agx::String::format( format.c_str(), std::forward<ArgsT>( args )... );
        return TimerBlock( [printout]( agx::UInt64 start, agx::UInt64 stop )
        {
          std::cout << "{Timer Block}: " << printout << std::endl;
          std::cout << "     - Execute time: " << agx::Timer::convertToMs( stop - start ) << " ms." << std::endl;
        } );
      }

    public:
      /**
      Construct given title. Print will be:
      {Timer Block}: $title
          - Execute time: $time ms.
      */
      inline TimerBlock( OnDeleteCallback onDeleteCallback )
        : m_onDeleteCallback( onDeleteCallback ),
          m_start( agx::Timer::getCurrentTick() ),
          m_stop( agx::InvalidIndex )
      {
      }

      /**
      Print during destruction.
      */
      inline ~TimerBlock()
      {
        if ( m_stop == agx::InvalidIndex )
          stop();

        if ( m_onDeleteCallback )
          m_onDeleteCallback( m_start, m_stop );
      }

      /**
      Stop the timer before we leave the scope.
      */
      inline void stop()
      {
        m_stop = agx::Timer::getCurrentTick();
      }

    private:
      OnDeleteCallback m_onDeleteCallback;
      agx::UInt64 m_start;
      agx::UInt64 m_stop;
  };
}
