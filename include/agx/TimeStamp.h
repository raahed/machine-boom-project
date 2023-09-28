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

#ifndef AGX_TIMESTAMP_H
#define AGX_TIMESTAMP_H

#include <agx/agxPhysics_export.h>

#include <agx/agx.h>


namespace agx {
  typedef agx::Real TimeStamp;

#if 0

  /**
  TimeStamp object identifies either the time at which an event should
  be triggered, or the time at which data was written.

  Since the objects defined in the user API contain copies of kinematic
  data produced by the DynamicsSystem, they are marked with a time
  stamp each time fresh data is published.
  \todo Define a time unit.
  \todo Provide accessors in TimeStamp for milliseconds, etc.
  \todo choose the basic unit of time stamp and decide what to do about
  interfacing Timer and TimeStamp.
  */
  class AGXPHYSICS_EXPORT TimeStamp {
  public:
    /**
    Default constructor
    */
    TimeStamp( Real t=0.0 );

    /**
    Copy constructor
    */
    TimeStamp( const TimeStamp &t );

    /**
    \return the current value of the TimeStamp
    */
    Real operator() () const ;
    /**
    Less than operator
    \return true if this TimeStamp is less that t.
    */
    bool operator< (const TimeStamp & t) const;
    /**
    Larger than operator
    \return true if this TimeStamp is larger that t.
    */
    bool operator> (const TimeStamp & t) const;
    /**
    Less or equal operator
    \return true if this TimeStamp is less or equal to  t.
    */
    bool operator<= (const TimeStamp & t) const;
    /**
    Larger or equal operator
    \return true if this TimeStamp is less or equal to  t.
    */
    bool operator>= (const TimeStamp & t) const;

    /**
    Equal operator
    \return true if this TimeStamp is equal to  t.
    */
    bool operator== (const TimeStamp & t) const;
    /**
    Not equal operator
    \return true if this TimeStamp is not equal to  t.
    */
    bool operator!= (const TimeStamp & t) const;

    /**
    Add operator
    */
    TimeStamp operator+ (Real r);
    const TimeStamp& operator+ (Real r) const;

    /**
    Add operator
    */
    TimeStamp operator+ (const TimeStamp& t);

    /**
    Add operator
    */
    TimeStamp& operator+= (Real r);

    /**
    Add operator
    */
    TimeStamp& operator+= (const TimeStamp& t);

    friend std::ostream& operator <<(std::ostream& os, const TimeStamp& t);

    operator double() const { return m_time; }

  private:
    Real m_time;
  };



  inline TimeStamp::TimeStamp( Real t ) : m_time(t) {}

  inline TimeStamp::TimeStamp( const TimeStamp &t )
  {
    if (this == &t)
      return;

    m_time = t.m_time;
  }

  inline Real TimeStamp::operator() () const { return m_time; }

  inline bool TimeStamp::operator< (const TimeStamp & t) const { return m_time < t.m_time; }

  inline bool TimeStamp::operator> (const TimeStamp & t) const  { return m_time > t.m_time; }

  inline bool TimeStamp::operator<= (const TimeStamp & t) const { return m_time <= t.m_time; }

  inline bool TimeStamp::operator>= (const TimeStamp & t) const { return m_time >= t.m_time; }

  inline bool TimeStamp::operator== (const TimeStamp & t) const { return m_time == t.m_time; }

  inline bool TimeStamp::operator!= (const TimeStamp & t) const { return m_time != t.m_time; }

  inline TimeStamp TimeStamp::operator+ (const TimeStamp & t) { return TimeStamp( m_time +t.m_time ); }
  inline TimeStamp TimeStamp::operator+ (Real t) { return TimeStamp( m_time+t ); }

  inline TimeStamp& TimeStamp::operator+= (const TimeStamp & t) { m_time += t.m_time; return *this; }
  inline TimeStamp& TimeStamp::operator+= (Real  t) { m_time += t; return *this; }

  inline std::ostream& operator <<(std::ostream& os, const TimeStamp& t) { os << t.m_time; return os; }
#endif

} // namespace agx
#endif
