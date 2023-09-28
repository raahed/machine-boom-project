/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_DATE_H
#define AGX_DATE_H

#include <agx/String.h>
#include <agx/Object.h>


namespace agx
{
  AGX_DECLARE_POINTER_TYPES(Date);
  class AGXCORE_EXPORT Date : public agx::Object
  {
  public:
    /**
    A copy of struct 'timeval' from BSD file sys/time.h.
    In windows, it is found in Winsock2.h.
    It is not part of the C++ standard, and under windows
    there are include problems in windows.h and Winsock2.h
    (compile errors if windows.h is included without WIN_32_LEAN_AND_MEAN
    before Winsock2.h).
    The easiest workaround seems to have our own struct.
    */
    struct TimeVal {
      long    tv_sec;         /* seconds */
      long    tv_usec;        /* and microseconds */
    };
    Date();
    Date(UInt seconds, UInt microSeconds = 0);


    /// \return The date formatted using strftime with format string "%Y.%m.%d-%H.%M.%S".
    agx::String toStringDateAndTime() const;

    /// \return The date formatted using strftime with the format string "%H:%M:%S".
    agx::String toStringTime() const;

    tm& getTmTime();
    const tm& getTmTime() const;

    TimeVal& getTimevalTime();
    const TimeVal& getTimevalTime() const;

    long long getMilliseconds() const;

    bool operator>(const Date& other) const;

    void update();


    /**
    Convert a date in the __DATE__ format ("Mmm dd yyyy"  -> "Aug 12 2012") into ISO standard: YYYY-MM-DD
    \param compileDate - The date in Mmm dd yyy format
    \param isoDate - The date in ISO standard format: YYYY-MM-DD
    \return true if date conversion was successful
    */
    static bool convertDateToISO(const agx::String& compileDate, agx::String& isoDate);


  protected:
    virtual ~Date();


  private:
    TimeVal m_timevalTime;
    tm m_tmTime;
  };


  /* Implementation */
  AGX_FORCE_INLINE tm& Date::getTmTime() { return m_tmTime; }
  AGX_FORCE_INLINE const tm& Date::getTmTime() const { return m_tmTime; }

  AGX_FORCE_INLINE Date::TimeVal& Date::getTimevalTime() { return m_timevalTime; }
  AGX_FORCE_INLINE const Date::TimeVal& Date::getTimevalTime() const { return m_timevalTime; }

}


#endif /* AGX_DATE_H */
