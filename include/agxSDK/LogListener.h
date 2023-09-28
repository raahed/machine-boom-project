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

#ifndef AGXSDK_LOGLISTENER_H
#define AGXSDK_LOGLISTENER_H

#include <agxSDK/agxSDK.h>
#include <agxSDK/StepEventListener.h>
#include <string>
#include <sstream>

namespace agxSDK
{

  /// Abstract class for logging properties per time step to a file/standard out
  class AGXPHYSICS_EXPORT LogListener : public StepEventListener
  {
  public:

    enum NumberFormat {
      DEFAULT,
      SCIENTIFIC,
      FIXED
    };

     /**
     LogListener constructor.
    \param fileName File name for logging. If empty, will write to standard out.
          If file exists, it will be overwritten.
    \param newFilePerTimeStep Should each time step be logged to a new file?
    \param valueSeparator String for separating the different columns
    */
    LogListener(const std::string& fileName = "",
      bool newFilePerTimeStep = false,
      const std::string& valueSeparator = "\t");

    /// inherited from StepEventListener
    virtual void pre(const agx::TimeStamp& /*time*/);
    /// inherited from StepEventListener
    virtual void post( const agx::TimeStamp& /*time*/ );

    inline void setFileName( const std::string& fileName );
    inline void setNewFilePerTimeStep( bool newFilePerTimeStep );
    inline void setValueSeparator( const std::string& valueSeparator );
    inline void setPrecision( int precision );
    inline void setNumberFormat( NumberFormat format );

    inline std::string getFileName() const;
    inline bool getNewFilePerTimeStep() const;
    inline std::string getValueSeparator() const;
    inline int getPrecision() const;
    inline NumberFormat getNumberFormat() const;

  protected:
    /// Implement this function when inheriting
    virtual void obtainHeader( std::stringstream& /*header*/ ) {}
    /// Implement this function when inheriting
    virtual void collectData ( std::stringstream& /*data*/, const agx::TimeStamp& /*time*/ ) {}

  private:
    void writeToFile( const std::string& text, const agx::TimeStamp& time, bool append );


  protected:
    std::string m_fileName;
    bool m_newFilePerTimeStep;
    bool m_hasTakenFirstStep;
    int m_precision;
    std::string m_separator;
    NumberFormat m_format;
  };

  typedef agx::ref_ptr<LogListener> LogListenerRef;


  inline void LogListener::setFileName( const std::string& fileName )
  {
    m_fileName = fileName;
  }

  inline void LogListener::setNewFilePerTimeStep( bool newFilePerTimeStep )
  {
    m_newFilePerTimeStep = newFilePerTimeStep;
  }

  inline void LogListener::setValueSeparator( const std::string& valueSeparator )
  {
    m_separator = valueSeparator;
  }

  inline void LogListener::setPrecision( int precision )
  {
    m_precision = precision;
  }

  inline void LogListener::setNumberFormat( LogListener::NumberFormat format )
  {
    m_format = format;
  }

  inline int LogListener::getPrecision() const
  {
    return m_precision;
  }

  inline bool LogListener::getNewFilePerTimeStep() const
  {
    return m_newFilePerTimeStep;
  }

  inline std::string LogListener::getValueSeparator() const
  {
    return m_separator;
  }

  inline std::string LogListener::getFileName() const
  {
    return m_fileName;
  }

  inline LogListener::NumberFormat LogListener::getNumberFormat() const
  {
    return m_format;
  }

}

#endif /* AGXSDK_LOGLISTENER_H */
