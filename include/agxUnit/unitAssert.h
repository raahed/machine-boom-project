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

#ifndef AGXUNIT_ASSERT_H
#define AGXUNIT_ASSERT_H

#include <agx/config.h>
#include <agx/config/AGX_USE_UNIT_TESTS.h>

#if AGX_USE_UNIT_TESTS()

#include <agx/agxPhysics_export.h>
#include <exception>
#include <string>
#include <sstream>
#include <vector>
#include <agxUnit/SourceCodeInfo.h>

DOXYGEN_START_INTERNAL_BLOCK()


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
# pragma warning(disable: 4275) // warning C4275: non dll-interface class 'std::exception' used as base for dll-interface class 'agxUnit::Exception'
#endif

namespace agxUnit
{

  class AGXPHYSICS_EXPORT AssertMessage
  {
  public:

    AssertMessage() {}

    AssertMessage(const agx::String& brief, const agx::String& msg1) : m_brief(brief) { m_msg.push_back(msg1); }
    AssertMessage(const agx::String& brief, const agx::String& msg1,
      const agx::String& msg2)  : m_brief(brief) { m_msg.push_back(msg1); m_msg.push_back(msg2); }
    AssertMessage(const agx::String& brief, const agx::String& msg1,
      const agx::String& msg2,
      const agx::String& msg3)  : m_brief(brief) { m_msg.push_back(msg1); m_msg.push_back(msg2); m_msg.push_back(msg3); }

    agx::String getMessage(size_t n) const { if (n >= m_msg.size()) return ""; return m_msg[n]; }
    size_t numMessages() const { return m_msg.size(); }
    void clear() { m_msg.clear(); }

    agx::String brief() const { return m_brief; }
    agx::String detailed() const;
    void addMessage( const agx::String& msg ) { m_msg.push_back( msg ); }
    void addMessage( const SourceCodeInfo &info )
    {
      m_msg.push_back(agx::String("File: ")+ info.filename());
      std::ostringstream str;
      str << "Line: " << info.line();
      m_msg.push_back(str.str());
      m_msg.push_back(agx::String("Function: ")+ info.function());
    }
    void addMessage( const AssertMessage &message )
    {
      m_msg.insert( m_msg.end(),
        message.m_msg.begin(),
        message.m_msg.end() );
    }


  protected:

    typedef std::vector<agx::String> DetailedVector;
    DetailedVector m_msg;
    agx::String m_brief;
  };

  class AGXPHYSICS_EXPORT AdditionalMessage : public AssertMessage
  {
  public:

    AdditionalMessage();

    AdditionalMessage( const agx::String &detail1 );

    AdditionalMessage( const char *detail1 );

    AdditionalMessage( const AssertMessage &other );

    AdditionalMessage &operator =( const AssertMessage &other );
  };


  class AGXPHYSICS_EXPORT Exception : public std::exception
  {
  public:
    Exception( const AssertMessage& msg, const SourceCodeInfo &info=SourceCodeInfo() );

    virtual ~Exception() throw();

    Exception( const Exception &copy );

    /// Assignment operator
    Exception &operator =( const Exception &copy );

    ///
    const char *what() const throw();
    const SourceCodeInfo& info() const { return m_info; }
    const AssertMessage& message() const { return m_message; }

  protected:
    AssertMessage m_message;
    SourceCodeInfo m_info;
    agx::String m_what;
  };

  class AGXPHYSICS_EXPORT Error : public Exception
  {
  public:
    Error( const AssertMessage& msg, const SourceCodeInfo &info=SourceCodeInfo() );

    virtual ~Error() throw();

    Error( const Error &copy );

    /// Assignment operator
    Error &operator =( const Error &copy );
  };


  struct  Assert
  {
    /*! \brief Throws a Exception with the specified message and location.
    */
    static void AGXPHYSICS_EXPORT fail( const AssertMessage &message,
      const SourceCodeInfo &sourceLine = SourceCodeInfo() );


    /*! \brief Throws a Exception with the specified message and location.
    * \param shouldFail if \c true then the exception is thrown. Otherwise
    *                   nothing happen.
    * \param message Message explaining the agxAssertion failure.
    * \param sourceLine Location of the agxAssertion.
    */
    static void AGXPHYSICS_EXPORT failIf( bool shouldFail,
      const AssertMessage &message,
      const SourceCodeInfo &sourceLine = SourceCodeInfo() );


    /*! \brief Returns a expected value string for a message.
    * Typically used to create 'not equal' message, or to check that a message
    * contains the expected content when writing unit tests for your custom
    * agxAssertions.
    *
    * \param expectedValue String that represents the expected value.
    * \return \a expectedValue prefixed with "Expected: ".
    * \see makeActual().
    */
    static agx::String AGXPHYSICS_EXPORT makeExpected( const agx::String &expectedValue );

    /*! \brief Returns an actual value string for a message.
    * Typically used to create 'not equal' message, or to check that a message
    * contains the expected content when writing unit tests for your custom
    * agxAssertions.
    *
    * \param actualValue String that represents the actual value.
    * \return \a actualValue prefixed with "Actual  : ".
    * \see makeExpected().
    */
    static agx::String AGXPHYSICS_EXPORT makeActual( const agx::String &actualValue );

#if 1
    static AssertMessage AGXPHYSICS_EXPORT makeNotEqualMessage( const agx::String &expectedValue,
      const agx::String &actualValue,
      const AdditionalMessage &additionalMessage = AdditionalMessage(),
      const agx::String &shortDescription = "equality agxAssertion failed");

    /*! \brief Throws an Exception with the specified message and location.
    * \param expected Text describing the expected value.
    * \param actual Text describing the actual value.
    * \param sourceLine Location of the agxAssertion.
    * \param additionalMessage Additional message. Usually used to report
    *                          what are the differences between the expected and actual value.
    * \param shortDescription Short description for the failure message.
    */
    static void AGXPHYSICS_EXPORT failNotEqual( agx::String expected,
      agx::String actual,
      const SourceCodeInfo &sourceLine,
      const AdditionalMessage &additionalMessage = AdditionalMessage(),
      agx::String shortDescription = "equality agxAssertion failed" );

    /*! \brief Throws an Exception with the specified message and location.
    * \param shouldFail if \c true then the exception is thrown. Otherwise
    *                   nothing happen.
    * \param expected Text describing the expected value.
    * \param actual Text describing the actual value.
    * \param sourceLine Location of the agxAssertion.
    * \param additionalMessage Additional message. Usually used to report
    *                          where the "difference" is located.
    * \param shortDescription Short description for the failure message.
    */
    static void AGXPHYSICS_EXPORT failNotEqualIf( bool shouldFail,
      agx::String expected,
      agx::String actual,
      const SourceCodeInfo &sourceLine,
      const AdditionalMessage &additionalMessage = AdditionalMessage(),
      agx::String shortDescription = "equality agxAssertion failed" );

#endif
  };
}

DOXYGEN_END_INTERNAL_BLOCK()
#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif // AGX_USE_UNIT_TESTS()

#endif  // AGXUNIT_UNITASSERT_H
