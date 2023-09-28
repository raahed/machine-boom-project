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

#ifndef AGXUNIT_UNITMACROS_H
#define AGXUNIT_UNITMACROS_H

#include <agx/config/AGX_USE_UNIT_TESTS.h>
#include <agx/Interrupt.h>
#include <agx/agxPhysics_export.h>
#include <agxIO/ArgumentParser.h>

namespace agxUnit
{
  // Query functions that can be used even when not building with unit tests enabled.

  /// \return true if any of the unit test command line arguments have been passed.
  bool inline haveUnittestingArguments(agxIO::ArgumentParser* arguments)
  {
    const bool unitTest = arguments->find("--unittest") != -1;
    const bool unitTestShort = arguments->find("-u") != -1;
    const bool sanityTest = arguments->find("--isLuaFileSanityUnitTest") != -1;
    return unitTest || unitTestShort || sanityTest;
  }



  /// \return true if any of the unit test command line arguments have been passed.
  bool inline haveUnittestingArguments(int argc, char** argv)
  {
    auto haveArgument = [argc, argv](const char* key)
    {
      auto begin = argv;
      auto end = argv + argc;
      auto strequal = [key](const char* arg) { return strcmp(arg, key) == 0; };
      auto it = std::find_if(begin, end, strequal);
      return it != end;
    };

    const bool unitTest = haveArgument("--unittest");
    const bool unitTestShort = haveArgument("-u");
    const bool sanityTest = haveArgument("--isLuaFileSanityUnitTest");
    return unitTest || unitTestShort || sanityTest;
  }
}


#if AGX_USE_UNIT_TESTS()
#include <agxUnit/agxUnit.h>


#ifdef _WIN32
#define AGXUNIT_FUNCTION __FUNCTION__
#else
#define AGXUNIT_FUNCTION __PRETTY_FUNCTION__
#endif

namespace agxUnit {
  /// Initializes the unit testing framework from an argument parser.
  void inline initUnittestFrameWork(agxIO::ArgumentParser* arguments)
  {
    agxUnit::UnitTestManager::instance()->parseArgs( arguments );
  }


  /// Initializes the unit testing framework from command line arguments.
  void inline initUnittestFrameWork(int argc, char** argv)
  {
    agxUnit::UnitTestManager::instance()->parseArgs( argc, argv );
  }


  /// Is unit testing enabled? (Has to be both built and initialized).
  bool inline isUnittestingEnabled()
  {
    return agxUnit::UnitTestManager::instance()->getEnable();
  }

  /// Should only fast unit tests be run?
  bool inline shouldOnlyFastUnittestsBeRun()
  {
    return agxUnit::UnitTestManager::instance()->getDoOnlyFastTests();
  }
}

#define AGXUNIT_POS_ARGS __LINE__,__FILE__, __FUNCTION__
#define AGXUNIT_SOURCECODEINFO() agxUnit::SourceCodeInfo( AGXUNIT_POS_ARGS )


#define AGXUNIT_BEGIN_SCOPE() if (1)
#define AGXUNIT_END_SCOPE() endif

#define AGXUNIT_PRINT_TEST_NAME() LOGGER_WARNING() << LOGGER_STATE(agx::Notify::PRINT_NONE) << "Running test " << __FUNCTION__ << "." << std::endl << LOGGER_END()


#define AGXUNIT_ASSERT(condition)                                                 \
  ( agxUnit::Assert::failIf( !(condition),                                   \
  agxUnit::AssertMessage( "agxAssertion failed",         \
  "Expression: " #condition), \
  AGXUNIT_SOURCECODEINFO() ) )


/** Assertion with a user specified message.
* \ingroup Assertions
* \param message Message reported in diagnostic if \a condition evaluates
*                to \c false.
* \param condition If this condition evaluates to \c false then the
*                  test failed.
*/
#define AGXUNIT_ASSERT_MESSAGE(message,condition)                          \
  ( agxUnit::Assert::failIf( !(condition),                            \
  agxUnit::AssertMessage( "agxAssertion failed", \
  "Expression: "      \
#condition,         \
  message ),          \
  AGXUNIT_SOURCECODEINFO() ) )

/** Fails with the specified message.
* \ingroup Assertions
* \param message Message reported in diagnostic.
*/
#define AGXUNIT_FAIL( message )                                         \
  ( agxUnit::Assert::fail( agxUnit::AssertMessage( "forced failure",  \
  message ),         \
  AGXUNIT_SOURCECODEINFO() ) )


/** Asserts that two values are equals.
* \ingroup Assertions
*
* Equality and string representation can be defined with
* an appropriate AGXUNIT::agxAssertion_traits class.
*
* A diagnostic is printed if actual and expected values disagree.
*
* Requirement for \a expected and \a actual parameters:
* - They are exactly of the same type
* - They are serializable into a std::strstream using operator <<.
* - They can be compared using operator ==.
*
* The last two requirements (serialization and comparison) can be
* removed by specializing the AGXUNIT::agxAssertion_traits.
*/
#define AGXUNIT_ASSERT_EQUAL(expected,actual)          \
  ( agxUnit::agxAssertEquals( (expected),              \
  (actual),                \
  AGXUNIT_SOURCECODEINFO() ) )

/** Asserts that two values are equals, provides additional message on failure.
* \ingroup Assertions
*
* Equality and string representation can be defined with
* an appropriate agxAssertion_traits class.
*
* A diagnostic is printed if actual and expected values disagree.
* The message is printed in addition to the expected and actual value
* to provide additional information.
*
* Requirement for \a expected and \a actual parameters:
* - They are exactly of the same type
* - They are serializable into a std::strstream using operator <<.
* - They can be compared using operator ==.
*
* The last two requirements (serialization and comparison) can be
* removed by specializing the AGXUNIT::agxAssertion_traits.
*/
#define AGXUNIT_ASSERT_EQUAL_MESSAGE(message,expected,actual)      \
  ( agxUnit::agxAssertEquals( (expected),              \
  (actual),                \
  AGXUNIT_SOURCECODEINFO(),    \
  (message) ) )


/*! \brief Macro for primitive double value comparisons.
* \ingroup Assertions
*
* The agxAssertion pass if both expected and actual are finite and
* \c fabs( \c expected - \c actual ) <= \c delta.
* If either \c expected or actual are infinite (+/- inf), the
* agxAssertion pass if \c expected == \c actual.
* If either \c expected or \c actual is a NaN (not a number), then
* the agxAssertion fails.
*/
#define AGXUNIT_ASSERT_DOUBLES_EQUAL(expected,actual,delta)        \
  ( agxUnit::agxAssertDoubleEquals( (expected),            \
  (actual),              \
  (delta),               \
  AGXUNIT_SOURCECODEINFO(),  \
  "" ) )


/*! \brief Macro for primitive double value comparisons, setting a
* user-supplied message in case of failure.
* \ingroup Assertions
* \sa AGXUNIT_ASSERT_DOUBLES_EQUAL for detailed semantic of the agxAssertion.
*/
#define AGXUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(message,expected,actual,delta)  \
  ( agxUnit::agxAssertDoubleEquals( (expected),            \
  (actual),              \
  (delta),               \
  AGXUNIT_SOURCECODEINFO(),  \
  (message) ) )

/*! \brief Macro for primitive double value comparisons, setting a
* user-supplied message in case of failure.
* \ingroup Assertions
* \sa AGXUNIT_ASSERT_DOUBLES_EQUAL for detailed semantic of the agxAssertion.
*/
#define AGXUNIT_ASSERT_DOUBLES_NOT_EQUAL_MESSAGE(message,expected,actual,delta)  \
  ( agxUnit::agxAssertDoubleNotEquals( (expected),            \
  (actual),              \
  (delta),               \
  AGXUNIT_SOURCECODEINFO(),  \
  (message) ) )

#define AGXUNIT_TRY_END(X) } \
  catch (agx::InterruptException&)   \
  {                                  \
    agx::shutdown();                 \
    exit(1);                         \
  }                                  \
  catch (agxUnit::Error& e) \
  { \
    agxUnit::AssertMessage msg("Error while running tests:", "" );        \
    msg.addMessage( agx::String("What: ") + e.what() );                                                    \
    agxUnit::UnitTestManager::instance()->reportError( msg, AGXUNIT_SOURCECODEINFO() );   \
    if (agxUnit::UnitTestManager::instance()->getPassOnExceptions()) \
      agxThrow e; \
    return X;\
  } \
  catch (agxUnit::Exception& e) \
  { \
    if (agxUnit::UnitTestManager::instance()->getPassOnExceptions()) \
      agxThrow e; \
    return X;\
   } \
   catch (std::exception& e) { \
     agxUnit::AssertMessage msg("Caught std::exception", typeid(e).name() );        \
     msg.addMessage( agx::String("What: ") + e.what() );                                                    \
     agxUnit::UnitTestManager::instance()->reportError( msg, AGXUNIT_SOURCECODEINFO() );   \
     if (agxUnit::UnitTestManager::instance()->getPassOnExceptions()) \
       agxThrow e; \
     return X;                                                                      \
   }\
catch (...) {                                                                      \
     agxUnit::AssertMessage msg("Caught undefined exception", "" );                 \
     agxUnit::UnitTestManager::instance()->reportError( msg, AGXUNIT_SOURCECODEINFO() );   \
     if (agxUnit::UnitTestManager::instance()->getPassOnExceptions()) \
       agxThrow std::exception(); \
     return X; \
   }


#define AGXUNIT_TRY_AND_CONTINUE_END() } \
  catch (agxUnit::Error& e) \
  { \
  agxUnit::AssertMessage msg("Error while running tests:", "" );        \
  msg.addMessage( agx::String("What: ") + e.what() );                                                    \
  agxUnit::UnitTestManager::instance()->reportError( msg, AGXUNIT_SOURCECODEINFO() );   \
  \
  } \
  catch (agxUnit::Exception& ) \
  { \
  \
   } \
   catch (std::exception& e) { \
   agxUnit::AssertMessage msg("Caught std::exception", typeid(e).name() );        \
   msg.addMessage( agx::String("What: ") + e.what() );                                                    \
   agxUnit::UnitTestManager::instance()->reportError( msg, AGXUNIT_SOURCECODEINFO() );   \
   \
   }\
   catch (...) {                                                                      \
   agxUnit::AssertMessage msg("Caught undefined exception", "" );                 \
   agxUnit::UnitTestManager::instance()->reportError( msg, AGXUNIT_SOURCECODEINFO() );   \
   \
   }

#define AGXUNIT_TRY_BEGIN() try {

#define AGXUNIT_BEGIN_TEST_GROUP(X)

#define AGXUNIT_END_TEST_GROUP(X)

#define AGXUNIT_BEGIN_TEST(X)
#define AGXUNIT_END_TEST(X)


#else // AGX_USE_UNIT_TESTS()

namespace agxUnit {
  /// Initializes the unit testing framework from an argument parser.
  void inline initUnittestFrameWork(agxIO::ArgumentParser* /*arguments*/)
  {
  }


  /// Initializes the unit testing framework from command line arguments.
  void inline initUnittestFrameWork(int /*argc*/, char** /*argv*/)
  {
  }


  /// Is unit testing enabled? (Has to be both built and initialized).
  bool inline isUnittestingEnabled()
  {
    return false;
  }


  /// Should only fast unit tests be run?
  bool inline shouldOnlyFastUnittestsBeRun()
  {
    return false;
  }
}

/// Start an #if 0 if UNITTEST IS NOT ENABLED
#define AGXUNIT_IF_HACK #if
#define AGXUNIT_ENDIF_HACK #endif
#define AGXUNIT_BEGIN_SCOPE() AGXUNIT_IF_HACK 0
#define AGXUNIT_END_SCOPE() AGXUNIT_ENDIF_HACK

/// Start a try scope for unittesting
#define AGXUNIT_TRY_END(X)

#define AGXUNIT_TRY_AND_CONTINUE_END()

/// End a try scope for unit testing, catches any exception thrown
#define AGXUNIT_TRY_BEGIN()

/// Begin a new group, not necessary to use
#define AGXUNIT_BEGIN_TEST_GROUP(X)

/// End a new group
#define AGXUNIT_END_TEST_GROUP(X)

/// Begin a new named test. Name (X) must be unique through all tests.
#define AGXUNIT_BEGIN_TEST(X)

/**
End a named test. Name (X) must be unique through all tests.
And it must match a BEGIN() with the same name
*/
#define AGXUNIT_END_TEST(X)

/** Report error if condition returns false */
#define AGXUNIT_ASSERT(condition)
/** Report error (with additional message) if condition returns false */
#define AGXUNIT_ASSERT_MESSAGE(message,condition)
/** Just fail with a message */
#define AGXUNIT_FAIL( message )
/** Report an error of expected != actual */
#define AGXUNIT_ASSERT_EQUAL(expected,actual)
/** Report an error (with additional message) if expected != actual */
#define AGXUNIT_ASSERT_EQUAL_MESSAGE(message,expected,actual)
/** Report an error if the difference between expected and actual exceeds delta */
#define AGXUNIT_ASSERT_DOUBLES_EQUAL(expected,actual,delta)
/** Report an error (with additional message) if the difference between expected and actual exceeds delta */
#define AGXUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(message,expected,actual,delta)

#define AGXUNIT_EMPTY_APPLICATION() \
  int main() \
{ \
\
  return 1; \
}


#endif // AGX_USE_UNIT_TESTS()
#endif // AGXUNIT_UNITMACROS_H
