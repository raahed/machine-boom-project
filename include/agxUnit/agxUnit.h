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

#ifndef AGXUNIT_AGXUNIT_H
#define AGXUNIT_AGXUNIT_H

#include <agx/config/AGX_USE_UNIT_TESTS.h>
#if AGX_USE_UNIT_TESTS()

#include <typeinfo>

#include <agx/agxPhysics_export.h>
#include <agxUnit/SourceCodeInfo.h>

#include <agx/Math.h>
#include <sstream>
#include <agxUnit/unitAssert.h>

namespace agxIO
{
  class ArgumentParser;
}

/**
\namespace agxUnit
\brief Contains classes/macros for handling unit tests within the AGX framework
*/
namespace agxUnit
{
  class AGXPHYSICS_EXPORT UnitTestManager// : agx::Singleton
  {
  public:
    UnitTestManager();

    static UnitTestManager *instance();

    void shutdown();

    void parseArgs( agxIO::ArgumentParser* arguments );
    void parseArgs(int& argc, char **argv );

    /// \return true if unit testing is enabled
    bool reportError( const AssertMessage &message, const SourceCodeInfo& info );

    void setEnable( bool flag ) { m_enable = flag; }
    bool getEnable( ) const { return m_enable; }

    void setDoOnlyFastTests( bool flag ) { m_doOnlyFastTests = flag; }
    bool getDoOnlyFastTests( ) const { return m_doOnlyFastTests; }

    void setPassOnExceptions( bool flag ) { m_passOnExceptions = flag; }
    bool getPassOnExceptions( ) const { return m_passOnExceptions; }

    ~UnitTestManager();

    //SINGLETON_CLASSNAME_METHOD();


  protected:


    bool m_enable;
    bool m_doOnlyFastTests;
    bool m_passOnExceptions;
  };

  template <class T>
  struct agxAssertion_utils
  {


    static bool equal(const T& a, const T&b)
    {
      return (a == b);
    }

    static agx::String toString( const T& x )
    {
      std::ostringstream ost;
      ost << x;
      return ost.str();
    }
  };


  /*! \brief Traits used by AGXUNIT_ASSERT_DOUBLES_EQUAL().
  *
  * This specialization from @c struct @c agxAssertion_traits<> ensures that
  * doubles are converted in full, instead of being rounded to the default
  * 6 digits of precision. Use the system defined ISO C99 macro DBL_DIG
  * within float.h is available to define the maximum precision, otherwise
  * use the hard-coded maximum precision of 15.
  */
  template <>
  struct agxAssertion_utils<double>
  {
    static bool equal(double a, double b)
    {
      return (a == b);
    }

    static agx::String toString( double x )
    {
      std::ostringstream str;

#ifdef DBL_DIG
      const int precision = DBL_DIG;
#else
      const int precision = 15;
#endif
      str.precision( precision );
      str << x;
      return str.str();
    }
  };

  template <typename T>
  void agxAssertEquals( const T& expected,
    const T& actual,
    SourceCodeInfo info,
    const agx::String &message = "" )
  {
    if ( !agxAssertion_utils<T>::equal(expected, actual ))
    {
      Assert::failNotEqual( agxAssertion_utils<T>::toString(expected),
        agxAssertion_utils<T>::toString(actual),
        info,
        message );
    }
  }

  /*! \brief (Implementation) Asserts that two double are not equal given a tolerance.
  * Use AGXUNIT_ASSERT_DOUBLES_NOT_EQUAL instead of this function.
  * \sa Assert::failNotEqual().
  * \sa AGXUNIT_ASSERT_DOUBLES_NOT_EQUAL for detailed semantic of the agxAssertion.
  */
  void AGXPHYSICS_EXPORT agxAssertDoubleNotEquals( double expected,
    double actual,
    double delta,
    SourceCodeInfo sourceLine,
    const agx::String &message );


  /*! \brief (Implementation) Asserts that two double are equals given a tolerance.
  * Use AGXUNIT_ASSERT_DOUBLES_EQUAL instead of this function.
  * \sa Assert::failNotEqual().
  * \sa AGXUNIT_ASSERT_DOUBLES_EQUAL for detailed semantic of the agxAssertion.
  */
  void AGXPHYSICS_EXPORT agxAssertDoubleEquals( double expected,
    double actual,
    double delta,
    SourceCodeInfo sourceLine,
    const agx::String &message );

} // agxUnit namespace

#else
#endif // AGX_USE_UNIT_TESTS()

#endif // AGXUNIT_AGXUNIT_H
