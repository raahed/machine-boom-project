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

#ifndef AGX_DEBUG_H
#define AGX_DEBUG_H

#include <cassert>
#include <cstdlib>
#include <stdexcept> // for std::runtime_error
#include <cstdarg>
#include <cstdio>
#include <string>
#include <agx/agxCore_export.h>


#if defined(__GNUC__) || defined (__clang__)
  #define AGX_NO_RETURN __attribute__((__noreturn__))
#else
  #define AGX_NO_RETURN
#endif


namespace agx
{
  AGXCORE_EXPORT void abort() AGX_NO_RETURN;

  AGXCORE_EXPORT std::string buildErrorString(const char *baseFormat, std::string msgFormat, ...);

  AGXCORE_EXPORT const char *cStr(const char *str);
  AGXCORE_EXPORT const char *cStr(const std::string& str);

  AGXCORE_EXPORT void log_throw(const std::string& where, const char* what);

  AGXCORE_EXPORT int getExceptionNotifyLevel();
  AGXCORE_EXPORT int setExceptionNotifyLevel(int level);

  typedef std::runtime_error Error;
}



// Macro for defining a portable way of getting a nice function string including namespace::classname::method
#ifdef _MSC_VER
#define AGX_FUNCTION __FUNCTION__   // Won't include signature, i.e. will have trouble showing differences between overloaded functions
//#define AGX_FUNCTION __FUNCSIG__    // Verbose signature - closer to __PRETTY_FUNCTION__
#else
#define AGX_FUNCTION __PRETTY_FUNCTION__
#endif


#define  AGX_WHERE_AM_I  agx::buildErrorString("[%s:%u (%s)]", "", __FILE__, __LINE__, AGX_FUNCTION)

/*
The 'Throw' macro: works like 'throw', but logs the message as well as file, line and function.
*/

#if 0
// No need to use this path - the overhead of Throw is small, and exceptions (per definition) are rare
#define agxThrow throw
#else

DOXYGEN_START_INTERNAL_BLOCK()
namespace agx_internal
{
  /*
  template<typename E>
  const char* decodeException(const E& e) { return e.what(); }
  template<>
  inline const char* decodeException(const char* const & e) { return e; }
  */
  inline const char* decodeException(const std::exception& e) { return e.what();  }
  inline const char* decodeException(const char* e)           { return e;         }   // Don't Throw these!
  inline const char* decodeException(const std::string& e)    { return e.c_str(); }   // Don't Throw these!

  class ExceptionDescriber
  {
  public:
    ExceptionDescriber(const std::string& where)
      : m_where(where) {}

    template<typename E>
    const E& operator%(const E& e) const
    {
      agx::log_throw(m_where, decodeException(e));
      return e;
    }

  private:
    std::string m_where;
  };
}
DOXYGEN_END_INTERNAL_BLOCK()


// Acts like a normal Throw but logs where the exception was thrown from
#define  agxThrow  throw  agx_internal::ExceptionDescriber(AGX_WHERE_AM_I) %
#endif // Throw or no Throw


#undef __assert

#if !defined(AGX_DEBUG) && !defined(NDEBUG)
#define AGX_DEBUG
#endif

#ifdef _MSC_VER
  // 0,0 "hack" needed for visual studio. Produces warnings with other compilers..
  #define AGX_MACRO(x) do {x} while(0,0)
  #define agx_snprintf _snprintf_s
  #define agx_fprintf fprintf_s
#else
  #define AGX_MACRO(x) do {x} while(0)
  #define agx_snprintf snprintf
  #define agx_fprintf fprintf
#endif

// Verify state in both DEBUG and RELEASE build
#define agxVerify(expr)               AGX_MACRO(if (!(expr)) {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxVerify failed: `%s'", "", __FILE__, __LINE__, #expr).c_str()); agx::abort();})
#define agxVerify1(expr, msg)         AGX_MACRO(if (!(expr)) {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxVerify failed: `%s', ", "%s", __FILE__, __LINE__, #expr, agx::cStr(msg)).c_str()); agx::abort();})
#define agxVerifyN(expr, format, ...) AGX_MACRO(if (!(expr)) {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxVerify failed: `%s', ", format, __FILE__, __LINE__, #expr, ##__VA_ARGS__).c_str()); agx::abort();})

#define agxVerifyThrow(expr)               AGX_MACRO(if (!(expr)) {std::string errorMessage = agx::buildErrorString("[%s:%u] agxVerify failed: `%s'", "", __FILE__, __LINE__, #expr); agxThrow agx::Error(errorMessage);})
#define agxVerifyThrow1(expr, msg)         AGX_MACRO(if (!(expr)) {std::string errorMessage = agx::buildErrorString("[%s:%u] agxVerify failed: `%s', ", "%s", __FILE__, __LINE__, #expr, agx::cStr(msg)); agxThrow agx::Error(errorMessage);})
#define agxVerifyThrowN(expr, format, ...) AGX_MACRO(if (!(expr)) {std::string errorMessage = agx::buildErrorString("[%s:%u] agxVerify failed: `%s', ", format, __FILE__, __LINE__, #expr, ##__VA_ARGS__); agxThrow agx::Error(errorMessage);})



#ifdef AGX_DEBUG
#define agxAssertVoid()                AGX_MACRO(             {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAssert failed: `%s'", "", __FILE__, __LINE__).c_str()); agx::abort();})
#define agxAssert(expr)                AGX_MACRO(if (!(expr)) {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAssert failed: `%s'", "", __FILE__, __LINE__, #expr).c_str()); agx::abort();})
#define agxAssert1(expr, msg )         AGX_MACRO(if (!(expr)) {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAssert failed: `%s', ", "%s", __FILE__, __LINE__, #expr, agx::cStr(msg)).c_str()); agx::abort();})
#define agxAssertN(expr, format, ... ) AGX_MACRO(if (!(expr)) {fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAssert failed: `%s', ", format, __FILE__, __LINE__, #expr, ##__VA_ARGS__).c_str()); agx::abort();})

// Will always execute expr, but will only evaluate if AGX_DEBUG is set
#define agxDebugVerify(expr)                agxAssert(expr)
#define agxDebugVerify1(expr, msg )         agxAssert1(expr, msg)
#define agxDebugVerifyN(expr, format, ... ) agxAssertN(expr, format, __VA_ARGS__)
#else
#define  agxAssertVoid()                ((void)0)
#define  agxAssert(e)               ((void)0)
#define  agxAssert1(e, msg)         ((void)0)
#define  agxAssertN(e, format, ...) ((void)0)

// Will always execute expr, but will only evaluate if AGX_DEBUG is set
#define agxDebugVerify(expr)                 ((void)(expr))
#define agxDebugVerify1( expr, msg )         ((void)(expr))
#define agxDebugVerifyN( expr, format, ... ) ((void)(expr))

#endif

#define agxAbort( )             AGX_MACRO(fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAbort", "", __FILE__, __LINE__ ).c_str()); agx::abort();)
#define agxAbort1(msg)          AGX_MACRO(fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAbort: ", "%s", __FILE__, __LINE__, agx::cStr(msg)).c_str()); agx::abort();)
#define agxAbortN(format, ... ) AGX_MACRO(fprintf(stderr, "%s\n", agx::buildErrorString("[%s:%u] agxAbort: ", format, __FILE__, __LINE__, ##__VA_ARGS__).c_str()); agx::abort();)


#ifdef __GNUC__
#define AGX_ASM_COMMENT(X)  asm("#" X)
#else
#define AGX_ASM_COMMENT(X)
#endif



#endif /* _AGX_DEBUG_H_ */
