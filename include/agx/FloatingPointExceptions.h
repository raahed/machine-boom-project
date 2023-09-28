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

#ifndef AGX_FLOATING_POINT_EXCEPTIONS_H
#define AGX_FLOATING_POINT_EXCEPTIONS_H


#include <agx/config.h>
#include <agx/agx.h>
#include <agx/agxCore_export.h>


namespace agx
{
  namespace FloatingPointExceptions {
    enum ExceptionCode {
      FP_EXC_NONE               = 0,
      FP_EXC_INVALID_OPERATION  = 1,        // Invalid operation (e.g., square root of a negative number - qNaN by default).
      FP_EXC_DIVISION_BY_ZERO   = 2, // Division by zero  (e.g., 1/0 or log(0) +- infinity by default)
      FP_EXC_OVERFLOW           = 4, // Overflow (a result is too large to be represented correctly) +- infinity by default).
      FP_EXC_UNDERFLOW          = 8, // Underflow (result too small and inexact) (returns a denormalized value by default).
      FP_EXC_INEXACT            = 16, // Inexact (returns correctly rounded result by default).
      FP_EXC_ALL                = FP_EXC_INVALID_OPERATION | FP_EXC_DIVISION_BY_ZERO | FP_EXC_OVERFLOW | FP_EXC_UNDERFLOW | FP_EXC_INEXACT
    };

    /**
    Sets global floating point exception settings for AGX, and applies them for current thread.
    Not thread-safe.
    Other threads will read this setting in their constructor and apply it using the
    applyFloatingPointExceptionSettings-function below.

    Note that these are hardware exceptions and will also be
    thrown by other binaries within the same process.

    For catching floating point exceptions, please see
    http://msdn.microsoft.com/en-us/library/te2k2f2t.aspx for visual studio
    and http://www.gnu.org/software/libc/manual/html_node/Program-Error-Signals.html
    on unix-like systems.

    \param mask A bitmask with settings according to the FloatingPointExceptions-namespace.
    */
    void AGXCORE_EXPORT setFloatingPointExceptionSettings(int mask);

    /**
    Gets the global floating point exception settings.
    See the ExceptionCode-enum.
    */
    int AGXCORE_EXPORT getFloatingPointExceptionSettings();

    /**
    Applies floating point exceptions as specified by the setFloatingPointExceptionSettings-function,
    but for current core.
    See setFloatingPointExceptionSettings for more information.
    */
    void AGXCORE_EXPORT applyFloatingPointExceptionSettings();

  }

}



#endif
