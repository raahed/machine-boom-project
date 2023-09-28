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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

// Copyright 2007-2010 Baptiste Lepilleur
// Distributed under MIT license, or public domain if desired and
// recognized in your jurisdiction.
// See file LICENSE for detail or copy at http://jsoncpp.sourceforge.net/LICENSE

#ifndef AGX_CPPTL_JSON_ASSERTIONS_H_INCLUDED
# define AGX_CPPTL_JSON_ASSERTIONS_H_INCLUDED

#include <stdlib.h>
#include <agx/debug.h> // Throw


#if !defined(JSON_IS_AMALGAMATION)
# include <external/json/config.h>
#endif // if !defined(JSON_IS_AMALGAMATION)

#if JSON_USE_EXCEPTION
#define JSON_ASSERT( condition ) assert( condition );  // @todo <= change this into an exception throw
#define JSON_FAIL_MESSAGE( message ) agxThrow std::runtime_error( message );
#else  // JSON_USE_EXCEPTION
#define JSON_ASSERT( condition ) assert( condition );

// The call to assert() will show the failure message in debug builds. In
// release bugs we write to invalid memory in order to crash hard, so that a
// debugger or crash reporter gets the chance to take over. We still call exit()
// afterward in order to tell the compiler that this macro doesn't return.
#define JSON_FAIL_MESSAGE( message ) { assert(false && message); strcpy(reinterpret_cast<char*>(666), message); exit(123); }

#endif

#define JSON_ASSERT_MESSAGE( condition, message ) if (!( condition )) { JSON_FAIL_MESSAGE( message ) }

#endif // AGX_CPPTL_JSON_ASSERTIONS_H_INCLUDED
