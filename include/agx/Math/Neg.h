/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


/////////////////////////////////////////////////////////////////////
// AUTOMATICALLY GENERATED, DO NOT EDIT! (except inline functions) //
/////////////////////////////////////////////////////////////////////

#ifndef AGXFN_MATH_NEG_H
#define AGXFN_MATH_NEG_H

#include <agxData/Array.h>
#include <agxData/EntityPtr.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/Math.h>
#include <agx/Job.h>
#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>


namespace agx { namespace Math { } }

namespace agxFn
{
  namespace Math
  {
    /**
    Function: Math.Neg
    Implementation: (default)

    \param result 
    \param arg 
    */
    template <typename AutoT1, typename AutoT2>
    AGX_FORCE_INLINE void Neg
    (
      /* Parameter list automatically generated, do not edit */
      AutoT1& result,
      const AutoT2& arg
    )
    {
      result = (AutoT1)(-arg);
    }

    /// Generated return-value wrapper. DO NOT EDIT!
    template <typename AutoT1, typename AutoT2>
    AGX_FORCE_INLINE AutoT1 Neg
    (
      const AutoT2& arg
    )
    {
      AutoT1 result;
      agxFn::Math::Neg(result, arg);
      return result;
    }
    


    /**
    Function: Math.Neg
    Implementation: (default)

    \param job The range job specifying what part of the data set to process
    \param result 
    \param arg 
    */
    template <typename AutoT1, typename AutoT2>
    AGX_FORCE_INLINE void Neg
    (
      /* Parameter list automatically generated, do not edit */
      const agx::RangeJob& job,
      agxData::Array< AutoT1 >& result,
      const agxData::Array< AutoT2 >& arg
    )
    {
      for (size_t i = job.range().begin(); i < job.range().end(); ++i)
      {
        agxFn::Math::Neg(result[i], arg[i]);
      }
    }


    /**
    Function: Math.Neg
    Implementation: (default)

    \param job The range job specifying what part of the data set to process
    \param result 
    \param arg 
    */
    template <typename AutoT1, typename AutoT2>
    AGX_FORCE_INLINE void Neg
    (
      /* Parameter list automatically generated, do not edit */
      const agx::RangeJob& job,
      agxData::Array< AutoT1 >& result,
      const AutoT2& arg
    )
    {
      for (size_t i = job.range().begin(); i < job.range().end(); ++i)
      {
        agxFn::Math::Neg(result[i], arg);
      }
    }


  }
}

#endif
