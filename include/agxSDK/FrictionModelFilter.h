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

#pragma once

#include <agxSDK/ExecuteFilter.h>

#include <agx/FrictionModel.h>

namespace agxSDK
{
  /**
  Execute filter matching friction model instance of the geometry contact.
  Note that this execute filter won't match separating geometries (for
  separation events).
  */
  class AGXPHYSICS_EXPORT FrictionModelFilter : public agxSDK::ExecuteFilter
  {
    public:
      /**
      Construct given friction model instance. If nullptr this filter
      won't match any contact.
      \param frictionModel - friction model instance to match
      */
      FrictionModelFilter( const agx::FrictionModel* frictionModel );

    public:
      virtual bool match( const agxCollide::GeometryContact& gc ) const override;

      using agxSDK::ExecuteFilter::match;

    protected:
      virtual ~FrictionModelFilter();

    private:
      agx::FrictionModelConstObserver m_frictionModel;
  };

  inline bool FrictionModelFilter::match( const agxCollide::GeometryContact& gc ) const
  {
    const auto cm = gc.getMaterial( true );
    return cm != nullptr &&
           m_frictionModel != nullptr &&
           cm->getFrictionModel() == m_frictionModel;
  }
}
