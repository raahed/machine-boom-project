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

#include <agx/NlmcpCallback.h>

namespace agx
{
  class FrictionController;

  /**
  Solver callback class for elementary FrictionController.
  */
  class FrictionControllerNlCallback : public agx::ConstraintNlmcpCallback
  {
    public:
      struct ContextData
      {
        ContextData()
        {
          rowOffset = agx::InvalidIndex;
          ecIndices[ 0 ] = ecIndices[ 1 ] = agx::InvalidIndex;
        }

        agx::UInt rowOffset;
        agx::Vec2u ecIndices;
      };


    public:
      static ContextData findContextData( const agx::ConstraintImplementation* context );
      static agx::Real calculateNormalForce( const agx::ConstraintImplementation* context, ContextData contextData );

    public:
      /**
      Default constructor.
      */
      FrictionControllerNlCallback( const agx::FrictionController* frictionController );

      agx::Real calculateNormalForce() const;
      agx::Real calculateNormalForce( const agx::ConstraintImplementation* context ) const;

      virtual agx::Bool initialize( const agx::NlmcpCallback::Args& args ) override;

      virtual agx::Real calculateResidual( const agx::NlmcpCallback::Args& args ) const override;

      virtual void update( const agx::NlmcpCallback::Args& args ) const override;

      virtual void postIterativeSolve( const agx::NlmcpCallback::Args& /*args*/ ) const override;

    protected:
      virtual ~FrictionControllerNlCallback();

      virtual void onSetContext( const NlmcpCallbackSolverData& ) override;

    protected:
      agx::Real calculateNormalImpact( const agx::RealValarray& z ) const;

    private:
      const FrictionController* m_frictionController;
      ContextData m_contextData;
  };
}
