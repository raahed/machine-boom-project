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

#include <agx/Interaction.h>

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4589 ) // Disable: 4589: Constructor of abstract class 'agx::FrictionModel' ignores initializer for virtual base class 'agxStream::Serializable'
#endif


namespace agxRender
{
  class RenderManager;
}

namespace agx
{
  class SparseRangeReal;

  AGX_DECLARE_POINTER_TYPES( StrongInteraction );

  /**
  Base class for a strong interaction concept where there interaction
  is defined in the mass matrix part of the system matrix.
  */
  class AGXPHYSICS_EXPORT StrongInteraction : public agx::Interaction
  {
    public:

#if (!defined SWIG ) || (!defined SWIGCSHARP && !defined SWIGJAVA)
      struct AGXPHYSICS_EXPORT MatrixData
      {
        MatrixData( agx::Physics::RigidBodyPtr rb ) : rb( rb ), data( nullptr ), writeTranspose( false ), leadingDimension( agx::InvalidIndex ), blockRowIndex( agx::InvalidIndex ) {}

        template< typename T >
        void plusEqual( const T& rhs, const agx::UInt numRows, const agx::UInt numCols );

        agx::Physics::RigidBodyPtr rb;
        agx::Real*                 data;
        bool                       writeTranspose;
        unsigned int               leadingDimension; // Argument to Sabre, has to be 'unsigned int'.
        agx::UInt                  blockRowIndex;
      };
#endif

    public:
      /**
      Enable or disable this interaction.
      \param enable - true to enable, false to disable
      */
      void setEnable( agx::Bool enable );

      /**
      \return true when this interaction is enabled
      */
      agx::Bool getEnable() const;

      /**
      \return true if this interaction is enabled and valid
      */
      agx::Bool getValid() const;

    public:
      /**
      \return first rigid body
      */
      virtual agx::RigidBody* getRigidBody1() const = 0;

      /**
      \return second rigid body
      */
      virtual agx::RigidBody* getRigidBody2() const = 0;

      /**
      Called when added to a simulation.
      */
      virtual void addNotification() override;

      /**
      Called when removed from a simulation.
      */
      virtual void removeNotification() override;

      /**
      Called before solver executes.
      */
      virtual void preSystemCallback( agx::DynamicsSystem* dynamicsSystem ) override;

      /**
      Called after the solver is done.
      */
      virtual void postSystemCallback( agx::DynamicsSystem* dynamicsSystem ) override;

      /**
      Render the interaction.
      */
      virtual void render( agxRender::RenderManager* mgr, float scale );

      /**
      Prepares for solve.
      */
      virtual void prepare();

#if (!defined SWIG ) || (!defined SWIGCSHARP && !defined SWIGJAVA)
      /**
      Write data for dynamic bodies.
      */
      virtual void writeMatrixData( agx::StrongInteraction::MatrixData rb1DiagonalBlock,
                                    agx::StrongInteraction::MatrixData rb2DiagonalBlock,
                                    agx::StrongInteraction::MatrixData offDiagonalBlock ) = 0;

      /**
      Write data to the right hand side, for dynamic bodies.
      \param rb1 - first rigid body
      \param rb1Rhs - right hand side buffer (size 6)
      \param rb2 - second rigid body (invalid if single interaction)
      \param isRestingSolve - true and current velocity buffers should be used, false and
                              temporary velocity storage (acceleration buffer) should be used.
      */
      virtual void writeRhs( agx::Physics::RigidBodyPtr rb1,
                             agx::Real* rb1Rhs,
                             agx::Physics::RigidBodyPtr rb2,
                             agx::Real* rb2Rhs,
                             agx::Bool isRestingSolve ) = 0;
#endif

      /**
      Interface for non-linear callback from solver.
      */
      virtual void addNlCallbacks( agx::SparseRangeReal& sparseRangeReal ) const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::StrongInteraction );

    protected:
      /**
      Construct given one or two rigid bodies.
      */
      StrongInteraction();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~StrongInteraction();

      /**
      Valid if enable and configuration is ok for solver.
      \param valid - true if valid, otherwise false
      */
      void setValid( agx::Bool valid );

      /**
      Store local content.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore local content.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

    private:
      Bool                m_valid;
      Bool                m_enable;
  };

  template< typename T >
  void StrongInteraction::MatrixData::plusEqual( const T& rhs, const agx::UInt numRows, const agx::UInt numCols )
  {
    agxAssert( data != nullptr );
    if ( writeTranspose )
      for ( agx::UInt i = 0; i < numRows; ++i )
        for ( agx::UInt j = 0; j < numCols; ++j )
          data[ j * leadingDimension + i ] += rhs[ i ][ j ];
    else
      for ( agx::UInt i = 0; i < numRows; ++i )
        for ( agx::UInt j = 0; j < numCols; ++j )
          data[ i * leadingDimension + j ] += rhs[ i ][ j ];
  }
}

#if defined(_MSC_VER)
#pragma warning( pop ) // restoring: warning( disable : 4127 )
#endif
