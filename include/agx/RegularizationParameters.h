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

#ifndef AGX_REGULARIZATION_PARAMETERS_H
#define AGX_REGULARIZATION_PARAMETERS_H

#include <agx/agxPhysics_export.h>
#include <agx/agx.h>
#include <agxStream/Serializable.h>

namespace agx
{
  /**
  All ghost variables have compliance and damping attributes.
  */
  class AGXPHYSICS_EXPORT RegularizationParameters : public agxStream::Serializable
  {
    public:
      enum VariableType
      {
        HOLONOMIC,
        NONHOLONOMIC,
        DEFAULT = HOLONOMIC
      };

    public:
      /**
      Default constructor. Default values: compliance = 1.0E-8, damping = 0.0333, type = HOLONOMIC.
      */
      RegularizationParameters();

      /**
      \return true if this parameter is associated to a holonomic equation - otherwise false
      */
      bool isHolonomic() const;

      /**
      Assign new type. This affects return value of \p getDiagonalPerturbation.
      \param type - new type
      */
      void setType( VariableType type );

      /**
      Set the value of compliance which is an inverse spring constant with
      suitable dimensionality given linear or angular variable. Default
      value: 1.0E-8, valid values: >= 0.
      \param c - compliance (valid if >= 0)
      */
      void setCompliance( agx::Real c );

      /**
      Set the damping rate with suitable dimensionality given linear or
      angular variable. Damping is used for holonomic variables only,
      and is a measure of the time the constraint variable have to
      fulfill the constraint definition (violation = 0).
      Default value: 0.0333, valid values: >= 0 (recommended >= time step).
      \note It is not recommended to set this value below the size
            if the integration step (time step). Damping < time step
            basically means that the solver has less than one
            time step to fulfill the constraint.
      */
      void setDamping( agx::Real d );

      /**
      \return the value of compliance which is an inverse spring constant
      */
      agx::Real getCompliance() const;

      /**
      \return the damping rate
      */
      agx::Real getDamping() const;

      /**
      Internal method used by solvers.
      \param h - the time step
      \return the diagonal perturbation given this variables type (holonomic or non-holonomic)
      */
      agx::Real getDiagonalPerturbation( agx::Real h ) const;

      /**
      Utility method to assign damping given time step used and the
      number of steps the solver gets to fulfill the constraint variable.
      \sa setDamping
      \param h - the time step
      \param numSteps - number of steps
      */
      void setDampingHalfLife( agx::Real h, agx::Real numSteps );

      /**
      Utility method to set compliance and assign damping given time step used and the
      number of steps the solver gets to fulfill the constraint variable.
      \sa setDamping
      \param compliance - compliance (valid if >= 0)
      \param h - the time step
      \param numSteps - number of steps (valid if > 0, recommended >= 1)
      */
      void setDampingHalfLife( agx::Real compliance, agx::Real h, agx::Real numSteps );

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::RegularizationParameters );


      /**
      Used by the solver, should in general not be called by user code.
      */
      void setIndexSetState( int8_t state );


      /**
      Used by the solver.
      */
      int8_t getIndexSetState() const;

    private:
      int  m_type;
      agx::Real m_compliance;
      agx::Real m_damping;
  };

  AGX_FORCE_INLINE bool RegularizationParameters::isHolonomic() const
  {
    return (m_type & 0xff) == HOLONOMIC;
  }

  AGX_FORCE_INLINE Real RegularizationParameters::getCompliance() const
  {
    return m_compliance;
  }

  AGX_FORCE_INLINE Real RegularizationParameters::getDamping() const
  {
    return m_damping;
  }

  AGX_FORCE_INLINE Real RegularizationParameters::getDiagonalPerturbation( Real h ) const
  {
    if ( isHolonomic() )
      return Real(4.0 / ( h * h * ( 1.0 + 4.0 * m_damping / h ) ) * m_compliance);
    else
      return Real(1.0) / h * m_compliance;
  }


  AGX_FORCE_INLINE int8_t RegularizationParameters::getIndexSetState() const
  {
    return (int8_t) ((m_type >> 8) & 0xff );
  }
} //namespace agx

#endif
