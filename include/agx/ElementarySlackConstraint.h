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

#include <agx/ElementaryConstraint.h>



#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agx
{

  /**
  Dot1Slack is similar to the Dot1 elementary constraint but also supports slack
  and uses twice as many equations in the solver.
  */
  class AGXPHYSICS_EXPORT Dot1Slack : public ElementaryConstraintNData< 2, Dot1SlackData >
  {
    public:
      /**
      Construct given Dot1SlackData.
      \param data - data for this elementary constraint
      */
      Dot1Slack( const agx::Dot1SlackData& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Dot1Slack);

      agx::Real getSlack() const;
      void setSlack( agx::Real slack );
    protected:
      /**
      Used during serialization restore.
      */
      Dot1Slack();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Dot1Slack();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };

  typedef agx::ref_ptr< Dot1Slack > Dot1SlackRef;


  /**
  Dot2Slack is similar to Dot2 elementary constraint.
  */
  class AGXPHYSICS_EXPORT Dot2Slack : public ElementaryConstraintNData< 2, Dot2SlackData >
  {
    public:
      /**
      Construct given Dot2SlackData.
      */
      Dot2Slack( const agx::Dot2SlackData& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Dot2Slack);

      agx::Real getSlack() const;
      void setSlack( agx::Real slack );

    protected:
      /**
      Used during serialization restore.
      */
      Dot2Slack();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Dot2Slack();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };


  /**
  SphericalRelSalck is similar to the SphericalRel elementary constraint, but also supports
  slack and uses twice as many equations in the solver.
  */
  class AGXPHYSICS_EXPORT SphericalRelSlack : public ElementaryConstraintNData< 6, Slack3ConstraintData >
  {
    public:
      /**
      Construct given SphericalData.
      */
      SphericalRelSlack( const agx::Slack3ConstraintData& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::SphericalRelSlack);


      agx::Vec3 getSlack() const;
      void setSlack( agx::Vec3 slack );

    protected:
      /**
      Used during serialization restore.
      */
      SphericalRelSlack();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~SphericalRelSlack();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };

  typedef agx::ref_ptr< SphericalRelSlack > SphericalRelSlackRef;
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
