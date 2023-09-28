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

#include <agxSDK/StepEventListener.h>

#include <agx/Constraint.h>

namespace agxCollide
{
  class GeometryContact;
}

namespace agxSDK
{
  class ExecuteFilter;
}

namespace agx
{
  class ElementaryContactPointFactory;

  AGX_DECLARE_POINTER_TYPES( GeometryContactConstraint );

  /**
  Class matching geometry contacts given an execute filter and is solving each
  matching contact point with the given elementary constraint implementation.
  Any added geometry contact will not be solved in the default contact solver,
  it will instead be solved using ordinary constraints. The solver data, such
  as forces and tangents, will be written back to the geometry contact.

  Note that this class currently only supports being a member of a custom
  agx::FrictionModel implementation with HAS_CUSTOM_CONTACT_IMPLEMENTATION callback
  mask set.
  */
  class AGXPHYSICS_EXPORT GeometryContactConstraint : public agxSDK::StepEventListener
  {
    public:
      /**
      Construct given execute filter matching IMPACT | CONTACT geometry contacts and
      elementary constraint factory with implementation specific constraints for each
      contact point.
      \param contactFilter - execute filter matching geometry contacts to be solved by this constraint
      \param elementaryFactory - elementary constraint factory, one elementary constraint for each
                                 enabled contact point in matched geometry contacts.
      */
      GeometryContactConstraint( agxSDK::ExecuteFilter* contactFilter,
                                 ElementaryContactPointFactory* elementaryFactory );

      /**
      \return the elementary constraint factory of this constraint
      */
      ElementaryContactPointFactory* getElementaryFactory() const;

      /**
      Match if the given geometry contact should be solved using this constraint.
      \sa add
      \param geometryContact - geometry contact to match
      \return true if the given geometry contact is a match, otherwise false
      */
      Bool match( agxCollide::GeometryContact* geometryContact ) const;
      
      /**
      Add geometry contact to be solved using this constraint. It's assumed the
      add geometry contact won't reach the contact solver, e.g., by being removed
      or having a friction model with HAS_CUSTOM_CONTACT_IMPLEMENTATION set.
      \param geometryContact - geometry contact to add and be solved using this constraint
      */
      void add( agxCollide::GeometryContact* geometryContact );

    public:
      /**
      Adding the constraint (agx::Constraint) to the simulation this
      instance has been added to.
      */
      virtual void addNotification() override;

      /**
      Removing the constraint (agx::Constraint) from the simulation this
      instance has been removed from.
      */
      virtual void removeNotification() override;

      /**
      Clearing constraint data and debug rendering (if enabled).
      */
      virtual void post( const TimeStamp& ) override;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~GeometryContactConstraint();

    private:
      ConstraintRef m_constraints;
  };
}
