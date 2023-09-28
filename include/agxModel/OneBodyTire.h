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

#ifndef AGXMODEL_ONE_BODY_TIRE_H
#define AGXMODEL_ONE_BODY_TIRE_H

#include <agxModel/export.h>

#include <agxModel/Tire.h>
#include <agxSDK/Assembly.h>
#include <agxUtil/agxUtil.h>

namespace agxModel
{
  class TireConstraintImplementation;


  AGX_DECLARE_POINTER_TYPES(OneBodyTire);

  /**
  \note: This is an experimental feature, not fully tested yet.

  A simple tire model with one rigid body.
  Contacts are reduced to only support the deepest one.
  To model any deformation/elastic behavior, the contact material
  has to be adapted.
  Friction directions follow the tire frame
  (and can be set differently via the contact material).
  */
  class AGXMODEL_EXPORT OneBodyTire : public agxModel::Tire
  {
    public:
      /**
      Construct tire given rigid body (with wheel geometry/geometries) and radius.
      By definition tire axis is assumed to be y direction in body coordinates.
      This is also the secondary friction direction.
      \param rigidBody - rigid body of this tire (including geometries)
      \param radius - radius of this tire
      \param referenceFrame - reference frame for friction directions.
                     If none given, the rigidBody's will be used.
      */
      OneBodyTire( agx::RigidBody* rigidBody, agx::Real radius, agx::Frame* referenceFrame=nullptr );

      /**
      \return the rigid body of this tire
      */
      agx::RigidBody* getRigidBody();

      /**
      \return the rigid body of this tire
      */
      const agx::RigidBody* getRigidBody() const;

      // Only used for external queries
      class AGXMODEL_EXPORT ContactInfo
      {
        public:
          ContactInfo() {};
          ContactInfo(agx::RigidBody *body, agxCollide::LocalContactPointVector *contacts) : m_body(body), m_contacts(contacts) {}

          agx::RigidBody *getBody() { return m_body; }
          agxCollide::LocalContactPointVector *getContactPoints() { return m_contacts; }

        private:
          agx::RigidBody *m_body;
          agxCollide::LocalContactPointVector *m_contacts;
        };

      typedef agx::Vector<ContactInfo> ContactInfoVector;

      const ContactInfoVector& getContacts();

    protected:
      OneBodyTire();
      /**
      Class holding the constraints between ONE tire and an arbitrary number of other bodies.
      */
      class TireConstraintController : public agxUtil::ConstraintHolder
      {
        public:
          TireConstraintController();

          /**
          Creates and initializes a constraint between the two bodies, or returns an already existing
          tire constraint between the two.
          \param rbTire - tire rigid body
          \param rbOther - other rigid body (valid if 0 but has to be != 0 for static rigid bodies)
          */
          TireConstraintImplementation* getOrCreateConstraint( agx::RigidBody* rbTire, agx::RigidBody* rbOther );

          /**
          All sub-constraints will be created. Call this method before solve.
          \param refFrame - some reference frame
          \param radius - radius of the tire
          */
          void prepare( const agx::Frame* refFrame, agx::Real radius );

          /**
          Clears and collects values that are used over time. Call this method after the solver is done.
          */
          void finalize( agx::Real h );

        protected:
          virtual ~TireConstraintController();
      };

      typedef agx::ref_ptr< TireConstraintController > TireConstraintControllerRef;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::OneBodyTire);
    protected:
      virtual ~OneBodyTire();

      virtual void addNotification( agxSDK::Simulation* simulation ) override;
      using agxSDK::Assembly::addNotification;
      virtual void removeNotification( agxSDK::Simulation* simulation ) override;
      using agxSDK::Assembly::removeNotification;

      friend class agxUtil::GeneralStepListener< OneBodyTire >;
      virtual void preCollide( const agx::TimeStamp& );
      virtual void pre( const agx::TimeStamp& );
      virtual void post( const agx::TimeStamp& );

      friend class agxUtil::GeneralContactEventListener< OneBodyTire >;
      virtual agxSDK::ContactEventListener::KeepContactPolicy impact( const agx::TimeStamp&, agxCollide::GeometryContact* gc ) { return handleContact( gc ); }
      virtual agxSDK::ContactEventListener::KeepContactPolicy contact( const agx::TimeStamp&, agxCollide::GeometryContact* gc ) { return handleContact( gc ); }
      virtual void                                            separation( const agx::TimeStamp&, agxCollide::GeometryPair& ) {}
      virtual agxSDK::ContactEventListener::KeepContactPolicy handleContact( agxCollide::GeometryContact* gc );

    protected:
      agx::ref_ptr< agxUtil::GeneralStepListener< OneBodyTire > > m_stepCallback;
      agx::ref_ptr< agxUtil::GeneralContactEventListener< OneBodyTire > > m_contactCallback;

      agx::RigidBodyRef             m_rigidBody;
      agx::Real                     m_radius;
      agx::FrameRef                 m_refFrame;
      TireConstraintControllerRef   m_constraintController;
      agx::Vector<ContactInfo>      m_contacts; // Only used for external queries
  };
}

#endif // AGXMODEL_ONE_BODY_TIRE_H
