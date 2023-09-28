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

#ifndef AGXOSG_PRESSURE_FROM_CONTACTS_H
#define AGXOSG_PRESSURE_FROM_CONTACTS_H

#include <agxCollide/Trimesh.h>

#include <agxSDK/StepEventListener.h>
#include <agxSDK/ContactEventListener.h>
#include <agx/ParticleContactSensor.h>

#include <agxOSG/export.h>
#include <agxOSG/PressureGenerator.h>

namespace agxOSG
{

  class PressureAtlas;

  AGX_DECLARE_POINTER_TYPES(PressureFromContacts);


  /**
   * Generates pressure on a mesh from contacts.
   */
  class AGXOSG_EXPORT PressureFromContacts : public agxOSG::PressureGenerator
  {
  public:
    /// The type of force to add. Tangential (friction) only, normal only, or both.
    enum ForceType {
      TANGENTIAL_FORCE,
      NORMAL_FORCE,
      TOTAL_FORCE
    };

    typedef agx::Vector <agx::Physics::ParticleGeometryContactInstance> ParticleContactVector;

    /**
     * @param simulation The simulation which is generating contacts.
     * @param atlas The pressure atlas to which pressure should be added.
     * @param rangeFactor Relationship between a contact force and area of the pressure circle. In m/N.
     * @param forceType The type of the force to add. Tangential (friction) only, normal only, or both.
     */
    PressureFromContacts(agxSDK::Simulation* simulation, agxOSG::PressureAtlas* atlas,
      agx::Real rangeFactor = 2.0e-5,
      ForceType forceType = TOTAL_FORCE);

    void setRangeFactor(agx::Real rangeFactor);
    agx::Real getRangeFactor() const;

    /**
     * Add a contact filter. A contact will only contribute pressure if it is
     * accepted by all filters.
     *
     * @param filter
     */
    void addFilter(agxSDK::ExecuteFilter* filter);

  protected:
    virtual ~PressureFromContacts();

  private:
    AGX_DECLARE_POINTER_TYPES(Gatherer);
    AGX_DECLARE_POINTER_TYPES(ParticleGatherer);
    AGX_DECLARE_POINTER_TYPES(Adder);
    AGX_DECLARE_POINTER_TYPES(ContactFilter);

    ContactFilterRef m_filter;
    GathererRef m_gatherer;
    ParticleGathererRef m_particleGatherer;
    AdderRef m_adder;

  private:
    friend class ParticleGatherer;
    friend class Gatherer;
    friend class Adder;
    agxCollide::GeometryContactPtrVector m_contacts;
    ParticleContactVector m_particleContacts;
    agx::Real m_rangeFactor;

  };


  /**
   * Filter that determines which contacts should be used to contribute
   * pressure. Consists of a set of filters which all must accept the contact
   * for it to be used.
   */
  class AGXOSG_EXPORT PressureFromContacts::ContactFilter : public agxSDK::ExecuteFilter
  {
  public:
    ContactFilter(const agxCollide::Trimesh* mesh);

    virtual bool match(const agxCollide::GeometryContact& contact) const override;
    virtual bool match(const agxCollide::GeometryPair& geometryPair) const override;
    using agxSDK::ExecuteFilter::match;

    void addFilter(agxSDK::ExecuteFilter* filter);

  protected:
    virtual ~ContactFilter() {}

  private:
    agx::Vector<agxSDK::ExecuteFilterRef> m_filters;
  };



  /**
   * Filter that only accepts contacts where at least on of the bodes in the
   * contact has a particular name.
   */
  class FilterContactByName : public agxSDK::ExecuteFilter
  {
  public:
    FilterContactByName(const char* name) : m_name(name)
    {
    }
    virtual bool match(const agxCollide::GeometryContact& contact) const override
    {
      return this->match(const_cast<agxCollide::GeometryContact&>(contact).getGeometryPair());

    }
    virtual bool match(const agxCollide::GeometryPair& pair) const override
    {
      return this->checkName(pair.first) || this->checkName(pair.second);
    }
    using agxSDK::ExecuteFilter::match;
  private:
    bool checkName(const agxCollide::Geometry* geometry) const
    {
      return geometry->getRigidBody() && geometry->getRigidBody()->getName() == m_name;
    }

  protected:
    virtual ~FilterContactByName() {}

  private:
    agx::Name m_name;
  };

  class AGXOSG_EXPORT PressureFromContacts::Gatherer : public agxSDK::ContactEventListener
  {
  public:
    Gatherer(agxOSG::PressureFromContacts* master);

    virtual agxSDK::ContactEventListener::KeepContactPolicy
    impact(const agx::TimeStamp&, agxCollide::GeometryContact* contact) override;

    virtual agxSDK::ContactEventListener::KeepContactPolicy
    contact(const agx::TimeStamp&, agxCollide::GeometryContact* contact) override;

  protected:
    virtual ~Gatherer() {};

  private:
    agxOSG::PressureFromContacts* m_master;
  };

  /// Not working right now.
  class AGXOSG_EXPORT PressureFromContacts::ParticleGatherer : public agx::ParticleContactSensor
  {
  public:
    ParticleGatherer(agxOSG::PressureFromContacts* master);

    /// Not working right now.
    virtual void contactCallback(agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData, agx::Physics::GeometryData& geometryData) override;

  protected:
    virtual ~ParticleGatherer() {};

  private:
    agxOSG::PressureFromContacts* m_master;
  };

  class AGXOSG_EXPORT PressureFromContacts::Adder : public agxSDK::StepEventListener
  {
  public:

    Adder(agxOSG::PressureFromContacts* master, agxOSG::PressureAtlas* atlas, PressureFromContacts::ForceType forceType);

    virtual void post(const agx::TimeStamp& time) override;

  protected:
    virtual ~Adder() {}

  private:
    void applyPressureFromGeometryContact(agxCollide::GeometryContact* contact);
    void applyPressureFromPoint(const agxCollide::Trimesh* mesh, agx::AffineMatrix4x4 worldToMesh, agxCollide::ContactPoint point);
    void applyPressureFromGeometryParticleContact(const agxCollide::Trimesh* mesh, agx::AffineMatrix4x4 worldToMesh, agx::Physics::ParticleGeometryContactInstance point);
    void applyContactForce(const agx::Vec3& point, agx::UInt32 triangleIndex, agx::Real contactForce);


  private:
    agxOSG::PressureFromContacts* m_master;
    agxOSG::PressureAtlas* m_atlas;
    PressureFromContacts::ForceType m_forceType;
  };

}


#endif
