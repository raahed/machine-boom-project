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

#ifndef AGX_STREAM_H
#define AGX_STREAM_H


#include <agx/Model.h>
#include <agx/Component.h>
#include <agxNet/FramePacket.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(EventModel);

  /**
  TODO: Documentation
  */
  class AGXCORE_EXPORT EventModel : public agx::Model
  {
  public:
    // EventModel(const agx::Name& name);

  protected:
    // virtual ~EventModel();
    EventModel()  : Model("", nullptr) {}
  private:
  };


  AGX_DECLARE_POINTER_TYPES(Stream);

  /**
  TODO: Documentation
  */
  class AGXCORE_EXPORT Stream : public agx::Object
  {
  public:
    AGX_DECLARE_POINTER_TYPES(Event);
    AGX_DECLARE_POINTER_TYPES(DataEvent);
    AGX_DECLARE_POINTER_TYPES(FunctionEvent);

    // Stream(const agx::Name& name);

  protected:
    // virtual ~Stream();

  private:
  };

  //---------------------------------------------------------------

  /**
  TODO: Documentation
  */
  class AGXCORE_EXPORT Stream::Event : public agx::Referenced
  {
  public:
    Event();

  protected:
    virtual ~Event();

  protected:
    agx::EventModelRef m_model;
  };

  //---------------------------------------------------------------

  /**
  TODO: Documentation
  */
  class AGXCORE_EXPORT Stream::DataEvent : public Event
  {
  public:

  private:
  };

  #if 0
  class AGXCORE_EXPORT Stream::AttributeDataEvent : public DataEvent
  {
  public:

  private:
    agx::UInt32 m_id;
    agxData::Attribute *m_attribute;
    agxData::EntityStorage *m_storage;
    agxData::ValueRef m_value;
  };
  #endif

  //---------------------------------------------------------------

  /**
  TODO: Documentation
  */
  class AGXCORE_EXPORT Stream::FunctionEvent : public Event
  {
  public:
    FunctionEvent(agx::Model *kernelModel);
    FunctionEvent(const agxNet::StructuredMessage *message);

    /// eg 'Add'
    // const agx::Name& getName() const;

    /// eg. 'Math.Add', should return agx::Path?
    // agx::String fullName() const;

    /**
    Evaluate the function with the registered parameters.
    */
    void evaluate();

    /// TODO: Common baseclass, agxData::Value -> agx::Scalar, agx::Buffer (agx::Buffer -> agx::Array<T> ?)
    const Object *getParameter(const Name& name) const;

  protected:
    // agxData::Value *restoreParameter(const Name& name);
    // agxData::Buffer *restoreBuffer(const Name& name);

    virtual void setup() {};

  private:
    typedef HashTable<Name, ObjectRef> ParameterTable;
    ParameterTable m_parameters;
    Callback m_implementation;
    TaskRef m_task; // TODO Remove
  };


  #if 0
  class SetRadiusEvent : public agx::Stream::FunctionEvent
  {
  public:
    SetRadiusEvent("Physics.Particle.SetRadius");

    void evaluate()
    {
      agx::Physics::ParticleData& particle = *static_cast<agx::Physics::ParticleData *>(m_particle->getData());
      const agx::UInt& index m_index->get<agx::UInt>();
      const agx::Real& radius = m_radius->get<agx::Real>();

      agxFn::Physics::Particle::SetRadius(particle, index, radius);

      m_radiusBuffer->commit(m_index);
      m_massBuffer->commit(m_index);
      m_invMassBuffer->commit(m_index);
      m_materialBuffer->commit(m_index);
    }

  protected:
    virtual void setup()
    {
      m_particle = this->getParameter<agxData:EntityStorage>("particle");
      m_index = this->getParameter<agxData::Value>("index");
      m_radius = this->getParameter<agxData::Value>("radius");

      // Implicit parameters, should not be serialized? No need to, they are part of ParticleData!
      // m_material = this->getParameter<agxData::Value>("material");
      // m_mass = this->getParameter<agxData::Value>("mass");
      // m_invMass = this->getParameter<agxData::Value>("invMass");

      m_radiusBuffer = m_particle->getBuffer("radius");
      m_massBuffer = m_particle->getBuffer("mass");
      m_invMassBuffer = m_particle->getBuffer("invMass");
      m_materialBuffer = m_particle->getBuffer("material");
    }

  private:
    agxData::EntityStorageRef m_particle;
    agxData::ValueRef m_radius;
    agxData::ValueRef m_index;

    agxData::Buffer *m_radiusBuffer;
    agxData::Buffer *m_massBuffer;
    agxData::Buffer *m_invMassBuffer;
    agxData::Buffer *m_materialBuffer;
  };
  #endif
  //---------------------------------------------------------------

  /* Implementation */
  // AGX_FORCE_INLINE const agx::Name& Stream::FunctionEvent::getName() const { return m_model->getName(); }
  // AGX_FORCE_INLINE const agx::String& Stream::FunctionEvent::fullName() const { return m_model->fullPath(); }
  AGX_FORCE_INLINE const Object *Stream::FunctionEvent::getParameter(const Name& name) const
  {
    ParameterTable::const_iterator it = m_parameters.find(name);
    return it != m_parameters.end() ? it->second : nullptr;
  }

}

#endif /* AGX_STREAM_H */
