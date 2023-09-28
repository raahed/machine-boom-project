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

#ifndef AGX_SIMULATIONPARAMETER_H
#define AGX_SIMULATIONPARAMETER_H

#include <agxSDK/agxSDK.h>

#include <agx/Referenced.h>
#include <agx/Name.h>
#include <agxData/Format.h>
#include <agxData/Value.h>

namespace agxSDK
{
  class Simulation;

  AGX_DECLARE_POINTER_TYPES(SimulationParameter);
  AGX_DECLARE_VECTOR_TYPES(SimulationParameter);
  class AGXPHYSICS_EXPORT SimulationParameter : public agx::Referenced
  {
  public:
    SimulationParameter(const agx::Name& name);

    const agx::Name& getName() const;
    const agx::String& getDescription() const;

    void setName(const agx::Name& name);
    void setDescription(const agx::String& description);

    virtual const agxData::Format *getFormat() const = 0;
    virtual const agxData::Value *getValue() const = 0;
    virtual void setValue(agxData::Value *value) = 0;
    virtual bool hasSetter() const = 0;
    virtual bool hasGetter() const = 0;


    agxSDK::Simulation *getSimulation();

  protected:
    virtual ~SimulationParameter();

    friend  class Simulation;
    virtual void addNotification(agxSDK::Simulation *simulation);
    virtual void removeNotification(agxSDK::Simulation *simulation);

  private:
    agxSDK::Simulation *m_simulation;
    agx::Name m_name;
    agx::String m_description;
  };

  template <typename T>
  class SimulationParameterT : public SimulationParameter
  {
  public:
    SimulationParameterT(const agx::Name& name, const T& initialValue = T());


    virtual void set(const T& value);
    virtual const T& get() const;

    virtual const agxData::Format *getFormat() const override;
    virtual const agxData::Value *getValue() const override;
    virtual void setValue(agxData::Value *value) override;
    virtual bool hasSetter() const override;
    virtual bool hasGetter() const override;


  protected:
    virtual ~SimulationParameterT();

  private:
    agxData::ValueRefT<T> m_value;
  };


  /* Implementation */
  template <typename T>
  SimulationParameterT<T>::SimulationParameterT(const agx::Name& name, const T& initialValue) : SimulationParameter(name)
  {
    m_value = new agxData::ValueT<T>(name, initialValue);
  }


  template <typename T>
  SimulationParameterT<T>::~SimulationParameterT()
  {

  }

  template <typename T>
  const T& SimulationParameterT<T>::get() const
  {
    return m_value->get();
  }

  template <typename T>
  void SimulationParameterT<T>::set(const T& value)
  {
    m_value->set(value);
  }

  template <typename T>
  const agxData::Value *SimulationParameterT<T>::getValue() const
  {
    return m_value.get();
  }

  template <typename T>
  void SimulationParameterT<T>::setValue(agxData::Value *value)
  {
    set(value->get<T>());
  }

  template <typename T>
  bool SimulationParameterT<T>::hasSetter() const {
    return true;
  }

  template <typename T>
  bool SimulationParameterT<T>::hasGetter() const {
    return true;
  }

  template <typename T>
  const agxData::Format *SimulationParameterT<T>::getFormat() const
  {
    return m_value->getFormat();
  }

}

#endif //AGX_SIMULATIONPARAMETER_H
