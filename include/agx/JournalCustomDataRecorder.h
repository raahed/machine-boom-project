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

#ifndef AGX_JOURNAL_CUSTOM_DATA_RECORDER_H
#define AGX_JOURNAL_CUSTOM_DATA_RECORDER_H

#include <agx/agx.h>
#include <agxSDK/StepEventListener.h>
#include <agxSDK/Simulation.h>
#include <agx/Uuid.h>

namespace agx
{
  class Journal;

  AGX_DECLARE_POINTER_TYPES(JournalCustomDataRecorder);
  AGX_DECLARE_VECTOR_TYPES(JournalCustomDataRecorder);
  class AGXPHYSICS_EXPORT JournalCustomDataRecorder : public agxSDK::StepEventListener
  {
  public:
    /**
    Standard constructor for the JournalCustomDataRecorder class
    \param dataNodeName The name of the specified data node that the object will store the custom values under
    */
    JournalCustomDataRecorder(const agx::Name& dataNodeName);

    /**
    Called before the collision detection step is taken in the simulation
    \param time - the current simulation time
    */
    virtual void preCollide(const agx::TimeStamp& time) override;

    /**
    Called before a step is taken in the simulation
    \param time - the current simulation time
    */
    virtual void pre(const agx::TimeStamp& time) override;

    /**
    Called after a step is taken in the simulation
    \param time - the current simulation time
    */
    virtual void post(const agx::TimeStamp& time) override;

    /**
    Called when this object is added to a simulation
    */
    virtual void addNotification() override;

    /**
    Gets a custom real value to the simulation given a name identifier. This value should already exist in the simulation
    by calling addCustomValueReal.
    \param name The name identifier of the value
    \param initialValue The real value that should be set as an initial value
    */
    void addCustomValueReal(const agx::String& name, agx::Real initialValue);

    /**
    Gets a custom real value to the simulation given a name identifier. This value should already exist in the simulation
    by calling addCustomValueReal.
    \param name The name identifier of the value
    \return The real value
    */
    agx::Real getCustomValueReal(const agx::String& name) const;

    /**
    Set a custom real value to the simulation given a name identifier. This value should already exist in the simulation
    by calling addCustomValueReal.
    \param name The name identifier of the value
    \param val The real value that should be set
    \return true or false depending of the the value was set successfully or not.
    */
    bool setCustomValueReal(const agx::String& name, agx::Real val) const;

    /**
    Extracts all JournalCustomDataRecorders from a simulation
    \param recorders vector containing all the custom journal recorders that exist in the simulation
    \param simulation specified simulation that the method will search in
    */
    static bool extractJournalCustomDataRecorders(agx::JournalCustomDataRecorderPtrVector& recorders, agxSDK::Simulation* simulation);

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::JournalCustomDataRecorder);

  protected:
    void initStorageBuffer();

    void createJournalBindings(agx::Journal* journal);

    void journalAttachedCallback(agxSDK::Simulation*, agx::Journal*);

    virtual ~JournalCustomDataRecorder();

  private:
    JournalCustomDataRecorder();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxSDK::Simulation::JournalAttachEvent::CallbackType m_journalAttachedCallback;
    agx::HashTable<agx::Name, agxData::ValueRef> m_valueTable;
    agx::ComponentRef   m_dataRoot;
    agx::Name           m_dataNodeName;
    agx::Journal*       m_journal;
  };
}




#endif
