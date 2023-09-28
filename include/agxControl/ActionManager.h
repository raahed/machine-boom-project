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

/// \cond CONTROL


#ifndef AGXCONTROL_ACTION_MANAGER_H
#define AGXCONTROL_ACTION_MANAGER_H

#include <agx/config.h>

#include <agxControl/Action.h>
#include <agxControl/Operation.h>
#include <agxSDK/EventListener.h>
#include <agx/List.h>
#include <agx/TimeStamp.h>


/**

\internal

A few things to consider.
 - Simulation::setTimeStamp(Real) will break things.
 - Simulation::clear() currently re-adds the ActionManager immediately. Is this correct?
 - What happens when switching back and forth between record and playback of a Journal?

*/

namespace agxControl
{
  AGX_DECLARE_POINTER_TYPES(ActionManager);




  /**
  The ActionManager keeps a list of Actions and triggers them when the siSimmulation
  that the ActionManager is part of reaches the event's time.
  */
  class AGXPHYSICS_EXPORT ActionManager : public agxSDK::EventListener
  {
  public:
    ActionManager();

    /**
    Make this ActionManager responsible for triggering the given action at
    the appropriate time. If the time of the action has passed already, then
    the action will not be triggered.

    \return True if the action was added. False if the action was already handled by this ActionManager.
    */
    bool add(agxControl::Action* action);

    /**
    Remove the given action from this ActionManager. The action will not be
    triggered by this ActionManager when its time comes.

    \return True if the action was removed. False if it was not handled by this ActionManager.
    */
    bool remove(const agxControl::Action* action);

    /**
    Make this ActionManager responsible for activating, updating and deactivating
    the given operation. If the operation should have been started but not ended
    already, then the operation will be activated immediately. If the end time
    has passed already, then the operation will never be activated.

    \return True if the operation was added. False if the operation was already handled by this ActionManager.
    */
    bool add(agxControl::Operation* operation);

    /**
    Remove the given operation from this ActionManager. It will be deactivated
    if it was currently active.

    \return True if the operation was removed. False if the operation was not handled by this ActionManager.
    */
    bool remove(agxControl::Operation* operation);


    void cleanup();

    void cleanup( agx::Real newCurrentTime );

    /**
    Perform a step. This includes triggering actions whose time has now passed,
    activating operations that have now started, and updating active events.
    \param time The time to compare against the actions activation time.
    */
    void stepActions(agx::TimeStamp time);

    /**
    Used by store/restore. Move actions from the given ActionManager into this
    ActionManager. Operations that span over the last seen time stamp are activated
    immediately.
    */
    void takeActions(ActionManager* other);

    /*
    The ActionManager doesn't quite follow the normal store/restore behavior
    because the simulation that the ActionManager is restored into may be at
    another time stamp than the time stored in the serialized ActionManager.
    Because of this is is not correct to read in and simply add all Actions and
    operations found since that could activate the wrong events. Instead, the
    restored ActionManager is used as a temporary Action/Operation container and
    the restored Actions and Operations are later
    */
    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::ActionManager);


  public:
    AGX_DECLARE_POINTER_TYPES(Tag);
    AGX_DECLARE_POINTER_TYPES(ActionTag);
    AGX_DECLARE_POINTER_TYPES(OperationTag);
    AGX_DECLARE_POINTER_TYPES(BeginOperationTag);
    AGX_DECLARE_POINTER_TYPES(EndOperationTag);

    /** Enum that identifies a Tag as representing a Action or a Operation. */
    enum Type { ACTION, OPERATION };

    /** Enum that identifies an Operation tag as being the beginning or ending tag. */
    enum Endpoint { OPERATION_BEGIN, OPERATION_END };

    typedef agx::List<TagRef> TagList;
    typedef agx::List<OperationRef> OperationList;

  private:
    /**
    Insert the given Tag into its proper place in the given TagList. Will
    reject the Tag if an equivalent Tag already exists in the list.

    \return An iterator to the inserted element, or list.end() if the Tag was rejected.
    */
    TagList::iterator insertSorted(TagRef& tag, TagList& list);

    /**
    Remove the next found OperationTag with the given endpoint for the given
    operation, starting the search from the given iterator. Throws exception if
    the first OperationTag for the given operation has the wrong endpoint type.
    Will update the next tag iterator if the element it pointed to was removed.
    The given iterator is updated to point to the element next to the removed tag,
    or is left unchanged if nothing was removed.

    \param begin Iterator into the TagList from where the search should start. Updated to point to the remove location.
    \param operation The operation to search for.
    \param endpoint The type of OperationTag to search for.
    \return True if the tag was erased, false otherwise.
    */
    bool findAndEraseOperationsTag(TagList::iterator& begin, const Operation* operation, Endpoint endpoint);

    /**
    Called by takeActions(.) to move tags from the other ActionManager into this.

    Insert the given nextToTake tag into the tags list at the location pointed
    to by the given writeIt iterator. Remove the first element from the given
    TagList and store a copy of the new first element, or nullptr if the given TagList
    becomes empty, into nextToTake.

    \param otherTags TagList owned by another ActionManager.
    \param nextToTake Is written into this' TagList and then updated to point to the next element in otherTags.
    \param writeIt Location in this' TagList where nextToTake should be written.
    */
    TagList::iterator takeAction( TagList& otherTags, TagRef& nextToTake, TagList::iterator writeIt );

    /**
    Utility method for reading in Actions stored in the older (2.4.1.1) serialization format.
    */
    void restoreActionQueue(agxStream::InputArchive& in, const char* sectionName);

    size_t countActions() const;
    size_t countOperations() const;

    /**
    Returns a list of all tags, containing all actions and operations sorted in ascending time.
    Note that it might get modified by further calls to ActionManager.
    */
    const TagList& getTags() const;

  private:
    friend class Tag;
    friend class ActionTag;
    friend class OperationTag;
    friend class BeginOperationTag;
    friend class EndOperationTag;

    // A trigger of a Tag will end up in one of these, which will do the actual
    // triggering and also perform the necessary updates to the internal state
    // of the ActionManager.
    void performTrigger(ActionTag* actionTag, agx::Real time);
    void performTrigger(BeginOperationTag* beginTag, agx::Real time);
    void performTrigger(EndOperationTag* endTag, agx::Real time);

  private:
    agx::Real m_lastSeenTimeStamp;

    /** List containing everything that may be triggered during stepping, sorted by time. */
    TagList m_tags;

    /** The tag that is next in line to be activated. */
    TagList::iterator m_nextTag;

    /** The operations whose begin tag has been triggered, but not its end tag. */
    OperationList m_activeOperations;
  };


  AGXPHYSICS_EXPORT bool operator==(const agxControl::ActionManager::Tag& left, const agxControl::ActionManager::Tag& right);

  /**
  Representation of everything that can be triggered by an ActionManager. Separate
  subclasses are given for Actions and Operations.
  */
  class agxControl::ActionManager::Tag : public agx::Referenced
  {
  public:
    Tag(ActionManager::Type type);
    virtual void trigger(agxControl::ActionManager* manager, agx::Real time) = 0;
    virtual agx::Real getTime() const = 0;
    ActionManager::Type getType() const;
  protected:
    virtual ~Tag() {};
  private:
    ActionManager::Type m_type;
  };

  class agxControl::ActionManager::ActionTag : public agxControl::ActionManager::Tag
  {
  public:
    ActionTag(Action* action);
    virtual void trigger(agxControl::ActionManager* manager, agx::Real time) override;
    virtual agx::Real getTime() const override;
    agxControl::Action* getAction();
    const agxControl::Action* getAction() const;
  protected:
    virtual ~ActionTag() {}
  private:
    agxControl::ActionRef m_action;
  };

  /**
  Representation of one of the end points of an operation. Separate subclasses
  are given for the begin- and end points.
  */
  class ActionManager::OperationTag : public ActionManager::Tag
  {
  public:
    OperationTag(agxControl::Operation* operation, ActionManager::Endpoint endpoint);
    virtual void trigger(agxControl::ActionManager* manager, agx::Real time) = 0;
    virtual agx::Real getTime() const = 0;
    agxControl::Operation* getOperation();
    const agxControl::Operation* getOperation() const;
    ActionManager::Endpoint getEndpoint() const;
  protected:
    virtual ~OperationTag() {};
  private:
    agxControl::OperationRef m_operation;
    ActionManager::Endpoint m_endpoint;
  };

  class ActionManager::BeginOperationTag : public OperationTag
  {
  public:
    BeginOperationTag(agxControl::Operation* operation);
    virtual void trigger(agxControl::ActionManager* manager, agx::Real time) override;
    virtual agx::Real getTime() const override;
  protected:
    virtual ~BeginOperationTag() {}
  };

  class ActionManager::EndOperationTag : public OperationTag
  {
  public:
    EndOperationTag(agxControl::Operation* operation);
    virtual void trigger(agxControl::ActionManager* manager, agx::Real time) override;
    virtual agx::Real getTime() const override;
  protected:
    virtual ~EndOperationTag() {}
  };
}

// Include guard.
#endif

/// \endcond
