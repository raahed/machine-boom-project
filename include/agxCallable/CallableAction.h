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

#ifndef AGXCALLABLE_ACTION_H
#define AGXCALLABLE_ACTION_H

#include <agxCallable/export.h>
#include <agxControl/Action.h>
#include <agx/Referenced.h>
#include <agx/Logger.h>


namespace agx
{
 class RigidBody;
}

namespace agxModel
{
  class SurfaceVelocityConveyorBelt;
}

namespace agxCallable
{
  /**
  Class for creating an action for a callable.
  Only the template specializations which are typedef:ed below can be chosen.
  */
  template<typename T1, typename T2, typename T3>
  class AGXCALLABLE_EXPORT CallableAction : public agxControl::Action
  {
    public:

      /**
      Constructor, taking a pointer to the callable object, the function name, the value to set and the time.
      Choose only amongst the template specializations given below.
      The functions to be used can be seen in Callable.h.
      \param callableObject The callable object.
      \param functionName The function name on the callable object.
      */
      CallableAction(T1* callableObject, const agx::String& functionName, const T2 value, const agx::Real time );

      // Is the callable valid? Mostly based on if the functionName exists for the callableObject.
      bool isValid() const;

      // Serialization code. Cannot use AGXSTREAM_DECLARE_SERIALIZABLE macro because
      // of templating. For example, the getConstructClassId() and getClassName()
      // methods have to translate the <T> part of the name into a substring containing
      // the actual type of T.
      agxStream::StorageAgent* getStorageAgent() const;
      static const char* getConstructClassId();
      friend class agxStream::DefStorageAgent<CallableAction>;
      const char* getClassName() const;
      static agxStream::Serializable *create();
      static agxStream::Serializable *create(agxStream::InputArchive& in);
      void store(agxStream::OutputArchive& out) const;
      void restore(agxStream::InputArchive& in);

    protected:
      virtual ~CallableAction();
      CallableAction();
      static const char* s_classname;
  };



  // These template specializations can be used.
  // They are of form: Type, argument type, return type (will be ignored).
  typedef CallableAction<agxModel::SurfaceVelocityConveyorBelt, agx::Real, void> SurfaceVelocityConveyorBeltSetRealAction;
  typedef CallableAction<agx::RigidBody, const agx::Vec3&, void> RigidBodySetVec3Action;
}

#endif
