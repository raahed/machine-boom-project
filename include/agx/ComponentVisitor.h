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

#ifndef AGX_COMPONENTVISITOR_H
#define AGX_COMPONENTVISITOR_H

#include <agx/Callback.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{

  class Object;
  class Component;

  /**
  Base class for ComponentVisitors.
  */
  class AGXCORE_EXPORT ComponentVisitor
  {
    public:
      virtual void visit( agx::Object* ) {}
      virtual void visit( agx::Component* comp );
      virtual ~ComponentVisitor() {}
  };

  /**
  ComponentVisitor which applies a callback function on each visited node.
  */
  class AGXCORE_EXPORT CallbackComponentVisitor : public ComponentVisitor
  {
    public:
      typedef Callback1<Object*> TraverseCallback;

      CallbackComponentVisitor( TraverseCallback callback );

      virtual void visit( agx::Object* obj );
      virtual void visit( agx::Component* comp );

      virtual ~CallbackComponentVisitor() {}

    private:
      TraverseCallback m_callback;
  };
}


#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif

