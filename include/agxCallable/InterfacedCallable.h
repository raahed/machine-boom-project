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

#ifndef AGX_CALLABLE_INTERFACED_CALLABLE_H
#define AGX_CALLABLE_INTERFACED_CALLABLE_H

#include <agx/Referenced.h>
#include <agx/String.h>
#include <agxCallable/export.h>
#include <functional>

namespace agx
{
  class RigidBody;
  class Constraint;
  class SurfaceVelocityConveyorBelt;
}

namespace agxCallable
{


   template<typename T1, typename T2, typename T3>
   class AGXCALLABLE_EXPORT InterfacedCallable1 : public agx::Referenced
   {
    public:
      typedef T1 ObjectType;
      typedef T2 Output;
      typedef T3 Input;
    public:
      InterfacedCallable1(const agx::String& callstring, ObjectType* object);
      InterfacedCallable1();

      /**
       * Create an Interfaced Callable from the callstring and rigid body
       * \param callstring
       * \param rb
       * \returns New value if successfull, null if not
       */

      Output makeCall(Input input);
      bool isValid();
    protected:
      std::function<Output(Input)> m_func;
  };

   template<typename T1, typename T2>
   class AGXCALLABLE_EXPORT InterfacedCallable0 : public agx::Referenced
   {
   public:
     typedef T1 ObjectType;
     typedef T2 Output;
   public:
     InterfacedCallable0(const agx::String& callstring, ObjectType* object);
     InterfacedCallable0();

     /**
     * Create an Interfaced Callable from the callstring and rigid body
     * \param callstring
     * \param rb
     * \returns New value if successfull, null if not
     */


     Output makeCall();
     bool isValid();
   protected:
     std::function<Output(void)> m_func;
   };

   template<typename T2, typename T3>
   class AGXCALLABLE_EXPORT ConstraintInterfacedCallable1 : public InterfacedCallable1 < agx::Constraint, T2, T3 >
   {
   public:
     typedef T2 Output;
     typedef T3 Input;
   public:
     ConstraintInterfacedCallable1(const agx::String& callstring, agx::Constraint* constraint);
   };

   template<typename T2>
   class AGXCALLABLE_EXPORT ConstraintInterfacedCallable0 : public InterfacedCallable0 < agx::Constraint, T2 >
   {
   public:
     typedef T2 Output;
   public:
     ConstraintInterfacedCallable0(const agx::String& callstring, agx::Constraint* constraint);
   };

   typedef InterfacedCallable0<agx::RigidBody, double> RigidBodyDoubleGetCallable;
   typedef InterfacedCallable1<agx::RigidBody, void, double> RigidBodyDoubleSetCallable;
   typedef InterfacedCallable0<agx::RigidBody, bool> RigidBodyBoolGetCallable;
   typedef InterfacedCallable1<agx::RigidBody, void, bool> RigidBodyBoolSetCallable;
   typedef ConstraintInterfacedCallable0<double> ConstraintDoubleGetCallable;
   typedef ConstraintInterfacedCallable1<void, double> ConstraintDoubleSetCallable;
   typedef ConstraintInterfacedCallable0<bool> ConstraintBoolGetCallable;
   typedef ConstraintInterfacedCallable1<void, bool> ConstraintBoolSetCallable;
   //typedef InterfacedCallable0<agx::SurfaceVelocityConveyorBelt, double> SurfaceVelocityConveyorBeltDoubleGetCallable;
   //typedef InterfacedCallable1<agx::SurfaceVelocityConveyorBelt, void, double> SurfaceVelocityConveyorBeltDoubleSetCallable;
   //typedef InterfacedCallable0<agx::SurfaceVelocityConveyorBelt, bool> SurfaceVelocityConveyorBeltBoolGetCallable;
   //typedef InterfacedCallable1<agx::SurfaceVelocityConveyorBelt, void, bool> SurfaceVelocityConveyorBeltBoolSetCallable;



}


#endif