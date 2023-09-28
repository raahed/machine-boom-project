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

#ifndef AGXCALLABLE_DATASERIES_H
#define AGXCALLABLE_DATASERIES_H

#include <agxCallable/export.h>
#include <agxPlot/DataSeries.h>
#include <agx/ContactForceReader.h>
#include <agx/Referenced.h>


namespace agx
{
  class RigidBody;
  class Constraint;
  class GranularContactForceReader;
  class ObserverFrame;
}

namespace agxModel
{
  class SurfaceVelocityConveyorBelt;
}

namespace agxCollide
{
  class Geometry;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxPlot
{
  class CallableWithObserverFrameDataGenerator;
}

namespace agxPlot
{
  class AGXCALLABLE_EXPORT CallableDataSeries : public DataSeries
  {
    public:

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a rigid body.
      */
      CallableDataSeries(agx::RigidBody* rigidBody, const agx::String& functionCall, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a constraint.
      */
      CallableDataSeries(agx::Constraint* constraint, const agx::String& functionCall, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a conveyor belt.
      */
      CallableDataSeries(agxModel::SurfaceVelocityConveyorBelt* conveyorBelt, const agx::String& functionCall, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on an observer frame.
      */
      CallableDataSeries(agx::ObserverFrame* observerFrame, const agx::String& functionCall, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given two rigid bodies.
      */
      CallableDataSeries(agx::ContactForceReader* cfr, const agx::String& functionCall, const agx::RigidBody* body1, const agx::RigidBody* body2, agx::ContactForceReader::ContactType contactType, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given a rigid body and a geometry.
      */
      CallableDataSeries(agx::ContactForceReader* cfr, const agx::String& functionCall, const agx::RigidBody* body, const agxCollide::Geometry* geo, agx::ContactForceReader::ContactType contactType, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given two geometries.
      */
      CallableDataSeries(agx::ContactForceReader* cfr, const agx::String& functionCall, const agxCollide::Geometry* geo1, const agxCollide::Geometry* geo2, agx::ContactForceReader::ContactType contactType, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given two rigid bodies.
      */
      CallableDataSeries(agx::GranularContactForceReader* cfr, const agx::String& functionCall, const agx::RigidBody* body, const agx::String& name);

      /**
      Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given two geometries.
      */
      CallableDataSeries(agx::GranularContactForceReader* cfr, const agx::String& functionCall, const agxCollide::Geometry* geo, const agx::String& name);

    protected:
      virtual ~CallableDataSeries();

  };

  AGX_DECLARE_POINTER_TYPES(CallableDataSeries);
  AGX_DECLARE_VECTOR_TYPES(CallableDataSeries);

  class AGXCALLABLE_EXPORT CallableWithObserverFrameDataSeries : public DataSeries
  {
  public:

    /**
    Create a DataSeries that uses the Callable interface to create a callback on a rigid body.
    */
    CallableWithObserverFrameDataSeries(agx::RigidBody* rigidBody, const agx::String& functionCall, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on a constraint.
    */
    CallableWithObserverFrameDataSeries(agx::Constraint* constraint, const agx::String& functionCall, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on an observerFrame.
    */
    CallableWithObserverFrameDataSeries(agx::ObserverFrame* constraint, const agx::String& functionCall, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on the ContactForceReader.
    */
    CallableWithObserverFrameDataSeries(agx::ContactForceReader* cfr, const agx::String& functionCall, const agx::RigidBody* body1, const agx::RigidBody* body2const, agx::ContactForceReader::ContactType contactType, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on the ContactForceReader.
    */
    CallableWithObserverFrameDataSeries(agx::ContactForceReader* cfr, const agx::String& functionCall, const agx::RigidBody* body, const agxCollide::Geometry* geom, agx::ContactForceReader::ContactType contactType, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on the ContactForceReader.
    */
    CallableWithObserverFrameDataSeries(agx::ContactForceReader* cfr, const agx::String& functionCall, const agxCollide::Geometry* geom1, const agxCollide::Geometry* geom2const, agx::ContactForceReader::ContactType contactType, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given two rigid bodies.
    */
    CallableWithObserverFrameDataSeries(agx::GranularContactForceReader* cfr, const agx::String& functionCall, const agx::RigidBody* body, const agx::String& name);

    /**
    Create a DataSeries that uses the Callable interface to create a callback on a contact force reader given two geometries.
    */
    CallableWithObserverFrameDataSeries(agx::GranularContactForceReader* cfr, const agx::String& functionCall, const agxCollide::Geometry* geo, const agx::String& name);

    void addTransformFunction(const agx::String& transformFunction, agx::ObserverFrame* frame);
    void addTransformFunctionInner(const agx::String& transformFunction, agx::ObserverFrame* frame);
    void addTransformFunctionOuter(const agx::String& transformFunction, agx::ObserverFrame* frame);

    void setRelativeFunction(const agx::String& transformFunction, agx::ObserverFrame* frame, agx::RigidBody* rb);
    void setRelativeFunction(const agx::String& transformFunction, agx::ObserverFrame* frame, agx::ObserverFrame* obs);
  protected:
    virtual ~CallableWithObserverFrameDataSeries();
  private:
    CallableWithObserverFrameDataGenerator* m_generator;
  };


  AGX_DECLARE_POINTER_TYPES(CallableWithObserverFrameDataSeries);
  AGX_DECLARE_VECTOR_TYPES(CallableWithObserverFrameDataSeries);
}

#endif
