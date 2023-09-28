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

#ifndef AGXCALLABLE_CALLABLEDADATAGENERATOR_H
#define AGXCALLABLE_CALLABLEDADATAGENERATOR_H

#include <agxPlot/LambdaDataGenerator.h>
#include <agx/Uuid.h>
#include <agx/ObserverFrame.h>
#include <agx/ContactForceReader.h>

namespace agx
{
  class RigidBody;
  class Constraint;
  class GranularContactForceReader;
}

namespace agxModel
{
  class SurfaceVelocityConveyorBelt;
}

namespace agxPlot
{
  class CallableDataGenerator : public LambdaDataGenerator
  {
  public:
    CallableDataGenerator(const agx::String& function, agx::RigidBody* body);

    CallableDataGenerator(const agx::String& function, agx::Constraint* constraint);

    CallableDataGenerator(const agx::String& function, agxModel::SurfaceVelocityConveyorBelt* conveyorBelt);

    CallableDataGenerator(const agx::String& function, agx::ObserverFrame* observerFrame);
  };

  class CallableWithObserverFrameDataGenerator : public LambdaDataGenerator
  {
  public:
    CallableWithObserverFrameDataGenerator(const agx::String& function, agx::RigidBody* body);
    CallableWithObserverFrameDataGenerator(const agx::String& function, agx::Constraint* constraint);
    CallableWithObserverFrameDataGenerator(const agx::String& function, agx::ObserverFrame* observerFrame);

    CallableWithObserverFrameDataGenerator(
      const agx::String& function, agx::ContactForceReader* cfr,
      const agx::RigidBody* body1, const agx::RigidBody* body2, agx::ContactForceReader::ContactType contactType );

    CallableWithObserverFrameDataGenerator(
      const agx::String& function, agx::ContactForceReader* cfr,
      const agx::RigidBody* body, const agxCollide::Geometry* geo, agx::ContactForceReader::ContactType contactType);

    CallableWithObserverFrameDataGenerator(
      const agx::String& function, agx::ContactForceReader* cfr,
      const agxCollide::Geometry* geo1, const agxCollide::Geometry* geo2, agx::ContactForceReader::ContactType contactType);

    CallableWithObserverFrameDataGenerator(
      const agx::String& function, agx::GranularContactForceReader* cfr,
      const agx::RigidBody* body2);

    CallableWithObserverFrameDataGenerator(
      const agx::String& function, agx::GranularContactForceReader* cfr,
      const agxCollide::Geometry* geo);

    void addTransformFunction(     const agx::String& transformFunction, agx::ObserverFrame* frame);
    void addTransformFunctionInner(const agx::String& transformFunction, agx::ObserverFrame* frame);
    void addTransformFunctionOuter(const agx::String& transformFunction, agx::ObserverFrame* frame);

    void setRelativeFunction(const agx::String& transformFunction, agx::ObserverFrame* frame, agx::RigidBody* rb);
    void setRelativeFunction(const agx::String& transformFunction, agx::ObserverFrame* frame, agx::ObserverFrame* obs);
  private:
    typedef std::vector<std::function<agx::Vec3(const agx::Vec3&)>> ObserverTransformFunctionList;

    void init(std::function<agx::Vec3()> vectorFunction, const agx::String& postAccess);

    ObserverTransformFunctionList m_transformFunctions;
    std::function<agx::Vec3(const agx::RigidBody*)> m_relativeBodyFunction;
    agx::RigidBody* m_relativeBody;
    std::function<agx::Vec3(const agx::ObserverFrame*)> m_relativeObserverFunction;
    agx::ObserverFrame* m_relativeObserverFrame;
  };

  class CallableContactForceReaderDataGenerator : public LambdaDataGenerator
  {
  public:
    CallableContactForceReaderDataGenerator(
      const agx::String& function, agx::ContactForceReader* cfr,
      const agx::RigidBody* body1, const agx::RigidBody* body2, agx::ContactForceReader::ContactType contactType);

    CallableContactForceReaderDataGenerator(
      const agx::String& function, agx::ContactForceReader* cfr,
      const agx::RigidBody* body, const agxCollide::Geometry* geo, agx::ContactForceReader::ContactType contactType);

    CallableContactForceReaderDataGenerator(
      const agx::String& function, agx::ContactForceReader* cfr,
      const agxCollide::Geometry* geo1, const agxCollide::Geometry* geo2, agx::ContactForceReader::ContactType contactType);

    CallableContactForceReaderDataGenerator(
      const agx::String& function, agx::GranularContactForceReader* cfr,
      const agx::RigidBody* body2);

    CallableContactForceReaderDataGenerator(
      const agx::String& function, agx::GranularContactForceReader* cfr,
      const agxCollide::Geometry* geo);
  };
}

#endif