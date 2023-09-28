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


#ifndef AGXHYDRAULICS_PISTON_ACTUATOR_H
#define AGXHYDRAULICS_PISTON_ACTUATOR_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/Pipe.h>
#include <agxHydraulics/PistonActuatorConnector.h>
#include <agxPowerLine/Actuator1DOF.h>
#include <agxPowerLine/ActuatorConnector.h>
#include <agxPowerLine/ActuatorUnit.h>

namespace agx
{
  class RigidBody;
}

namespace agxModel
{
  class Connector;
}

namespace agxHydraulics
{

  class PistonActuator;
  class FlowUnit;




  AGX_DECLARE_POINTER_TYPES(PistonActuator);
  AGX_DECLARE_VECTOR_TYPES(PistonActuator);


  /**
  A PistonActuator is a model of a hydraulic cylinder. It is created from a
  translational agx::Constraint1DOF (prismatic or distance joint) and acts in
  the same way as the motor those constraints contain. The PistonActuator
  contains two chambers, the input and the output, and two attachment points,
  also called the input and the output. Flow into the input causes the piston
  to extend and flow into the output causes it to compress.

  The two chambers can have different areas, causing the same working pressure
  to result in different mechanical forces on the bodies connected by the
  original constraint depending on if the high pressure is at the input or the
  output side. An effect of this is that the cylinder may move when the
  pressure is the same, e.g., tank pressure, on both the input and the output.
  */
  class AGXHYDRAULICS_EXPORT PistonActuator : public agxPowerLine::TranslationalActuator
  {
    public:

      /**
      Create a PistonActuator from the given prismatic. The 'barrelArea'
      defines the force/pressure ratio for the input chamber and
      'pistonArea' defines the force/pressure ratio for the output chamber.

      The geometry of the cylinder is defined by the 'barrelArea' and
      'pistonArea' parameters, and the range of the prismatic. It is important
      that the prismatic has a valid enabled range for the PistonActuator to
      function properly.

      \param prismatic The constraint that defines the cylinder.
      \param inputChamberArea The area of fluid contact in the input chamber.
      \param outputChamberArea The area of fluid contact in the output chamber.
      \param fluidDensity The density of the fluid in the chambers.
       */
      PistonActuator(agx::Prismatic* prismatic, agx::Real inputChamberArea, agx::Real outputChamberArea, agx::Real fluidDensity);

      /**
      Same as the constructor taking a agx::Prismatic, but with a
      agx::DistanceJoint instead.

      \see PistonActuator
       */
      PistonActuator(agx::DistanceJoint* distancejoint, agx::Real inputChamberArea, agx::Real outputChamberArea, agx::Real fluidDensity);


      /**
      \return The input chamber.
      */
      agxHydraulics::FlowUnit* getInputChamber();
      const agxHydraulics::FlowUnit* getInputChamber() const;

      /**
      \return The output chamber.
      */
      agxHydraulics::FlowUnit* getOutputChamber();
      const agxHydraulics::FlowUnit* getOutputChamber() const;

      /**
      Get one of the cylinder chambers.
      \param side The chamber to return.
      \return The chamber at the given side, or nullptr if the side is invalid or NO_SIDE.
      */
      agxHydraulics::FlowUnit* getChamber(agxPowerLine::Side side);


      /**
      \return The Connector that connects the input chamber flow unit to the mechanical bodies.
      */
      agxHydraulics::PistonActuatorConnector* getInputChamberConnector();
      const agxHydraulics::PistonActuatorConnector* getInputChamberConnector() const;

      /**
      \return The Connector that connects the output chamber flow unit to the mechanical bodies.
      */
      agxHydraulics::PistonActuatorConnector* getOutputChamberConnector();
      const agxHydraulics::PistonActuatorConnector* getOutputChamberConnector() const;

      /**
      \param side The connector to return.
      \return The connector that connectes the chamber at the given side to the
              mechanical bodies, or nullptr if the side is invalid or
              NO_SIDE.
      */
      agxHydraulics::PistonActuatorConnector* getChamberConnector(agxPowerLine::Side side);


      /**
      \return The length of the input chamber.
      */
      agx::Real getPistonPosition() const;





    // Deprecated methods.
    public:
      /**
      \deprecated Use getInputChamber instead.
      \return The front, i.e. input, chamber.
      */
      agxHydraulics::FlowUnit* getFrontChamber();
      const agxHydraulics::FlowUnit* getFrontChamber() const;

      /**
      \deprecated Use getOutputChamber instead.
      \return The rear, i.e. output, chamber.
      */
      agxHydraulics::FlowUnit* getRearChamber();
      const agxHydraulics::FlowUnit* getRearChamber() const;


      /**
      \deprecated Use getInputChamberConnector instead.
      \return The connector that connects the front, i.e. input, chamber to the
              mechanical bodies.
      */
      agxHydraulics::PistonActuatorConnector* getFrontChamberConnector();

      /**
      \deprecated Use getOutputChamberConnector instead.
      \return The connector that connects the rear, i.e. output, chamber to the
              mechanical bodies.
      */
      agxHydraulics::PistonActuatorConnector* getRearChamberConnector();




    // Methods called by the rest of the power line framework.
    public:
      using agxPowerLine::Actuator1DOF::connect;

      DOXYGEN_START_INTERNAL_BLOCK()
      virtual bool preUpdate(agx::Real timeStep) override;

      /**
      The length of a chamber is defined by the range of the AGX constraint
      from which the PistonActuator is created and the relative position of the
      mechanical bodies.

      \return The length of the chamber at the given side.
      */
      virtual agx::Real calculateChamberLength(agxPowerLine::Side side) const;
      DOXYGEN_END_INTERNAL_BLOCK()


#ifndef SWIG
      virtual void getConnectableDimensionTypes(
        agxPowerLine::PhysicalDimension::TypeVector& types, agxPowerLine::Side side) const override;

      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
        agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side) override;


      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::PistonActuator);
#endif


    protected:
      void create(agx::Constraint1DOF* constraint1DOF, agx::Real inputChamberArea, agx::Real outputChamberArea, agx::Real fluidDensity);
      bool connect(agxPowerLine::ActuatorBodyUnit* flowSideBody, agxPowerLine::ActuatorBodyUnit* oppositeSideBody, agxPowerLine::Side index);

      PistonActuator();
      virtual ~PistonActuator();

    private:
      agx::Real m_chamberBufferLength;
      agx::ref_ptr<FlowUnit> m_flowUnits[2];
      agx::ref_ptr<agxHydraulics::PistonActuatorConnector> m_pistonChamberConnectors[2];
  };






}

#endif
