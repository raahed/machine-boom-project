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

#ifndef AGXSDK_SIMULATION_CONTROL_H
#define AGXSDK_SIMULATION_CONTROL_H

#include <agx/agx.h>
#include <agx/Singleton.h>
#include <agx/String.h>
#include <agxSDK/Simulation.h>
#include <agx/agxPhysics_export.h>


namespace agxMex {
  class Simulation;
}


namespace agxNet {
  class CoSimulationClient;
}




namespace agxSDK {

  AGX_DECLARE_POINTER_TYPES(SimulationControlArgument);

  /// Base class for input/output arguments.
  class AGXPHYSICS_EXPORT SimulationControlArgument : public agx::Referenced {
  public:
    SimulationControlArgument(
      size_t numInputValues, size_t numOutputValues);

    /// \return number of input values
    size_t getNumInputValues() const;

    /// \return number of output values
    size_t getNumOutputValues() const;

    virtual bool applyInput(const agxSDK::Simulation* simulation, const agx::RealVector& input) const = 0;
    virtual bool obtainOutput(const agxSDK::Simulation* simulation, const agx::RealVector& output) const = 0;

  protected:
    size_t m_numInputValues;
    size_t m_numOutputValues;
  };

  /**
  * A class for adding input and output arguments,
  * to be used when running a co-simulation.
  */
  class AGXPHYSICS_EXPORT SimulationControl : public agx::Singleton
  {

    public:
      friend class agxMex::Simulation;
      friend class agxNet::CoSimulationClient;
    AGX_DECLARE_POINTER_TYPES(Argument);



    typedef agx::Vector<SimulationControlArgumentRef> InputArgumentRefVector;
    typedef agx::Vector<SimulationControlArgumentRef> OutputArgumentRefVector;


    class InputArgumentCollection {
      public:
        InputArgumentCollection();
        void clear();
      public:
        InputArgumentRefVector arguments;
        agx::StringVector valueNames;
        size_t cachedTotalInputSize;
    };

    class OutputArgumentCollection {
      public:
        OutputArgumentCollection();
        void clear();
      public:
        OutputArgumentRefVector arguments;
        agx::StringVector valueNames;
        size_t cachedTotalOutputSize;
    };

    public:
      /// Gets (and possibly creates) and instance of the singleton.
      static SimulationControl* instance();

      /**
      Adds an input argument. Input to AGX Dynamics (output from Simulink)
      \param portName - The name of the Simulink output port
      */
      void addInputArgument( const agx::String& portName, SimulationControlArgument* inputArgument );

      /**
      Adds an output argument. Output from AGX Dynamics (input to Simulink)
      \param portName - The name of the Simulink input port
      */
      void addOutputArgument(const agx::String& portName, SimulationControlArgument* outputArgument);

      /**
      Adds an input argument. Input to AGX Dynamics (output from Simulink)
      Will get data at the initialization of the scene as specified in the AGX block-argument in Simulink
      \param portName - Name of the variable, useful at debugging. DOes not correspond to any port in simulink.
      */
      void addInitInputArgument(const agx::String& portName, SimulationControlArgument* inputArgument);

      /**
      Adds an output argument. Output from AGX Dynamics (input to Simulink)
      \param portName - The name of the Simulink output port (
      Will send data from AGX at the initialization to Simulink to the named input port
      */
      void addInitOutputArgument(const agx::String& portName, SimulationControlArgument* outputArgument);

      /**
      * Steps the simulation one time step, using inputs and returning outputs.
      * If anything goes wrong, reports to LOGGER_WARNING and returns false.
      * \param inputs The inputs. Fixed size.
      * \param outputs The outputs. Fixed size, just write at the right positions.
      * \retval Did everything go ok?
      */
      bool step(const agx::RealVector& inputs, const agx::RealVector& outputs, agxSDK::Simulation* simulation);


      bool applyInitInputValues(agxSDK::Simulation* simulation, const agx::RealVector& initInputs) const;

      bool obtainInitOutputValues(agxSDK::Simulation* simulation, const agx::RealVector& initOutputs) const;



      /// Get total size of input values from all input arguments.
      size_t getTotalNumInputValues() const;

      /// Get total size of output values from all output arguments.
      size_t getTotalNumOutputValues() const;

      /// Get total size of input values from all input arguments only for scene initialization.
      size_t getTotalNumInitInputValues() const;

      /// Get total size of output values from all output arguments only for scene initialization.
      size_t getTotalNumInitOutputValues() const;

      /**
      Returns name associated to input value by the argument it belongs to.
      \param valueNr The number of the input value.
      \retval The name. 0 if valueNr >= number of input values.
      */
      const char* getInputValueName(size_t valueNr);

      /**
      Returns name associated to output value by the argument it belongs to.
      \param valueNr The number of the output value.
      \retval The name. 0 if valueNr >= number of output values.
      */
      const char* getOutputValueName(size_t valueNr);

      /**
      Returns name associated to input value only for scene initialization,
      by the argument it belongs to.
      \param valueNr The number of the input value.
      \retval The name. 0 if valueNr >= number of input values.
      */
      const char* getInitInputValueName(size_t valueNr);

      /**
      Returns name associated to output value only for scene initialization,
      by the argument it belongs to.
      \param valueNr The number of the output value.
      \retval The name. 0 if valueNr >= number of output values.
      */
      const char* getInitOutputValueName(size_t valueNr);

      void clear();

      /// \return the last error message set by any method. Will be cleared every timestep
      agx::String getLastErrorMessage() const;

      SINGLETON_CLASSNAME_METHOD();

    protected:
      SimulationControl();
      void shutdown() override;
      /// Clears input and output vectors.

    private:
      static SimulationControl* s_instance;
      InputArgumentCollection m_inputArguments;
      OutputArgumentCollection m_outputArguments;
      InputArgumentCollection m_initInputArguments;
      OutputArgumentCollection m_initOutputArguments;
      mutable agx::String m_errorMessage;
  };

}

#endif // AGXSDK_SIMULATION_CONTROL_H
