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

#ifndef AGXFMI2_EXPORT_MODULE_H
#define AGXFMI2_EXPORT_MODULE_H

#include <agx/config/AGX_USE_FMI.h>

#if AGX_USE_FMI()

#include <agxFMI2/export.h>

#include <agxFMI2/export/Variable.h>
#include <agxFMI2/export/RigidBodyAttributeAccessors.h>
#include <agx/Object.h>
#include <agx/Uuid.h>
#include <agx/Xml.h>
#include <agxSDK/Simulation.h>

#include <agx/config/AGX_USE_OSG.h>
#include <agx/RealTimeTrigger.h>
#if AGX_USE_OSG()
#include <agxOSG/ExampleApplication.h>
#include <agxOSG/Node.h>
#endif


extern "C"
{
  #include <external/fmi/2.0/fmi2Functions.h>
}
#if defined(_MSC_VER)
#pragma warning( pop )
#endif

namespace agxOSG
{
  class ExampleApplication;
  class Group;
}


namespace agxIO
{
  class ArgumentParser;
}




namespace agxFMI2
{
  namespace Export
  {

    // AGX_DECLARE_POINTER_TYPES(Module);

    /**
    FMI export module. The view of the module from inside the FMU.
    */
    class AGXFMI_EXPORT Module
    {
    public:
      Module(const agx::Name& name, const char* GUID, bool createOsgWindow = false);
      ~Module();

      /// Register a variable to the module interface
      agx::UInt32 registerVariable(Variable* variable);

      /// Register a rigid body as a strong coupling body
      void registerStrongCoupling(agx::RigidBody* body);

      void initRendering(agxIO::ArgumentParser* argParser = nullptr);
      void initRemoteConnection();
      void cleanup();

      const agx::Name& getName() const;

      Variable* getVariable(const agx::Name& name);

      const char *getGUID() const;

      const agx::String& getDescription() const;
      void setDescription(const agx::String& description);

      const agx::String& getResourcePath() const;
      void setResourcePath(const agx::String& path);

      // setSimulation can only be called _before_ the module is initialized!
      void setSimulation(agxSDK::Simulation* simulation);
      agxSDK::Simulation *getSimulation();

      void setOsgApplication(agxOSG::ExampleApplication* app);
      agxOSG::ExampleApplication *getOsgApplication();

      void setOsgRoot(osg::Group *group);
      osg::Group *getOsgRoot();

      void exportModelDescriptionFile(const agx::String& filePath);

      bool getCreateOsgWindow() const;

    protected:
      void setCreateOsgWindow(bool flag);
      void setThrottleRealTime(bool flag);
      void setUseDebugRendering(bool flag);
      void updateViewer(agx::Real dt);
      void setAutoStepCallback(bool flag);
      void setRenderFrequency(agx::Real freq);
      void setEnableJournalRecord(bool flag);
      void setEnableTaskProfiling(bool flag);
      void setEnableThreadTimeline(bool flag);
      void setJournalPath(const agx::String& path);
      void setJournalConfiguration(const agx::String& config);
      void setJournalRecordFrequency(agx::Real frequency);
      void setInternalStepMultiplier(agx::Int multiplier);
      void setApplicationArguments(const agx::String& args);
      void createRenderObjects();

      /*
      Enable automatic OSG rendering of all geometries.
      */
      void setEnableAutoRendering(bool enable);

    public:
      /////////// PRIVATE /////////////////
#ifndef SWIG
      void finalizeSetup();
      fmi2Status fmiSetupExperiment(fmi2Boolean toleranceDefined, fmi2Real relativeTolerance, fmi2Real tStart, fmi2Boolean stopTimeDefined, fmi2Real tStop);
      fmi2Status fmiEnterInitializationMode();
      fmi2Status fmiExitInitializationMode();
      fmi2Status fmiTerminate();
      fmi2Status fmiReset();
      fmi2Status fmiSetDebugLogging(fmi2Boolean loggingOn, size_t nCategories, const fmi2String categories[]);
      fmi2Status fmiGetReal(const fmi2ValueReference vr[], size_t nvr, fmi2Real value[]);
      fmi2Status fmiGetInteger(const fmi2ValueReference vr[], size_t nvr, fmi2Integer value[]);
      fmi2Status fmiGetBoolean(const fmi2ValueReference vr[], size_t nvr, fmi2Boolean value[]);
      fmi2Status fmiGetString(const fmi2ValueReference vr[], size_t nvr, fmi2String  value[]);
      fmi2Status fmiSetReal(const fmi2ValueReference vr[], size_t nvr, const fmi2Real value[]);
      fmi2Status fmiSetInteger(const fmi2ValueReference vr[], size_t nvr, const fmi2Integer value[]);
      fmi2Status fmiSetBoolean(const fmi2ValueReference vr[], size_t nvr, const fmi2Boolean value[]);
      fmi2Status fmiSetString(const fmi2ValueReference vr[], size_t nvr, const fmi2String  value[]);
      fmi2Status fmiSetRealInputDerivatives(const fmi2ValueReference vr[], size_t nvr, const fmi2Integer order[], const fmi2Real value[]);
      fmi2Status fmiGetRealOutputDerivatives(const fmi2ValueReference vr[], size_t nvr, const fmi2Integer order[], fmi2Real value[]);
      fmi2Status fmiCancelStep();
      fmi2Status fmiDoStep(fmi2Real currentCommunicationPoint, fmi2Real communicationStepSize, fmi2Boolean commitStep);
      fmi2Status fmiGetStatus(const fmi2StatusKind s, fmi2Status*  value);
      fmi2Status fmiGetRealStatus(const fmi2StatusKind s, fmi2Real*    value);
      fmi2Status fmiGetIntegerStatus(const fmi2StatusKind s, fmi2Integer* value);
      fmi2Status fmiGetBooleanStatus(const fmi2StatusKind s, fmi2Boolean* value);
      fmi2Status fmiGetStringStatus(const fmi2StatusKind s, fmi2String*  value);
      fmi2Status fmiGetDirectionalDerivative(const fmi2ValueReference vUnknown_ref[], size_t nUnknown, const fmi2ValueReference vKnown_ref[] , size_t nKnown, const fmi2Real dvKnown[], fmi2Real dvUnknown[]);

      fmi2Status fmiGetFMUstate(fmi2FMUstate* state);
      fmi2Status fmiSetFMUstate(fmi2FMUstate state);
      fmi2Status fmiFreeFMUstate(fmi2FMUstate* state);
#endif
    private:
      template<typename T>
      T *createParameter(const agx::String& name, typename T::Type startValue = typename T::Type());

      template <typename T>
      void registerBodyAccessor(agx::RigidBody *body, agx::UInt numComponents, bool createMobilityVariable = false);

      agx::TiXmlElement *exportModel();
      agx::TiXmlElement *exportCoSimulation();
      agx::TiXmlElement *exportUnitDefinitions();
      agx::TiXmlElement *exportTypeDefinitions();
      agx::TiXmlElement *exportModelVariables();
      agx::TiXmlElement *exportModelStructure();

      void createExportGeneralVariables();
      void applyAccumulators();

      template <typename T, typename VariableT>
      fmi2Status fmiSetT(const fmi2ValueReference vr[], size_t nvr, const T value[]);

      template <typename T, typename VariableT>
      fmi2Status fmiGetT(const fmi2ValueReference vr[], size_t nvr, T value[]);

    private:
      AGX_DECLARE_POINTER_TYPES(SteppingThread);

    private:
      friend class SteppingThread;

      void signalStepperDone(bool completedStep);

      agx::UInt getInternalStepMultiplier() const;

    private:
      class RigidBodyMobilityOutputVariable;
      using RigidBodyMobilityTable = agx::HashTable<Variable *, agx::Real>;
      RigidBodyMobilityTable m_mobilityTable;

    private:
      agx::Name   m_name;
      const char *m_GUID;
      bool        m_initialized;
      bool        m_createOsgWindow;
      bool        m_throttleRealTime;
      bool        m_useDebugRendering;
      bool        m_enableJournalRecord;
      bool        m_enableTaskProfiling;
      bool        m_enableThreadTimeline;
      bool        m_initializedApplication;
      agx::String m_journalPath;
      agx::String m_journalConfiguration;
      agx::Real   m_journalRecordFrequency;
      agx::UInt   m_internalStepMultiplier;
      bool        m_sharedApplication;
      agx::String m_description;
      agx::String m_resourcePath;
      agx::String m_applicationArguments;
      bool        m_enableAutoRendering;
#if AGX_USE_OSG()
      agxOSG::ExampleApplicationRef m_application;
      osg::ref_ptr<osg::Group> m_renderRoot;
#endif
      agxSDK::SimulationRef m_simulation;
      agx::Real m_rtThrottleAccumulation;
      agx::Timer m_rtThrottleTimer;
      agx::RealTimeTrigger m_rtTimer;
      agx::Block m_stepBlock;
      agx::Callback1<bool> m_autoStepCallback;


      // fmi2ValueReference m_vrCounter;
      bool m_useLogging;
      agx::HashSet<agx::String> m_loggingCategories;
      agx::Vector<RigidBodyTorqueAccumulatorInputVariableRef> m_torqueAccumulators;
      agx::Vector<RigidBodyForceAccumulatorInputVariableRef> m_forceAccumulators;

      VariableRefVector m_variables;

      agx::Real m_relativeTolerance;
      agx::Real m_tStart;
      agx::Real m_tStop;
      bool m_hasStopDefined;
      bool m_firstStep;
      SteppingThreadRef m_steppingThread;
      agx::Block m_steppingBlock;
      bool m_strongCouplingStepping;
      bool m_stepComplete;
      bool m_preSolve;
      bool m_enableDirectSolverDataSharing;
      agx::MobilitySolverRef m_mobilityReferenceSolver;
      agx::MobilitySolverRef m_savedState;
    };

#ifndef SWIG
    class Module::SteppingThread : public agx::BasicThread, public agx::Object
    {
    public:
      SteppingThread(Module *module);

      void start();
      void stop();

      void beginStep();
      void continueStep();

      void setupStrongCoupling();

    protected:
      virtual ~SteppingThread();

    private:
      virtual void run() override;
      void preSolveCallback(agx::Task *);
      // void postSolveCallback(agx::Task *);

    private:
      Module *m_module;
      bool m_running;
      bool m_stepping;
      bool m_initialized;
      agx::Block m_startBlock;
      agx::Block m_activationBlock;
      agx::Block m_solverBlock;
      agx::Task::ExecutionEvent::CallbackType m_preSolveCallback;
      agx::TaskRef m_reintegrateVelocitiesTask;
      // agx::Task::ExecutionEvent::CallbackType m_postSolveCallback;
    };
#endif

    AGX_FORCE_INLINE bool Module::getCreateOsgWindow() const { return m_createOsgWindow; }

  }

}

#endif /* AGX_USE_FMI */

#endif /* AGXFMI2_EXPORT_MODULE_H */
