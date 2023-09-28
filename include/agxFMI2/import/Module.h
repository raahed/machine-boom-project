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

#ifndef AGXFMI2_IMPORT_MODULE_H
#define AGXFMI2_IMPORT_MODULE_H

#include <agx/config/AGX_USE_FMI.h>

#if AGX_USE_FMI()

#include <agxFMI2/import/Variable.h>

#include <agx/Component.h>
#include <agx/PluginMacros.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
extern "C"
{
  #include <JM/jm_callbacks.h>
  #include <FMI/fmi_import_context.h>
  #include <FMI2/fmi2_functions.h>
  #include <FMI2/fmi2_import.h>
  #include <FMI2/fmi2_types.h>
}
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxFMI2
{
  namespace Export
  {
    class Module;
  }

  namespace Import
  {
    class Master;

    AGX_DECLARE_POINTER_TYPES(RigidBodyConnector);
    AGX_DECLARE_VECTOR_TYPES(RigidBodyConnector);

    AGX_DECLARE_POINTER_TYPES(Module);
    AGX_DECLARE_VECTOR_TYPES(Module);


    /**
    Module
    */
    class AGXFMI_EXPORT Module : public agx::Component
    {
    public:
      enum LogLevel {
        LOG_LEVEL_NOTHING =  jm_log_level_nothing, /**< \brief Must be first in this enum. May be usefull in application relying solely on jm_get_last_error() */
        LOG_LEVEL_FATAL =  jm_log_level_fatal,     /**< \brief Unrecoverable errors */
        LOG_LEVEL_ERROR =  jm_log_level_error,     /**< \brief Errors that may be not critical for some FMUs. */
        LOG_LEVEL_WARNING =  jm_log_level_warning, /**< \brief Non-critical issues */
        LOG_LEVEL_INFO =  jm_log_level_info,       /**< \brief Informative messages */
        LOG_LEVEL_VERBOSE =  jm_log_level_verbose, /**< \brief Verbose messages */
        LOG_LEVEL_DEBUG =  jm_log_level_debug,     /**< \brief Debug messages. Only enabled if library is configured with FMILIB_ENABLE_LOG_LEVEL_DEBUG */
        LOG_LEVEL_ALL = jm_log_level_all           /**< \brief Must be last in this enum. */
      };

      const VariableRefVector& getVariables() const;

      agxFMI2::Import::Variable *getVariable(const agx::Name& name);

      void instantiate(bool visible);
      void init(fmi2_real_t tStart, fmi2_real_t tEnd, fmi2_boolean_t stopTimeDefined = true);
      void terminate();

      bool isInstantiated() const;

      Master* getMaster();

      Export::Module *getAgxExportModule();

      agx::String getModelDescription();


      fmi2_status_t doStep(agx::Real currentTime, agx::Real timeStep, bool newStep);

      fmi2_status_t getDirectionalDerivative(const fmi2_value_reference_t vKnown_ref[] , size_t nKnown, const fmi2_value_reference_t vUnknown_ref[], size_t nUnknown, const fmi2_real_t dvKnown[], fmi2_real_t dvUnknown[]);

      // fmi2_real_t getOutputDerivative(agx::UInt32 vr, fmi2_integer_t order = 1);

      fmi2_import_t *getFmuImpl() const;

    private:
      friend class Master;
      Module(Master *master, const agx::String& zipPath, LogLevel logLevel = LOG_LEVEL_INFO);
      static void jmLogger_dispatch(jm_callbacks* c, jm_string module, jm_log_level_enu_t log_level, jm_string message);
      void jmLogger(jm_string module, jm_log_level_enu_t log_level, jm_string message);


    protected:
      virtual ~Module();

    private:
      Master *m_master;
      bool m_ownUnpackedData;
      agx::String m_diskRootPath;
      agx::String m_fmuLocation;
      jm_callbacks m_jmCallbacks;
      fmi_import_context_t *m_context;
      fmi2_import_t *m_fmu;
      fmi2_callback_functions_t m_fmi2callbackFunctions;
      VariableRefVector m_variables;
      bool m_instantiated;
      agx::PluginHandle m_plugin;
      Export::Module *m_agxExportModule;
      // agx::TiXmlElement *m_modelDescription;
    };

  }
}

#endif /* AGX_USE_FMI */

#endif /* AGXFMI2_IMPORT_MODULE_H */
