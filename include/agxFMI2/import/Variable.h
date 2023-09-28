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

#ifndef AGXFMI2_IMPORT_VARIABLE_H
#define AGXFMI2_IMPORT_VARIABLE_H

#include <agx/config/AGX_USE_FMI.h>

#if AGX_USE_FMI()

#include <agxFMI2/Variable.h>


#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
extern "C"
{
  #include <FMI2/fmi2_import_variable.h>
  #include <FMI2/fmi2_import_capi.h>
}
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#if defined(__GNUC__) && !defined(SWIGPYTHON)
// GCC has a bug[1] that casuses our PushDisableWarnings.h to be ignored under
// certain circumstances. In this case we get a bunch of "unused function"
// warnings. The warnings are suppressed by using the symbol names below.
//
// [1]: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
static void warningSuppression()  __attribute__((unused));
static inline void warningSuppression()
{
  (void)jm_get_last_error;
  (void)jm_clear_last_error;
  (void)jm_log_debug_v;
  (void)jm_log_debug;
  (void)fmi2_get_types_platform;
  (void)fmi1_get_platform;
}
#endif


namespace agxFMI2
{
  namespace Import
  {
    class Module;

    AGX_DECLARE_POINTER_TYPES(Variable);
    AGX_DECLARE_VECTOR_TYPES(Variable);

    /**
    Variable
    */
    class AGXFMI_EXPORT Variable : public agxFMI2::Variable
    {
    public:
      Module *getModule();


      agx::Real getStartValue_Real() const;
      agx::Int32 getStartValue_Int() const;
      agx::Bool getStartValue_Bool() const;
      agx::String getStartValue_String() const;


      void setValue_Real(agx::Real value);
      void setValue_Int(agx::Int32 value);
      void setValue_Bool(agx::Bool value);
      void setValue_String(const agx::String& value);

      agx::Real getValue_Real() const;
      agx::Int32 getValue_Int() const;
      agx::Bool getValue_Bool() const;
      agx::String getValue_String() const;

      agx::String toString() const;

    protected:
      virtual ~Variable();

    private:
      friend class Module;
      Variable(Module *module, fmi2_import_variable_t *variable);

    private:
      Module *m_module;
      fmi2_import_variable_t *m_variable;
    };
  }

}

#endif /* AGX_USE_FMI */

#endif /* AGXFMI2_IMPORT_VARIABLE_H */
