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

#ifndef AGXFMI2_EXPORT_VARIABLE_H
#define AGXFMI2_EXPORT_VARIABLE_H

#include <agx/config/AGX_USE_FMI.h>

#if AGX_USE_FMI()

#include <agxFMI2/Variable.h>
#include <agx/Callback.h>
#include <agxData/Value.h>
#include <agx/Xml.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
extern "C"
{
  #include <external/fmi/2.0/fmi2Functions.h>
}
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxSDK
{
  class Simulation;
}


namespace agxFMI2
{
  namespace Export
  {
    class Module;

    AGX_DECLARE_POINTER_TYPES(Variable);
    AGX_DECLARE_VECTOR_TYPES(Variable);
    AGX_DECLARE_POINTER_TYPES(InputVariable);
    AGX_DECLARE_POINTER_TYPES(OutputVariable);


    ////////////////////////////////////////////////////////////////////

    /**
    FMI export variable wrapper. The view of the variable from inside the FMU.
    */
    class AGXFMI_EXPORT Variable : public agxFMI2::Variable
    {
    public:

#ifndef SWIG
      template <typename T>
      struct TypMap;
#endif

      typedef fmi2Status StatusT;

    public:
      Variable(const agx::Name& name = agx::Name());

      /// Set the start value
      void setStartValue(agxData::Value *startValue);

      /// \return The start value
      const agxData::Value *getStartValue() const;

      /// \return The module which the variable is part of
      Module *getModule();

      /// \return The simulation instance
      agxSDK::Simulation *getSimulation();

      /// Generate the modelDescription.xml entry
      agx::TiXmlElement *exportDescription();

    protected:
      virtual ~Variable();

    private:
      friend class Module;
      void setModule(Module *module);

    private:
      Module *m_module;
      agxData::ValueRef m_startValue;
    };


    ////////////////////////////////////////////////////////////////////

    /**
    FMI export output variable wrapper.
    */
    template <typename T>
    class AGXFMI_EXPORT OutputVariableT : public Variable
    {
    public:

#ifndef SWIG
      using CallbackType = agx::Callback1<T&>;
      using CallbackFn = void (*)(T&);
      using Type = T;
#endif

    public:
      OutputVariableT(const agx::Name& name = agx::Name());

#ifndef SWIG
      OutputVariableT(CallbackType callback);
#endif

      /// \return The current value
      virtual fmi2Status get(T& value);

      /// Set the variable start value
      void setStartValue(T startValue);

      /// \return The variable start value
      T getStartValue() const;

#ifndef SWIG
      template <typename T2>
      void setCallback(T2 callback)
      {
        m_callback = CallbackType(callback);
      }
#endif

    protected:
      virtual ~OutputVariableT();

    private:
      CallbackType m_callback;
    };

    typedef OutputVariableT<fmi2Real> OutputVariable_Real;
    typedef OutputVariableT<fmi2Integer> OutputVariable_Int;
    typedef OutputVariableT<fmi2Boolean> OutputVariable_Bool;
    typedef OutputVariableT<fmi2String> OutputVariable_String;

    ////////////////////////////////////////////////////////////////////

    /**
    FMI export input variable wrapper.
    */
    template <typename T>
    class AGXFMI_EXPORT InputVariableT : public OutputVariableT<T>
    {
    public:

#ifndef SWIG
      typedef agx::Callback1<T> CallbackType;
      typedef void (*CallbackFn)(T);
#endif

    public:
      InputVariableT(const agx::Name& name = agx::Name());

#ifndef SWIG
      InputVariableT(CallbackType callback);
#endif

      /// Set the variable value
      virtual fmi2Status set(T value);

      /// \return The current value
      virtual fmi2Status get(T& value);

#ifndef SWIG
      /// Convenience method for getting the current value
      //T get() const;

      template <typename T2>
      void setCallback(T2 callback)
      {
        m_callback = CallbackType(callback);
      }
#endif

    protected:
      virtual ~InputVariableT();

    protected:
      CallbackType m_callback;
      T m_currentValue;
    };

    typedef InputVariableT<fmi2Real> InputVariable_Real;
    typedef InputVariableT<fmi2Integer> InputVariable_Int;
    typedef InputVariableT<fmi2Boolean> InputVariable_Bool;
    typedef InputVariableT<fmi2String> InputVariable_String;


    ////////////////////////////////////////////////////////////////////

    /* Implemenation */

    // template <> struct Variable::TypMap<TYPE_REAL> { typedef agx::Real Type; };
    // template <> struct Variable::TypMap<TYPE_INT> { typedef agx::Int32 Type; };
    // template <> struct Variable::TypMap<TYPE_BOOL> { typedef agx::Bool Type; };
    // template <> struct Variable::TypMap<TYPE_STRING> { typedef agx::String Type; };

#ifndef SWIG
    template <> struct Variable::TypMap<fmi2Real> { static const Type type = TYPE_REAL; };
    template <> struct Variable::TypMap<fmi2Integer> { static const Type type = TYPE_INT; };
    // template <> struct Variable::TypMap<fmi2Boolean> { static const Type type = TYPE_BOOL; };
    template <> struct Variable::TypMap<fmi2String> { static const Type type = TYPE_STRING; };
#endif


  }


}

#endif /* AGX_USE_FMI */

#endif /* AGXFMI2_EXPORT_VARIABLE_H */
