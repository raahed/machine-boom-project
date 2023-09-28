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

#ifndef AGXFMI_PYTHONVARIABLE_H
#define AGXFMI_PYTHONVARIABLE_H

#include <agx/config/AGX_USE_FMI.h>
#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_FMI()

#include <agxFMI2/export.h>
#include <agxFMI2/Variable.h>
#include <agxFMI2/export/Variable.h>
#include <agxFMI2/export/Module.h>

#include <agx/debug.h>

#include <csignal>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#if AGX_USE_PYTHON()
#if defined(_DEBUG)
#undef _DEBUG
#include <Python.h>
#include <frameobject.h>
#define _DEBUG 1
#else
#include <Python.h>
#include <frameobject.h>
#endif
#endif
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxFMI2
{

  namespace Export
  {

    class AGXFMI_EXPORT PythonErrorSignal
    {
    public:
      PythonErrorSignal();

      void signalError();

      fmi2Status getFmiStatus() const;

    protected:

      fmi2Status m_status;
    };


    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    class AGXFMI_EXPORT OutputVariableBool : public OutputVariable_Bool, public PythonErrorSignal
    {
    public:
      OutputVariableBool(const agx::Name& name);

      virtual bool getValue();

      void setStartValue(bool startValue);

      bool getStartValue();

#ifndef SWIG
      virtual fmi2Status get(fmi2Boolean& value) override;
#endif

    protected:
      virtual ~OutputVariableBool();

    private:

    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    class AGXFMI_EXPORT OutputVariableInt : public OutputVariable_Int, public PythonErrorSignal
    {
    public:
      OutputVariableInt(const agx::Name& name);

      virtual int getValue();

      void setStartValue(int startValue);

      int getStartValue();

#ifndef SWIG
      virtual fmi2Status get(fmi2Integer& value) override;
#endif

    protected:
      virtual ~OutputVariableInt();

    private:

    };

    class AGXFMI_EXPORT OutputVariableReal : public OutputVariable_Real, public PythonErrorSignal
    {
    public:
      OutputVariableReal(const agx::Name& name);

      virtual double getValue();

      void setStartValue(double startValue);

      double getStartValue();

#ifndef SWIG
      virtual fmi2Status get(fmi2Real& value) override;
#endif

    protected:
      virtual ~OutputVariableReal();

    private:

    };

    class AGXFMI_EXPORT OutputVariableString : public OutputVariable_String, public PythonErrorSignal
    {
    public:

      OutputVariableString(const agx::Name& name);

      virtual std::string getValue();

      void setStartValue(std::string startValue)
      {
        OutputVariableT::setStartValue(startValue.c_str());
      }

      std::string getStartValue();

#ifndef SWIG
      virtual fmi2Status get(fmi2String& value) override;
#endif

    protected:
      virtual ~OutputVariableString();

    private:

    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    class AGXFMI_EXPORT InputVariableBool : public InputVariable_Bool, public PythonErrorSignal
    {
    public:
      InputVariableBool(const agx::Name& name);

      void setStartValue(bool startValue);

      bool getStartValue() const;

      virtual bool setValue(bool value);

      virtual bool getValue();

#ifndef SWIG
      fmi2Status get(fmi2Boolean& value) override;
      fmi2Status set(fmi2Boolean value) override;
#endif

    protected:
      virtual ~InputVariableBool();

      fmi2Boolean& getCurrentValue();

    private:

    };
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    class AGXFMI_EXPORT InputVariableInt : public InputVariable_Int, public PythonErrorSignal
    {
    public:
      InputVariableInt(const agx::Name& name);

      void setStartValue(int startValue);

      int getStartValue() const;

      virtual bool setValue(int value);

      virtual int getValue();

#ifndef SWIG
      fmi2Status get(fmi2Integer& value) override;
      fmi2Status set(fmi2Integer value) override;
#endif

    protected:
      virtual ~InputVariableInt();

      fmi2Integer& getCurrentValue();

    private:

    };

    class AGXFMI_EXPORT InputVariableReal : public InputVariable_Real, public PythonErrorSignal
    {
    public:
      InputVariableReal(const agx::Name& name);

      void setStartValue(double startValue);

      double getStartValue() const;

      virtual bool setValue(double value);

      virtual double getValue();

#ifndef SWIG
      fmi2Status get(fmi2Real& value) override;
      fmi2Status set(fmi2Real value) override;
#endif

    protected:
      virtual ~InputVariableReal();

      fmi2Real& getCurrentValue();

    private:

    };

    class AGXFMI_EXPORT InputVariableString : public InputVariable_String, public PythonErrorSignal
    {
    public:
      InputVariableString(const agx::Name& name);

      void setStartValue(std::string startValue);

      std::string getStartValue() const;

      virtual bool setValue(const std::string& value);

      virtual std::string getValue();

#ifndef SWIG
      fmi2Status get(fmi2String& value) override;
      fmi2Status set(fmi2String value) override;
#endif

    protected:
      virtual ~InputVariableString();

      std::string m_value;

    private:

    };

  }
}


#endif // AGX_USE_FMI()

#endif // AGXFMI_PYTHONVARIABLE_H
