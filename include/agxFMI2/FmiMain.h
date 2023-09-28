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

/** This header is required by all FMI plugins and exporters of AGX Dynamics. */


#ifndef AGX_FMI_VER
#error AGX_FMI_VER is undefined
#endif

#define nsFMI agxFMI2

#define fmiOK fmi2OK
#define fmiError fmi2Error
#define fmiReal fmi2Real
#define fmiInteger fmi2Integer
#define fmiBoolean fmi2Boolean
#define fmiString fmi2String

typedef fmi2Status fmiStatus;

namespace agx
{
  // These must be implemented in the FMI plugin
  fmiStatus fmiInit(nsFMI::Export::Module *module);
  fmiStatus fmiShutdown(nsFMI::Export::Module *module);
  fmiStatus fmiInitApplication(nsFMI::Export::Module *module);
}
