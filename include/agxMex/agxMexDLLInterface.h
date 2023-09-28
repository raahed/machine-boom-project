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

#ifndef AGXMEX_DLL_INTERFACE_H
#define AGXMEX_DLL_INTERFACE_H

#include <agxMex/export.h>


#ifdef __cplusplus
extern "C" {
#endif

  /**
  Initialize the "simulation", must be called prior to any of the other functions.
  Can only be called once.
  \param debugFile Which file should debug information be written to? 0 means stdout.
  \param agxFilePath The agx file path. Only needed in mixed OS-environments.
  \param agxPluginPath The agx plugin path. Only needed in mixed OS-environments.
  \param osgFilePath The osg file path. Only needed in mixed OS-environments.
  \return 0 on success, non-zero on failure.
  */
  AGXMEX_EXPORT int AGXMEX_initAgx( const char* debugFile, const char* agxFilePath, const char* agxPluginPath, const char* osgFilePath );

  /// Returns last error message. Empty if no error.
  AGXMEX_EXPORT const char* AGXMEX_getLastErrorMessage();

  /**
  Reset the "simulation".
  */
  AGXMEX_EXPORT int AGXMEX_resetAgx();

  /**
  Preloads the scene specified int a .agxLua file into to the simulation.
  \param file - path to a configuration file specifying the simulation content
  \param initInputSize: Number of elements in input array of doubles used in init function
  \param initOutputSize: Number of elements in output array of doubles used in init function
  \param stepInputSize: Number of elements in input array of doubles used in step function
  \param stepOutputSize: Number of elements in output array of doubles used in step function
  \return 0 on success, non-zero on failure.
  */
  AGXMEX_EXPORT int AGXMEX_preloadFileAgx( const char* file, int* initInputSize, int* initOutputSize,
    int* stepInputSize, int* stepOutputSize );

  /**
  Loads the scene specified int a .agxLua file into to the simulation.
  \param initInput The input to the init function. Should be of size initInputSize as
         given from the function AGXMEX_preloadFileAgx.
  \param initOutput The output from the init function. Should be of size initOutputSize as
         given from the function AGXMEX_preloadFileAgx.
  \return 0 on success, non-zero on failure.
  */
  AGXMEX_EXPORT int AGXMEX_loadFileAgx( double* initInput, double* initOutput );

  /**
  \param timeStep The simulation time step. E.g. 0.01 for a 100Hz simulation.
  \return 0 on success, non-zero on failure.
  */
  AGXMEX_EXPORT int AGXMEX_setTimeStepAgx( double timeStep );


  /**
  Step the "simulation".
  Requirement: AGXMEX_initAgx and AGXMEX_loadFileAgx must be called prior to this call.

  \param input - a pointer to an array with doubles containing in-data, size is specified by input_size
  \param output - a pointer to an array with doubles containing out-data, size is specified by output_size
  \return 0 on success, non-zero on failure.
  */
  AGXMEX_EXPORT int AGXMEX_stepAgx(const double* input, double* output);


  /**
  Sets if the simulation should use remote debugging.
  This will have a quite high performance cost, since the whole simulation will be sent at 100Hz.
  Requirement: AGXMEX_initAgx must be called prior to this call.

  \param flag: 0 for false (deactivate remote debugger),
               1 true (activate remote debugger)
               2 hidden (activate remote debugger, no graphics window)
  \return 0 on success, non-zero on failure.
  */
  AGXMEX_EXPORT int AGXMEX_setUseCoSimulation(int flag);

  /// Returns the size of the init input array for AGXMEX_stepAgx.
  AGXMEX_EXPORT int AGXMEX_getInitInputSize();

  /// Returns the size of the init output array for AGXMEX_stepAgx.
  AGXMEX_EXPORT int AGXMEX_getInitOutputSize();

  /// Returns the size of the step input array for AGXMEX_stepAgx.
  AGXMEX_EXPORT int AGXMEX_getStepInputSize();

  /// Returns the size of the step output array for AGXMEX_stepAgx.
  AGXMEX_EXPORT int AGXMEX_getStepOutputSize();

  /**
  Returns the name of a specific init input value for AGXMEX_stepAgx.
  \param valueNr The number of the value. Must be smaller than
         the return value of AGXMEX_getInputSize().
  \retval The name of the input value. 0 for invalid valueNr.
  */
  AGXMEX_EXPORT const char* AGXMEX_getInitInputValueName(unsigned int valueNr);

  /**
  Returns the name of a specific init output value for AGXMEX_stepAgx.
  \param valueNr The number of the value. Must be smaller than
         the return value of AGXMEX_getOutputSize().
  \retval The name of the output value. 0 for invalid valueNr.
  */
  AGXMEX_EXPORT const char* AGXMEX_getInitOutputValueName(unsigned int valueNr);

  /**
  Returns the name of a specific step input value for AGXMEX_stepAgx.
  \param valueNr The number of the value. Must be smaller than
         the return value of AGXMEX_getInputSize().
  \retval The name of the input value. 0 for invalid valueNr.
  */
  AGXMEX_EXPORT const char* AGXMEX_getStepInputValueName(unsigned int valueNr);

  /**
  Returns the name of a specific step output value for AGXMEX_stepAgx.
  \param valueNr The number of the value. Must be smaller than
         the return value of AGXMEX_getOutputSize().
  \retval The name of the output value. 0 for invalid valueNr.
  */
  AGXMEX_EXPORT const char* AGXMEX_getStepOutputValueName(unsigned int valueNr);

#ifdef __cplusplus
}
#endif

#endif // AGXMEX_DLL_INTERFACE_H
