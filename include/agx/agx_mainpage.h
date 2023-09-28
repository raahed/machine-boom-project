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

/**
\mainpage API Reference

\section intro Introduction

AGX Dynamics is a multi-physics engine written in platform independent C++.

AGX is built/tested on:

- Windows Windows 10; both 32 and 64 bit. Compilers: Visual studio 2017, 2019, 2022
- Linux, Ubuntu 18.04 (gcc 7.3.0), Ubuntu 20.04 (gcc 9.3.0), Ubuntu 22.04 (gcc 11.3.0)
- Mac OS X (clang 11.0.0)

AGX Dynamics is designed to be a modern, easy to use, object oriented API, with access to
efficient numerical implementations and different types of solvers depending on the circumstances.

The API is divided into a number of namespaces:

- \ref agx - contains the dynamic simulation part of the system.
- \ref agxCable - Simulate flexible cables/ropes
- \ref agxCollide - contains the geometric intersection system (collision detection).
the dynamical simulation system including Material, event listeners etc.
- \ref agxFMI2 - Functional Mockup Interface version 2.0
- \ref agxDriveTrain - Classes for building complete drive trains including clutches, gearboxes etc..
- \ref agxHydraulics - Classes for building hydraulic systems
- \ref agxIO - For reading/writing data
- \ref agxMex - Coupling to Matlab/Simulink
- \ref agxModel - Higher level modeling primitives such as Tree, Terrain.
- \ref agxNet - Network API
- \ref agxOSG - Utility classes for using AGX together with the rendering scene graph OpenSceneGraph. Not required.
- \ref agxPowerLine - Framework for building drivetrain components.
- \ref agxRender - Debug rendering and color classes
- \ref agxSensor - Gamepad/Joystick drivers
- \ref agxStream - contains classes for streaming data into archives for serialization.
- \ref agxSDK - contains classes to bridge the collision detection system and
- \ref agxTerrain - A terrain model based a 3D grid model with overlapping height field that can be deformed by interacting shovels objects performing digging motions, converting solid mass to dynamic mass which can be moved.
- \ref agxUtil - Utility classes.
- \ref agxVehicle - Classes for modelling vehicles
- \ref agxWire - API for creating wires with dynamic resolution.

\section dependencies Dependencies

- CMake for creating build files, http://www.cmake.org/
- Other dependencies are automatically downloaded from the configuration system or part of the installer.

\section installation Installation
For more information, access to a binary or source distribution or license,
please send an email to contact@algoryx.se.


\section contactInfo Contact information
Algoryx Simulation AB<br>
Uminova Science Park<br>
Kuratorvägen 2B<br>
907 36 Umeå, Sweden <br>
<br>
WWW: http://www.algoryx.se/ <br>
E-mail: <a href="mailto:info@algoryx.se">info@algoryx.se</a> <br>
Phone: +46-90-3484990 <br>
Cell: +46-70-6315520 (Kenneth Bodin) <br>

\section documentation Documentation

- <a href="UserManual/source/changelog.html">Changelog</a>
- <a href="UserManual/source/index.html">AGX User Manual</a>
- \ref Tutorials

\section License
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

The integrity of this Software is protected by technical protection measures (TPM)
so that the intellectual property rights, including copyright, in the Software of
AGX are not misappropriated. You must not attempt in any way to remove or circumvent any such TPM,
nor apply or manufacture for sale or hire, import, distribute, sell or let for hire,
offer or expose for sale or hire, advertise for sale or hire or have in your possession
for private or commercial purposes any means the sole intended purpose of which is
to facilitate the unauthorized removal or circumvention of such TPM.


*/
#include <agx/tutorials.h>
