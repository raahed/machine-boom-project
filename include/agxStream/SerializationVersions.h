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
#pragma once


#define AGXSTREAM_SERIALIZATION_VERSION_ARCHIVE_DATE 81 // Added "date" to header which contains the date/time when archive was written
#define AGXSTREAM_SERIALIZATION_VERSION_STORE_WIREWINDCONTROLLER 85 // Added StepEventListener data to WireAndWindController
#define AGXSTREAM_SERIALIZATION_VERSION_MODIFICATION_HASH 86 // Added modification hash
#define AGXSTREAM_SERIALIZATION_VERSION_POWER_LINE_ROTAT_LOCAL_OR_WORLD_DIRECTION 87 // Serialization of local/world flag for RotationalDimension::m_direction.
#define AGXSTREAM_SERIALIZATION_VERSION_POWER_LINE_PHYS_DIM_FULL_VALUE_GRADIENT 88 // Serialization of complete 6-DOF state instead of get/set Value/Gradient. get/set is destructive.
#define AGXSTREAM_SERIALIZATION_VERSION_PACKED_POWERLINE_BODIES 89 // Serialization changes dues to support for body packing of power line dimensions.
