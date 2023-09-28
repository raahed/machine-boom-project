/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/Task.h>

#include <agx/agxPhysics_export.h>

namespace agxUtil
{
  struct AGXPHYSICS_EXPORT TaskGraphWriter
  {
    /**
    Write a .dot file containing all tasks and dependencies within the given
    task group.

    \param task - The task to write task- and dependency graph for.
    \param filename - The file to write the resulting graph to.
    \param withTempFile - If true, then the graph is written to a temporary file first and the file renamed when writing is done.
    \return True if the graph was written successfully. False otherwise.
    */
    static bool writeTaskGraph(const agx::TaskGroup& task, const char* filename, bool withTempFile = true);
  };
}
