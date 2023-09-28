/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
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

#include <agx/agx.h>
#include <agx/String.h>
#include <agx/agxCore_export.h>
#include <agxData/JournalArchive.h>

namespace agx
{
  /**
   * A Chrome trace is a thread time line written in the chrome tracing
   * format. It can be viewed in any Chrome-style browser such Chrome, Chromium,
   * and Brave. And possibly others as well.
   *
   * The trace is written to a single file name ChromeTracing.json that contains
   * the thread timelines for all frames stored in the profiling journal
   * session.
   *
   * Open chrome://tracing/ in your browser, click Load in the top-left, and
   * select the ChromeTracing.json file.
   */
  void AGXCORE_EXPORT generateChromeTracing(const String& rootPath, agxData::JournalArchive::Session* journalSession);
}
