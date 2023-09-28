/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material without having a written signed agreement with Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/Json.h>
#include <agxData/EntityData.h>
#include <agxData/EntityStorage.h>
#include <agx/DirectSolverData.h>

namespace agx
{
  class AGXPHYSICS_EXPORT JsonDataDumpDebug
  {
  public:
    JsonDataDumpDebug();
    ~JsonDataDumpDebug();

    agxJson::Value& addStorage(agxData::EntityStorage *storage);
    agxJson::Value& addData(agxData::EntityData& data);
    agxJson::Value& addBuffer(agxData::Buffer *buffer);
    agxJson::Value& addSolverData(agx::DirectSolverData *data);

    agxJson::Value& addStorage(agxData::EntityStorage *storage, agxJson::Value& eParent);
    agxJson::Value& addData(agxData::EntityData& data, agxJson::Value& eParent);
    agxJson::Value& addBuffer(agxData::Buffer *buffer, agxJson::Value& eParent);
    agxJson::Value& addSolverData(agx::DirectSolverData *data, agxJson::Value& eParent);

    agxJson::Value& addRealValarray(const agx::RealValarray& array, agxJson::Value& eValues);

    void write(const agx::String& fileName);

  private:
    agxJson::Value m_root;
  };
}
