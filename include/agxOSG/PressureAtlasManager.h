/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXOSG_PRESSURE_ATLAS_MANAGER_H
#define AGXOSG_PRESSURE_ATLAS_MANAGER_H

#include <agx/HashTable.h>
#include <agx/Object.h>

#include <agxSDK/StepEventListener.h>
#include <agxSDK/Simulation.h>

#include <agxOSG/export.h>
#include <agxOSG/PressureAtlas.h>

namespace agx
{
  class Constraint;
}

namespace agxCollide
{
  class Trimesh;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxOSG
{

  AGX_DECLARE_POINTER_TYPES(PressureAtlasManager);
  class AGXOSG_EXPORT PressureAtlasManager : public agx::Object
  {
  public:
    typedef agx::HashTable<const agxCollide::Trimesh *, std::pair<agxCollide::TrimeshConstObserver,agxOSG::PressureAtlasRef> > MeshToAtlasTable;

  public:
    PressureAtlasManager(agxSDK::Simulation* simulation, agx::Real defaultTexelsPerMeter);

    /**
     * Create a pressure atlas for a mesh. If a positive 'texelsPerMeter' is
     * given, then that resolution will be used for the atlas. Otherwise the
     * default resolution set for this manager is used.
     *
     * If the mesh has been registered already then nullptr will be returned. Use
     * 'getAtlas' to fetch the atlas for a already registered mesh.
     *
     * No pressure generators are created by the registering process. Use the
     * various 'create.*PressureGenerator' methods provided, or call 'addForce'
     * on the atlas through some other means.
     *
     * @param mesh The mesh for which a pressure map is to be generated.
     * @param texelsPerMeter The resolution of the pressure map. Passing a non-positive value gives the default.
     * @return A newly created PressureAtlas, or nullptr if the mesh was registered previously.
     */
    agxOSG::PressureAtlas* registerMesh(const agxCollide::Trimesh* mesh, agx::Real texelsPerMeter = agx::Real(-1.0));

    agxOSG::PressureAtlas* getAtlas(const agxCollide::Trimesh* mesh);

    const MeshToAtlasTable& getAtlases();

    void createContactPressureGenerator(const agxCollide::Trimesh* mesh);

    agxSDK::Simulation* getSimulation() const;

    /// Not implemented.
    void createConstraintPressureGenerator(const agxCollide::Trimesh* mesh, const agx::Constraint* constraint);

  protected:
    virtual ~PressureAtlasManager();


  private:
    AGX_DECLARE_POINTER_TYPES(AtlasCommitter);
    class AtlasCommitter : public agxSDK::StepEventListener
    {
    public:
      AtlasCommitter(PressureAtlasManager* master);
      virtual void post(const agx::TimeStamp& time);

    protected:
      virtual ~AtlasCommitter() {}

    private:
      PressureAtlasManager* m_master;
    };


    friend class AtlasCommitter;
    void commitPressures();
    AtlasCommitterRef m_committer;

  private:
    MeshToAtlasTable m_meshes;
    agxSDK::SimulationObserver m_simulation;
    agx::Real m_defaultTexelsPerMeter;
  };

}


#endif
