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

#ifndef AGX_ANALYSISBOX_H
#define AGX_ANALYSISBOX_H

#include <agx/Referenced.h>
#include <agx/Bound.h>
#include <agxSDK/Simulation.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(AnalysisBox);

  /**
  Class that can be used to extract data from elements in a simulation within a specified bound.
  */
  class AGXPHYSICS_EXPORT AnalysisBox : public agx::Referenced
  {
  public:
    typedef HashSet<agx::Index> ParticleIdSet;

  public:
    /**
    Constructor for the AnalysisBox.

    \param simulation The simulation to couple the analysis box too.
    \param center Center of the Analysis Box.
    \param halfVec Half vector that span the bound.
    */
    AnalysisBox(agxSDK::Simulation * simulation, const agx::Vec3& center, const agx::Vec3& halfVec);

    /// Update the analysis box contents.
    void updateContents();

    /// Update the bounds of the Analysis Box.
    void setAnalysisBound(const agx::Vec3& center, const agx::Vec3& halfVec);

    /// Get the bound of the Analysis Box.
    agx::Bound3 getAnalysisBound() const;

    /// Return the particle mass in the particles inside the Analysis Box.
    Real getParticleMassInsideBound();

    /// Returns the particles ids inside the Analysis Box.
    ParticleIdSet getParticlesInside() const;

    /// Returns true/false if the particle given the specified particle id is inside the box.
    bool containsParticle(agx::Index particleId) const;

    /// Returns true/false if the point is contained inside the Analysis Box bound.
    bool containsPoint(const agx::Vec3& point);

    /// Return true/false if the box is enabled.
    bool getEnable() const;

    /// Set true/false to enable/enable the Analysis Box.
    void setEnable(bool enable);

    /// Get the center point of the Analysis Box bound.
    agx::Vec3 getBoundCenter();

    /// Get the half vector of the Analysis Box bound.
    agx::Vec3 getBoundHalfVec();

    void printContents();

  protected:
    /// Virtual Destructor
    virtual ~AnalysisBox();
    void updateBound(const agx::Vec3& center, const agx::Vec3& halfVec);

    agx::ParticleSystem * getParticleSystem();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxSDK::Simulation * m_simulation;
    agx::Bound3          m_bound;
    RigidBodyPtrVector   m_rigidBodiesInside;
    ParticleIdSet        m_particleIdsInside;
    bool                 m_enabled;
  };

  AGX_FORCE_INLINE agx::Bound3 AnalysisBox::getAnalysisBound() const { return m_bound; }

  AGX_FORCE_INLINE agx::AnalysisBox::ParticleIdSet AnalysisBox::getParticlesInside() const { return m_particleIdsInside; }

  AGX_FORCE_INLINE bool AnalysisBox::getEnable() const { return m_enabled; }

  AGX_FORCE_INLINE void AnalysisBox::setEnable(bool enable) { m_enabled = enable; }
}

#endif