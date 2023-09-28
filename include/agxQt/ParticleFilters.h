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

#ifndef AGXQT_FILTER_H
#define AGXQT_FILTER_H

#include <agx/config/AGX_USE_KEYSHOT.h>
#include <agxQt/export.h>
#include <agx/Referenced.h>
#include <agx/Physics/GranularBodySystem.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxQt
{

  // Forward declaration
  class ParticleRenderFilter;

  /**
  Struct containing information and constructors for a specific particle filter, that removes rendering
  of particles based on a critera determined by sublcasses of ParticleRenderFilter.
  */
  typedef std::function<ParticleRenderFilter*()> ParticleFilterConstructor;
  struct AGXQT_EXPORT ParticleFilterMode
  {
    agx::Name                 m_id;
    ParticleFilterConstructor m_constructor;
    agx::Name                 m_unit;

    bool operator==(const ParticleFilterMode& rhs) { return m_id == rhs.m_id; }
  };
  typedef agx::Vector<ParticleFilterMode> ParticleRenderFilterModeVector;

  /**
  This is a manager class that has the resposibility of filtering particles, by excluding rendering,
  on particles according to a criteria determined by the active ParticleRenderFilter.
  */
  AGX_DECLARE_POINTER_TYPES(ParticleRenderFilterManager);
  class ParticleRenderFilterManager : public agx::Referenced
  {
  public:
    typedef agx::HashTable<agx::Name, ParticleFilterMode> ParticleColorModeTable;

  public:
    ParticleRenderFilterManager();

    /// Execute on particles before rendering.
    void preRender(agx::ParticleSystem * system);

    /// Execute after render state.
    void postRender(agx::ParticleSystem * system);

    /**
    Switches active particle render filter based on name identifier. A particle filter with
    the specified name must exist in the manager for the switch to happen.
    \param name The name identifier of a particle filter already added in the manager.
    */
    void setActiveParticleRenderFilterFromString(const agx::String& name);

    /**
    Switches active particle render filter based on a ParticleFilter mode. This mode must already exist
    in the manager for the switch to happen.
    \param mode The mode identifier of a particle filter already added in the manager.
    */
    void setActiveParticleRenderFilter(ParticleFilterMode mode);

    /**
    Returns the ParticleFilterMode struct for the active filter in the manager.
    \return The mode identifier of the active particle filter.
    */
    ParticleFilterMode getActiveParticleRenderFilterMode() const;

    /**
    Returns the ParticleFilterMode structs added to the manager.
    \return The modes currently existing in the filter manager
    */
    ParticleRenderFilterModeVector getAvailableParticleRenderModes() const;

    /**
    Returns true/false if an ParticleRenderFilter with given name identifier exists in the manager.
    \param name The name identifier for the ParticleRenderFilter
    \return true/false if ParticleRenderFilter exists.
    */
    bool particleRenderFilterExists(const agx::String& name);

    /**
    Sets the filter threshold of the ParticleRenderFilter. What this means and how it is interpreted
    depends on the currently active filter. What it usually means is that if a certain particle value
    is beneth the threshold, the particle will be filitered out.
    \param threshold The filter threshold scalar to be used in the particle filter.
    */
    void setFilterThresholdScalar(agx::Real threshold);

    /**
    Returns the filter threshold of the ParticleRenderFilter. What this means and how it is interpreted
    depends on the currently active filter. What it usually means is that if a certain particle value
    is beneth the threshold, the particle will be filitered out.
    \return The filter threshold scalar that is used in the particle filter.
    */
    agx::Real getFilterThresholdScalar() const;

    /**
    Resets the filtered state in the particle system. Usually used after setting the "None" filter as active.
    */
    void resetParticleFilterState(agx::ParticleSystem * system) const;

  protected:
    ~ParticleRenderFilterManager();

    void initStandardFilters();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    ParticleColorModeTable  m_availableParticleFilters;
    ParticleRenderFilter*   m_activeParticleFilter;
    agx::Real               m_scalarThreshold;
  };

  /**
  This is a base abstract class for filtering particles. How particles are filtered is determined
  by the implementation of child classes.

  The filtering is accomplished by setting particle alpha values to 0, which will cause them to be invisible.
  */
  class ParticleRenderFilter : public agx::Referenced
  {
  public:
    static const char * const PARTICLEFILTER_NONE_NAME;

  public:
    /// Execute on particles before rendering.
    virtual void preRender(agx::ParticleSystem * system) = 0;

    /// Execute after render state.
    virtual void postRender(agx::ParticleSystem * system) = 0;

    /**
    Sets the filter threshold of the ParticleRenderFilter. What this means and how it is interpreted
    depends on the derived child classes. What it usually means is that if a certain particle value
    is beneth the threshold, the particle will be filitered out.
    \param scalarThreshold The filter threshold scalar to be used in the particle filter.
    */
    void setScalarRenderThreshold(agx::Real scalarThreshold);

    /**
    Returns the name type identifier for the ParticleFilter. This identifier should be unique for each
    specific type of particle filter. Each subclass get their own name, except for the Labmbda filter which
    requires the user to determined the name type.
    */
    agx::Name getName() const;

  protected:
    ParticleRenderFilter(const agx::Name& name);

    void filterParticle(agx::Physics::ParticlePtr ptr, bool shouldFilter);

    virtual ~ParticleRenderFilter();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Name m_name;
    agx::Real m_scalarThreshold;
  };

  AGX_FORCE_INLINE agx::Name agxQt::ParticleRenderFilter::getName() const { return m_name; }

  /**
  * This is a class that filters particles from contact energy
  */
  class AccumulatedContactEnergyFilter : public ParticleRenderFilter
  {
  public:
    AccumulatedContactEnergyFilter();

    virtual void preRender(agx::ParticleSystem * system) override;

    virtual void postRender(agx::ParticleSystem * system) override;

  protected:
    virtual ~AccumulatedContactEnergyFilter();
  };

  /**
  * This is a class that filters particles from certain criteria
  */
  class ParticleVelocityRenderFilter : public ParticleRenderFilter
  {
  public:
    ParticleVelocityRenderFilter();

    virtual void preRender(agx::ParticleSystem * system) override;

    virtual void postRender(agx::ParticleSystem * system) override;

  protected:
    virtual ~ParticleVelocityRenderFilter();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
  };

  /**
  * This is a class that filters particles from certain criteria
  */
  class LambdaParticleRenderFilter : public ParticleRenderFilter
  {
  public:
    typedef std::function<bool(agx::ParticleSystem*, agx::Physics::ParticlePtr)> ParticleFilterFunction;

  public:
    LambdaParticleRenderFilter(const agx::Name& name, const ParticleFilterFunction& function);

    virtual void preRender(agx::ParticleSystem * system) override;

    virtual void postRender(agx::ParticleSystem * system) override;

  protected:
    virtual ~LambdaParticleRenderFilter();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    ParticleFilterFunction m_filterFunction;
  };

  AGX_FORCE_INLINE ParticleFilterMode AGXQT_EXPORT createNoneParticleFilterMode()
  {
    return ParticleFilterMode{ ParticleRenderFilter::PARTICLEFILTER_NONE_NAME, []{ return nullptr; }, "" };
  }
}

#endif /*AGXQT_QTOSGUTILS_H*/
