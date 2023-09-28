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



#ifndef AGXPOWERLINE_POWER_LINE_CONTROLLER_H
#define AGXPOWERLINE_POWER_LINE_CONTROLLER_H


#include <agxModel/export.h>
#include <agxPowerLine/Unit.h>
#include <agxPowerLine/PowerLine.h>

#include <agx/Singleton.h>
#include <agx/HashTable.h>
#include <agx/HashVector.h>

#include <functional>


namespace agxPowerLine
{

  /**
  A PowerLinesHandler is a collection of PowerLines that are part of the same
  agxSDK::Simulation. It passes on events from the simulation to the individual
  PowerLines.
  */
  class PowerLinesHandler : public agxSDK::StepEventListener
  {
  public:
    /**
    Called from the PowerLine's addNotification, via the PowerLineController,
    when a PowerLine has been added to a Simulation.
    */
    void addPowerLine(agxPowerLine::PowerLine* powerLine);

    /**
    Called from the PowerLine's removeNotification, via the PowerLineController,
    when a PowerLine has been removed from a Simulation.
    */
    void removePowerLine(agxPowerLine::PowerLine* powerLine);

    /**
    \return True if there are no PowerLines associated with this PowerLinesHandler.
    */
    bool empty() const;

    virtual void pre(const agx::TimeStamp& time) override;
    virtual void post(const agx::TimeStamp& time) override;

    /**
    Provides read access to the list of PowerLines that are part of this PowerLinesHandler.
    */
    const agxPowerLine::PowerLineRefVector& getPowerLines();

  private:
    virtual ~PowerLinesHandler();
    agxPowerLine::PowerLineRefVector m_powerLines;
  };

  typedef agx::ref_ptr<PowerLinesHandler> PowerLinesHandlerRef;



  /**
  Each type of physical dimension handled by the power line has to have a
  unique name. When a physical dimension object is created its unique name
  will be indexed and hashed in the power line dimension name singleton.

  The indices could as well be a pre defined enum. But then each new
  implementation of a physical dimension would have to be added to the enum and
  AGX would have to be recompiled.

  Using this approach a user with only the agx binaries and .h files can add
  his own physical dimension.
  */
  class AGXMODEL_EXPORT PowerLineController : public agx::Singleton
  {
    public:
      typedef std::function<Connector*()> ConnectorFactory;

    public:
      /**
      Add a named physical dimension. (introduce it to the name singleton and
      the simulation).
      */
      bool addPhysicalDimension(std::string name);

      /**
      The PowerLineController is in charge of keeping reference pointers to connectors that are created
      but not added to the simulation yet.
      This is needed since ConnectOperations creates default connectors that are hidden to the user.
      */
      void add(agxPowerLine::Connector* connector);
      void remove(agxPowerLine::Connector* connector);


      bool getPowerLines(agxSDK::Simulation* simulation, PowerLinePtrVector& powerLines);

      /**
      \return the indexed value for a named physical dimension. Will index a
      non existing name by a call to bool addPhysicalDimension(std::string
      name);

      \returns -1 if the name is not indexed and the name fails to be
      indexed(how?).
      */
      int getDimensionType(std::string name);

      /**
      \return the name (string) of the physical dimension with ID
      */
      std::string getDimensionTypeName(size_t id);

      /**
      \returns the number of different types of physical dimensions introduced
      (added) to the simulation
      */
      agxPowerLine::PhysicalDimension::Type getNumberOfPhysicalDimensions();

      /**
      Used while preparing power line for being restored.
      \param currentId - the id for the physical dimension in this simulation
      \param previousId - the id for the physical dimension in the restored simulation
      */
      void setStoredPhysicalDimensionId(int currentId, int previousId);

      /**
      Return true if mappings for stored dimensions IDs has been configured.
      False otherwise.

      The mappings are controlled by PowerLine and configured during PowerLine::
      restore. Mappings are unavailable if PowerLine components are serialized
      without going via a PowerLine.
      */
      bool hasStoredDimensionsMappings() const;

      /**
      Usable during restore of power line to map between indices in
      PhysicalDimension::m_dimensions. If there is no mapping then agx::InvalidIndex
      is returned.

      \param currentId - The id for the physical dimension in this simulation.
      \returns The id of a specific physical dimension id, what it was in the stored simulation.
      */
      int getStoredPhysicalDimensionId(int currentId);

      /**
      Converts a stored PhysicalDimension type ID to the corresponding ID for
      the current PowerLineController.

      If no mapping has been registered for the given ID then agx::InvalidIndex
      is returned.

      \param storedId A PhysicalDimension type ID read from a serialization.
      \return The PhysicalDimension type ID currently used for the same dimension type.
      */
      int getCurrentPhysicalDimensionId(int storedId);

      /**
      Clear the stored-to-current type ID mappings so that future PowerLine restores
      doesn't accidentally use stored IDs from the previous restore.
      */
      void clearStoredPhysicalDimensionIds();


      void registerDefaultConnector(
          agxPowerLine::PhysicalDimension::Type inputType,
          agxPowerLine::PhysicalDimension::Type outputType,
          ConnectorFactory connectorCreator);

      agxPowerLine::Connector* createConnector(
          agxPowerLine::PhysicalDimension::Type inputType,
          agxPowerLine::PhysicalDimension::Type outputType) const;

      /**
      \return a pointer to the PowerLineDimensionNameSingleton singleton.
      */
      static PowerLineController* instance();

      /**
      \return true if Singleton is initialized (instance() returns != 0)
      */
      static bool isInitialized();


    private:
      friend class agxPowerLine::PowerLine;
      void registerPowerLine(agxPowerLine::PowerLine* powerLine);
      void unregisterPowerLine(agxPowerLine::PowerLine* powerLine, agxSDK::Simulation* simulation);

    protected:
      typedef agx::HashTable<std::string, int> DimensionNameIdTable;


      PowerLineController();

      virtual ~PowerLineController();

      friend class PhysicalDimension;
      void replaceIndexWhenRestoring(std::string name, int index);
      DimensionNameIdTable::iterator findIndex(int index);
      virtual void shutdown() override;

      DimensionNameIdTable m_nameIdTable;
      agx::HashVector<int,int> m_storeRestoreIdTranslator;

      typedef agx::HashTable<agxSDK::Simulation*, PowerLinesHandlerRef> SimulationPowerLinesHash;
      SimulationPowerLinesHash m_powerLinesInSimulation;

      typedef agx::SymmetricPair<agxPowerLine::PhysicalDimension::Type> DimTypePair;
      typedef agx::HashTable<DimTypePair, ConnectorFactory> ConnectorFactoryTable;
      ConnectorFactoryTable m_connectorFactories;

    private:
#ifndef SWIG
      SINGLETON_CLASSNAME_METHOD()
#endif
      static PowerLineController* s_instance;

      ConnectorRefSet m_connectors;

  };
}


#endif
