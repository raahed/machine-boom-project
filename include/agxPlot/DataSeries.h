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

#ifndef AGXPLOT_DATASERIES_H
#define AGXPLOT_DATASERIES_H

#include <agxPlot/DataGenerator.h>
#include <agxPlot/CallbackDataGenerator.h>
#include <agxPlot/Queue.h>

#include <agx/Uuid.h>
#include <agx/agx_vector_types.h>
#include <agx/Event.h>

#include <agxUtil/Statistic.h>

#include <functional>
#include <memory>

namespace agxSDK
{
  class Simulation;
}

namespace agxPlot
{
  class AGXPHYSICS_EXPORT DataSeries : public agx::Referenced
  {
    public:
      enum Settings
      {
        DATASERIES_SETTINGS_NONE = 0,
        DATASERIES_SETTINGS_ISTIME = 1
      };

      /**
       Create a DataSeries that takes fields from the rigidBody specified by the entity ID. A good way to
       directly access fields in the Journal.
      */
      DataSeries(const agx::Uuid& entityId, const agx::String& dataPath, agxSDK::Simulation* simulation, const agx::String& name);

      /**
       Create a DataSeries that calls the specified function to generate data.
      */
      DataSeries(std::function<double()> lambdaFunction, const agx::String& name);

      /**
       Create a DataSeries that calls the getValue operator of the DataListener to generate data.
      */
      DataSeries(agxPlot::DataListener* listener, const agx::String& name);

      /**
      Create a DataSeries that calls getValue on the given DataGenerator to generate data.
      */
      DataSeries(agxPlot::DataGenerator* dataGenerator, const agx::String& name);

      /**
      Create a DataSeries that calls the getValue operator of the DataListener to generate data.
      */
      DataSeries(const agx::String& dataPath, agxSDK::Simulation* simulation, const agx::String& name);

      /**
       Create a DataSeries that gets fed by the source specified. A good way to connect simulations.
      */
      DataSeries(agxPlot::DataSeries* source);

      /**
       Create a DataSeries that gets fed by two sources and adding their data. Used for contacts where
       we want a statistic on one of them but not the other.

       Sets name and unit from sourceA
      */
      DataSeries(agxPlot::DataSeries* sourceA, agxPlot::DataSeries* sourceB);

      /**
       Create a DataSeries that can only get values by manually pushing them.
      */
      DataSeries(const agx::String& name = "");

      /**
       Manually add a value at the end of the DataSeries.
       \param value the new value to append
       */
      void push(agx::Real value);

      /**
       Fetches the current value from the specified way of generation and
       append it to the series.
       */
      void generateCurrentValue();

      /**
       Get the index of the oldest new value. All values after this are 'dirty'.
       */
      size_t getNewValueMarker() const;

      /**
       Get all values that have been generated so far
       */
      const agx::RealVector& getAllValues() const;

      /**
      Remove all values in stored
      */
      void clearData();

      /**
       All current values are now observed and dirty flags can be cleaned up.
       */
      void resetDirty();

      /**
       Set the name of the data series.
       */
      void setName(const agx::String& name);

      /**
       Get the name of the data series.
      */
      const agx::String& getName() const;

      /**
       Set a string representing the unit of the data series.
      */
      void setUnit(const agx::String& unit);

      /**
       Get the unit of the data series.
      */
      const agx::String& getUnit() const;

      /**
       Should the axis be logarithmic if displayed
      */
      void setLogarithmic(bool isLogarithmic);

      bool getLogarithmic() const;

      /**
      Set special settings for the data series
      */
      void setSettings(agx::UInt16 settings);

      /**
       Get the description of the following format

       ["name"] = the name of the data series (axis)
       ["unit"] = the unit of the data series
       ["isLogarithmic"] = should the data series be viewed as logarithmic
       ["isTime"] = the data series represents time
       ["id"] = the id of the data series
      */
      void getDescription(agxJson::Value& root) const;
      /**
       Set from a description of the format from getDescription
      */
      void setFromDescription(const agxJson::Value& root);

      /**
      Pass data read from the data generator through the given statistic before
      adding it to the data series.

      This DataSeries will take ownership of the given statistic object and will
      call delete on it when destroyed.
      */
      void setStatistic(agxUtil::Statistic* statistic);

      /**
      Pass data read from the data generator through the given statistic before
      adding it to the data series.

      This DataSeries will take ownership of the statitic in the given statistic
      handle and will call delete on it when destroyed. The given statistic
      handle will be empty and cannot be reused when setStatistic returns.
      */
      void setStatistic(agxUtil::StatisticHandleRef statistic);


      /**
      Return the statistic, if any, through which data from generator is passed
      before being added to the data series. May be nullptr.
      */
      const agxUtil::Statistic* getStatistic() const;

      // INTERNAL
      /**
      Time works different, thus the time data series have
      this flag.
      */
      bool isTime() const;
      bool hasImportedJournalData() const;
      bool hasDataPath() const;
      agx::Uuid getEntityId() const;
      void importedJournal();
      const agx::String& getDataPath() const;
      agx::UInt getID() const;

      agx::Event representationChanged;

    protected:
      virtual ~DataSeries();

      void init(DataGeneratorRef generator, const agx::String& name);

    private:
      void addOutput(Queue* queue);
      void setID(agx::UInt id);
      bool isMultipleSource() const;
      void pushSecondary(agx::Real value);
      void sumValues();

      agx::RealVector m_values;
      agx::RealVector m_secondaryValues;
      agx::RealVector m_sumValues;
      size_t m_currentSumIndex;

      agx::String m_name;
      agx::String m_unit;
      bool m_isLogarithmic;
      bool m_journalDataImported;
      agx::UInt16 m_settings;

      agx::UInt m_id;

      size_t m_currentIndex;

      DataGeneratorRef m_generator;
      std::unique_ptr<agxUtil::Statistic> m_statistic;

      agx::Uuid m_entityId;
      agx::String m_dataPath;

      QueueRef m_inputQueue;
      QueueRef m_secondaryInputQueue;
      agx::Vector<Queue*> m_outputQueues;
  };

  AGX_DECLARE_POINTER_TYPES(DataSeries);
  AGX_DECLARE_VECTOR_TYPES(DataSeries);

  class AGXPHYSICS_EXPORT TimeDataSeries : public DataSeries
  {
  public:
    // Create a data series that makes a time lambda function connected to the selected simulation
    TimeDataSeries(agxSDK::Simulation*);

  protected:
    virtual ~TimeDataSeries();

  private:
    agxSDK::Simulation* m_simulation;
  };

  AGX_DECLARE_POINTER_TYPES(TimeDataSeries);
  AGX_DECLARE_VECTOR_TYPES(TimeDataSeries);
}

#endif
