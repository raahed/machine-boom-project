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

#include <agx/version.h>
#include <agxStream/shouldReadDataFromArchive.h>
#include <agxStream/Serializable.h>
#include <agxStream/StorageStream.h>
#include <agxStream/SerializationVersions.h>

namespace agxPowerLine
{
  namespace detail
  {
    /**
    \return true if the inspected serialization contains data for the
            RotationalActuatorConnector and related classes. False otherwise.
    */
    inline bool shouldReadRotationalActuatorConnectorData(size_t archiveAgxVersion, agx::UInt16 archiveSerializationVersion)
    {

      /*
      The RotationalActuatorConnector was given serialization version 71 and merged
      to the RC along with a bump to AGX version 2.13.4.
      */
      agx::UInt16 const serializationVersionInRc = 58;
      agx::UInt16 const serializationVersionAtChange = 71;
      size_t const agxVersionInRc = AGX_CALC_VERSION(2, 13, 4, 0);
      size_t const agxVersionInTrunkAfterRc = AGX_CALC_VERSION(2, 14, 0, 0);

      return agxStream::detail::shouldReadDataFromArchive(
            archiveAgxVersion,
            archiveSerializationVersion,
            serializationVersionInRc,
            serializationVersionAtChange,
            agxVersionInRc,
            agxVersionInTrunkAfterRc);
    }



    /**
    \return true if the inspected serialization contain data for the Rotational-
            Dimension reference direction. False otherwise.
    */
    inline bool shouldReadRotationalDimensionDirectionReference(size_t archiveAgxVersion, agx::UInt16 archiveSerializationVersion)
    {
      /*
      The direction reference for RotationalDimension was given serizalizeation
      version 87 and merged to the RC along with a bump to AGX version 2.15.0.6.
      */
      const agx::UInt16 serializationVersionInRc = 80;
      const agx::UInt16 serializationVersionAtChange = AGXSTREAM_SERIALIZATION_VERSION_POWER_LINE_ROTAT_LOCAL_OR_WORLD_DIRECTION;
      const size_t agxVersionInRc = AGX_CALC_VERSION(2, 15, 0, 6);
      const size_t agxVersionInTrunkAfterRc = AGX_CALC_VERSION(2, 16, 0, 0);

      return agxStream::detail::shouldReadDataFromArchive(
        archiveAgxVersion,
        archiveSerializationVersion,
        serializationVersionInRc,
        serializationVersionAtChange,
        agxVersionInRc,
        agxVersionInTrunkAfterRc);
    }



    const agx::UInt16 MARKER_SERIALIZATION_VERSION = 92;

    // A bunch of randomly selected numbers.
    const agx::UInt16 POWERLINE_BEGIN_MARKER = 0x5b04;
    const agx::UInt16 POWERLINE_END_MARKER = 0x7245;
    const agx::UInt16 CONNECTOR_END_MARKER = 0x0543;
    const agx::UInt16 UNIT_END_MARKER = 0x4a73;
    const agx::UInt16 LIGHT_DATA_BEGIN_MARKER = 0x51b7;
    const agx::UInt16 LIGHT_DATA_END_MARKER = 0xfc4b;



    inline bool hasMarkers(agxStream::StorageStream& in)
    {
      return in.getSerializationVersion() >= MARKER_SERIALIZATION_VERSION;
    }


    template<typename Component>
    bool shouldStore(const Component& c)
    {
      return !c->getIgnoreForStoreRestoreStream();
    }


    template<typename Container>
    size_t countStoreRestorable(const Container& components)
    {
      using Component = typename Container::value_type;
      using namespace std;
      size_t count = count_if(begin(components), end(components), shouldStore<Component>);

      return count;
    }


    template<typename Container>
    void storeStorablesCount(const Container& components, agxStream::StorageStream& out)
    {
      size_t numStorables = countStoreRestorable(components);
      out << numStorables;
    }


    template<typename Container>
    bool restoreRestorablesCount(const Container& components, agxStream::StorageStream& in)
    {
      if (!hasMarkers(in))
        return true;

      size_t numStored;
      in >> numStored;

      size_t numRestorable = countStoreRestorable(components);

      bool match = numStored == numRestorable;
      if (!match)
      {
        LOGGER_WARNING() << "PowerLine stream restore given stream with incorrect number of connectors." << LOGGER_ENDL();
        LOGGER_WARNING() << "  Stream contains " << numStored << " and power line contains " << numRestorable << " restorable connectors." << LOGGER_ENDL();
        LOGGER_WARNING() << "  Aborting restore." << LOGGER_ENDL();
        return false;
      }
      return match;
    }


    inline void storePowerLineBeginMarker(agxStream::StorageStream& out)
    {
      out << POWERLINE_BEGIN_MARKER;
    }


    inline void storePowerLineEndMarker(agxStream::StorageStream& out)
    {
      out << POWERLINE_END_MARKER;
    }


    inline void storeConnectorEndMarker(agxStream::StorageStream& out)
    {
      out << CONNECTOR_END_MARKER;
    }


    inline void storeUnitEndMarker(agxStream::StorageStream& out)
    {
      out << UNIT_END_MARKER;
    }


    inline void storeLightDataBeginMarker(agxStream::StorageStream& out)
    {
      out << LIGHT_DATA_BEGIN_MARKER;
    }


    inline void storeLightDataEndMarker(agxStream::StorageStream& out)
    {
      out << LIGHT_DATA_END_MARKER;
    }




    inline bool restoreMarker(agxStream::StorageStream& in, agx::UInt16 marker, const char* name)
    {
      if (!hasMarkers(in))
        return true;

      uint16_t endMarker;
      in >> endMarker;

      bool match = endMarker == marker;
      if (!match)
        LOGGER_WARNING() << "PowerLine stream restore did not find " << name << " marker. Aborting restore." << LOGGER_ENDL();
      return match;
    }


    inline bool restorePowerLineBeginMarker(agxStream::StorageStream& in)
    {
      return restoreMarker(in, POWERLINE_BEGIN_MARKER, "start-of-powerline");
    }


    inline bool restorePowerLineEndMarker(agxStream::StorageStream& in)
    {
      return restoreMarker(in, POWERLINE_END_MARKER, "end-of-powerline");
    }


    inline bool restoreConnectorEndMarker(agxStream::StorageStream& in)
    {
      return restoreMarker(in, CONNECTOR_END_MARKER, "end-of-connector");
    }


    inline bool restoreUnitEndMarker(agxStream::StorageStream& in)
    {
      return restoreMarker(in, UNIT_END_MARKER, "end-of-unit");
    }


    inline bool restoreLightDataBeginMarker(agxStream::StorageStream& in)
    {
      return restoreMarker(in, LIGHT_DATA_BEGIN_MARKER, "start-of-light-data");
    }


    inline bool restoreLightDataEndMarker(agxStream::StorageStream& in)
    {
      return restoreMarker(in, LIGHT_DATA_END_MARKER, "end-of-light-data");
    }




    /**
    Member function signature for methods that read data from a StorageStream.
    Was intended to be \p restore and \p restoreLightData, but \p restore isn't
    part of Serializable so all of this is kind of moot. Maybe it will be added
    later. Also, the signatures are different.
    */
    typedef void (agxStream::Serializable::*RestoreMemFn)(agxStream::StorageStream&);


    /**
    Main worker helper for optinally restore. Checks the object pointer and the
    stream flag and calls the passed in restore member function is both the
    pointer and the stream flag says that reading is ok.
    */
    inline bool optionallyRestore_helper(
      agxStream::StorageStream& stream,
      agxStream::Serializable* object,
      RestoreMemFn restore,
      const char* const parentName,
      const char* const dataName)
    {
      // Determine who wants to read and who does not.
      const bool parentHasObject = object != nullptr;
      bool streamHasObject;
      stream >> streamHasObject;

      // Check if they agree on whether we should read or not.
      bool const bothHaveObject =       parentHasObject &&  streamHasObject;
      bool const neitherHasObject =    !parentHasObject && !streamHasObject;
      bool const onlyParentHasObject =  parentHasObject && !streamHasObject;
      bool const onlyStreamHasObject = !parentHasObject &&  streamHasObject;

      if (bothHaveObject)
      {
        (object->*restore)(stream);
        return true;
      }
      else if (neitherHasObject)
      {
        // Nothing to do.
        return true;
      }
      else if (onlyParentHasObject)
      {
        LOGGER_WARNING() << parentName << " found unexpected " << dataName << " data in StorageStream." << LOGGER_ENDL();
        return false;
      }
      else if (onlyStreamHasObject)
      {
        LOGGER_WARNING() << parentName << " expected " << dataName << " data in StoageStream, but there were none." << LOGGER_ENDL();
        return false;
      }
      else
      {
        agxAbort1("Logic error. All possible cases should have been handled already.");
        return false;
      }
    }



    inline void optionallyStoreLightData(
      agxStream::StorageStream& out,
      agxStream::Serializable* object)
    {
      storeLightDataBeginMarker(out);

      bool hasConstraint = object != nullptr;
      out << hasConstraint;
      if (hasConstraint)
        object->storeLightData(out);

      storeLightDataEndMarker(out);
    }



    /**
    Restores light data for the given \p object if the \p object is not nullptr
    and the stream contains data for it. Does not restore light data if \p object
    is nullptr and the stream contains no data. Returns true in both of these cases.

    Prints a warning and returns false if the \p object pointer and the stream
    contents is inconsistent.

    \param in - The stream to read from.
    \param object - The object to read into. May be nullptr.
    \param parentName - Becomes part of error message if the parent and stream disagree.
    \param dataName - Becomes part of the error message if the parent and stream disagree.
    \return True if the object and stream agree whether to read or not. False otherwise.
    */
    inline bool optionallyRestoreLightData(
      agxStream::StorageStream& in,
      agxStream::Serializable* object,
      const char * const parentName,
      const char * const dataName)
    {
      if (!restoreLightDataBeginMarker(in))
        return false;

      if (!optionallyRestore_helper(in, object, &agxStream::Serializable::restoreLightData, parentName, dataName))
        return false;

      if (!restoreLightDataEndMarker(in))
        return false;

      return true;
    }



    /**
    Restores data for the given \p object if the \p object is not nullptr
    and the stream contains data for it. Does not restore data if \p object
    is nullptr and the stream contains no data. Returns true in both of these
    cases, unless a read is attempted and failed.

    Prints a warning and returns false if the \p object pointer and the stream
    contents is inconsistent.

    \param stream - The stream to read from.
    \param object - The object to read into. May be nullptr.
    \param parentName - Becomes part of error message if the parent and stream disagree.
    \param dataName - Becomes part of the error message if the parent and stream disagree.
    \return True if the object and stream agree whether to read or not. False otherwise.
    */
    template<typename T>
    bool optionallyRestore(
      agxStream::StorageStream& stream,
      T* object,
      const char* const parentName,
      const char* const dataName)
    {
      const bool parentHasObject = object != nullptr;
      bool streamHasObject;
      stream >> streamHasObject;

      bool const bothHaveObject =       parentHasObject &&  streamHasObject;
      bool const neitherHasObject =    !parentHasObject && !streamHasObject;
      bool const onlyParentHasObject =  parentHasObject && !streamHasObject;
      bool const onlyStreamHasObject = !parentHasObject &&  streamHasObject;

      if (bothHaveObject)
      {
        return object->restore(stream);
      }
      else if (neitherHasObject)
      {
        // Nothing to do.
        return true;
      }
      else if (onlyParentHasObject)
      {
        LOGGER_WARNING() << parentName << " found unexpected " << dataName << " data in StorageStream." << LOGGER_ENDL();
        return false;
      }
      else if (onlyStreamHasObject)
      {
        LOGGER_WARNING() << parentName << " expected " << dataName << " data in StoageStream, but there were none." << LOGGER_ENDL();
        return false;
      }
      else
      {
        agxAbort1("Logic error. All possible cases should have been handled already.");
        return false;
      }
    }


  } // namespace detail.
} // namespace agxPowerLine.

