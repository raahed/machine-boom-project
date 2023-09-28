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



#ifndef AGXPOWERLINE_DOT_GRAPH_WRITER_H
#define AGXPOWERLINE_DOT_GRAPH_WRITER_H

#include <agx/Referenced.h>
#include <agx/HashTable.h>
#include <agxPowerLine/PowerLine.h>

namespace agxPowerLine
{
  class PowerLine;

  AGX_DECLARE_POINTER_TYPES( DotGraphWriter );


  class AGXMODEL_EXPORT DotGraphWriter : public agx::Referenced
  {
  public:
    DotGraphWriter(const char* rotRateUnit = "rad/s", const char* transRateUnit = "m/s");

    void registerUnit(const char* dimensionName, const char* unitName);

    bool writeDimensionsGraph(const agx::String& filename, const PowerLine* powerline);
    bool writeDimensionsGraph(const agx::String& filename, const agxPowerLine::Unit* startUnit);
    bool writeDimensionsGraph(const agx::String& filename, const UnitPtrPowerLinePtrHashVector& roots);

    //bool writeUnitGraph(const char* filename, agxPowerLine::PowerLine* powerline);

  protected:
    virtual ~DotGraphWriter() {}

  private:
    void clear();
    bool writeDimensionsGraph(const agx::String& filename);
    void visitDimensionConnections( const agxPowerLine::ConnectionRefVector& connections );
    void visitConnectorDimensions( const agxPowerLine::ConnectionRefVector& connections );

  private:
    typedef agx::HashTable<agx::String, agx::String> UnitsTable;
    UnitsTable m_units;

    typedef agx::Vector<const PhysicalDimension*> DimensionVector;
    typedef agx::Vector<const Connector*> ConnectorVector;

    // The index of each dimension and connector in these vectors determines it's
    // id within the DOT graph. Each dimension and connector may only be represented
    // once.
    DimensionVector m_seenDimensions;
    ConnectorVector m_seenConnectors;

    typedef agx::HashTable<const agxPowerLine::PhysicalDimension*, size_t> DimensionToIndexTable;
    typedef agx::HashTable<const agxPowerLine::Connector*, size_t> ConnectorToIndexTable;

    // The index of each dimension and connector is kept in these hash tables.
    DimensionToIndexTable m_dimensionToIndexTable;
    ConnectorToIndexTable m_connectorToIndexTable;

    typedef agx::Vector<const Unit*> UnitVector;

    UnitVector m_queue;
  };
}



#endif

