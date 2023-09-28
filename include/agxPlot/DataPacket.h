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

#ifndef AGXPLOT_DATAPACKET_H
#define AGXPLOT_DATAPACKET_H

#include <agxPlot/Packet.h>
#include <agxPlot/Curve.h>

namespace agxPlot
{
  struct DataPoint
  {
    CurveID ID;
    CurvePoint point;
  };

  typedef agx::Vector<DataPoint> DataPointVector;

  class DataSeriesPacket : public Packet
  {
    public:
      DataSeriesPacket();

      void addData(agx::Real point);

      const agx::RealVector& getDataPoints() const;

    protected:
      virtual ~DataSeriesPacket();

    private:
      agx::RealVector m_dataPoints;
  };
  AGX_DECLARE_POINTER_TYPES(DataSeriesPacket);

  class DataPacket : public Packet
  {
  public:
    DataPacket();

    void addCurveData(CurveID curveId, CurvePoint point);
    void addCurveData(CurveID curveId, CurvePointVector points);

    const DataPointVector& getDataPoints() const;

  protected:
    virtual ~DataPacket();

  private:
    DataPointVector m_dataPoints;
  };
  AGX_DECLARE_POINTER_TYPES(DataPacket);
}

#endif
