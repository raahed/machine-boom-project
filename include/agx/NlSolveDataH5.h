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

#include <agx/config/AGX_USE_HDF5.h>
#include <agx/Referenced.h>

#include <array>

namespace agx
{
  class AGXCORE_EXPORT INlSolveDataH5 : public Referenced
  {
    public:
      struct Data
      {
        Data( agx::Real lowerBound,
              agx::Real upperBound,
              agx::Real normalForce,
              agx::Real frictionCoefficient )
          : lowerBound( lowerBound ),
            upperBound( upperBound ),
            normalForce( normalForce ),
            frictionCoefficient( frictionCoefficient )
        {
        }

        agx::Real lowerBound;
        agx::Real upperBound;
        agx::Real normalForce;
        agx::Real frictionCoefficient;
      };

      enum DataType
      {
        CONTACT_POINT = 0,
        CONSTRAINT,
        OTHER = 128
      };

    public:
      virtual void beginSimulationStep() = 0;
      virtual void endSimulationStep() = 0;

      virtual void beginSolve() = 0;
      virtual void endSolve() = 0;

      virtual void beginIteration() = 0;
      virtual void endIteration() = 0;

      virtual void write( DataType dataType, const Data& data ) = 0;

      virtual agx::String getFilename() const = 0;
      virtual void close() = 0;
  };
}

#if AGX_USE_HDF5()
#include <agx/AgxH5.h>

namespace agx
{
  class AGXCORE_EXPORT NlSolveDataH5 : public INlSolveDataH5
  {
    public:
      enum IterationDataEntryName : agx::UInt
      {
        LOWER_BOUND,
        UPPER_BOUND,
        NORMAL_FORCE,
        FRICTION_COEFFICIENT,
        NUM_ITERATION_DATA_ENTRIES
      };

      using IterationDataEntry = VectorPOD<Real>;
      using IterationData = std::array<IterationDataEntry, NUM_ITERATION_DATA_ENTRIES>;

      struct IterationNode
      {
        IterationNode( DataType type )
          : type( type )
        {
        }

        DataType type;
        IterationData data;
      };

      using IterationNodeVector = Vector<IterationNode>;

    public:
      NlSolveDataH5( const agx::String& filename );

      virtual void beginSimulationStep() override;
      virtual void endSimulationStep() override;

      virtual void beginSolve() override;
      virtual void endSolve() override;

      virtual void beginIteration() override;
      virtual void endIteration() override;

      virtual void write( DataType dataType, const Data& data ) override;

      virtual agx::String getFilename() const override;
      virtual void close() override;

    protected:
      virtual ~NlSolveDataH5();

    private:
      agx::String m_filename;
      IterationNodeVector m_iterationNodes;
      H5::H5File* m_file;
      H5::Group m_root;
      H5::Group m_simulationStepGroup;
      agx::UInt m_simulationStepCounter;
      agx::UInt m_solveCounter;
      agx::UInt m_iterationCounter;
      agx::UInt m_iterationDataCounter;
  };
}
#endif
