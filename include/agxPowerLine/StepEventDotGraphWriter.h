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

#pragma once

#include <agxSDK/StepEventListener.h>
#include <agxPowerLine/PowerLine.h>
#include <agxPowerLine/DotGraphWriter.h>

namespace agxPowerLine
{
    AGX_DECLARE_POINTER_TYPES(StepEventDotGraphWriter);

    /**
    A StepEventListener that writes a power line.dot graph in the LAST event.
    The output files is named "dimensions.dot".
    */
    class AGXMODEL_EXPORT StepEventDotGraphWriter : public agxSDK::StepEventListener
    {
    public:
        /**
        \param powerline - The power line whose graph should be written.
        */
        StepEventDotGraphWriter(agxPowerLine::PowerLine* powerline);

        /**
        Register a unit string to be displayed when the gradient of physical
        dimensions of the given dimension type is given. Units "m/s" and
        "rad/s", for translational and rotational dimensions respectively, are
        registed by default.

        \param dimensionName - The static name of the PhysicalDimension that should use this unit.
        \param unitName - The unit to print after velocities of that PhysicalDimension type.
        */
        void registerUnit(const char* dimensionName, const char* unitName);

        virtual void last(const agx::TimeStamp&) override;

    private:
        agxPowerLine::PowerLineObsPtr m_powerline;
        agxPowerLine::DotGraphWriterRef m_writer;
    };
}
