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

#ifndef AGXHYDRAULICS_PIPE_H
#define AGXHYDRAULICS_PIPE_H


#include <agxHydraulics/FlowUnit.h>

namespace agxHydraulics
{

  /**
  A pipe carries fluid flow through the hydraulics system. It is a power line
  Unit, so it can be connected to other Units, including other pipes, using
  Connectors.
  */
  class AGXHYDRAULICS_EXPORT Pipe : public agxHydraulics::FlowUnit
  {
    public:
      /**
      Create a new pipe.
      \param length The length of the pipe.
      \param area The area of the pipe.
      \param density The density of the fluid passing through the pipe.
      */
      Pipe(agx::Real length, agx::Real area, agx::Real density);

      Pipe(agx::Real length, agx::Real area, agx::Real density, agx::UInt8 elementIndex, agx::RigidBody* body);

    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:

      DOXYGEN_START_INTERNAL_BLOCK()
#ifndef SWIG
      /**
      Called by the power line when all components have been stored if the
      power line was inititalized when stored.
      */
      virtual bool postStore(agxStream::StorageStream& str) const override;

      /**
      Called by the power line when all components have been restored and
      the power line initialized. Is not called if the power line was not
      initialized when stored.
      */
      virtual bool postRestore(agxStream::StorageStream& str) override;

      /**
      Called by the power line during store. Stores the attached PipeAttachements
      to the given StorageStream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Called by the power line during restore.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::Pipe);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      Pipe();
      virtual ~Pipe();
  };

  AGX_DECLARE_POINTER_TYPES(Pipe);
  AGX_DECLARE_VECTOR_TYPES(Pipe);
}

#endif
