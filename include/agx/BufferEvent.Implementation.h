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

//-------------------------------------------------
// AUTOMATICALLY GENERATED EVENT, DO NOT EDIT!   
//-----------------------------------------------

#ifndef GENERATED_AGX_BUFFER_H_IMPLEMENTATION
#define GENERATED_AGX_BUFFER_H_IMPLEMENTATION

#include <agx/Integer.h>
#include <agx/IndexRange.h>
#include <agxData/Array.h>

AGX_FORCE_INLINE agx::Object* agxData::Buffer::EventListener::getOwner() { return m_owner; }
template <typename T>
AGX_FORCE_INLINE T* agxData::Buffer::EventListener::getOwner() { agxAssert(!m_owner || m_owner->is<T>()); return m_owner ? m_owner->as<T>() : nullptr; }

#endif // GENERATED_AGX_BUFFER_H_IMPLEMENTATION
