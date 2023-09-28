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



#include <agx/agxPhysics_export.h>

#include <agxStream/agxStream.h>
#include <fstream>

namespace agxStream
{

  /**
  Write a Serializable object to a file.
  The type of file (binary/ascii) is determined from the file type.

  \param filename - The path to the file where data will be param.
  \param data - Pointer to the serializable data that will be stored.
  \return true if writing to disk was successful.
  */
  bool AGXPHYSICS_EXPORT write( const std::string& filename, const agxStream::Serializable* data );

  /**
  Write a Serializable object to a file.
  The type of file (binary/ascii) is determined from the file type.

  \param filename - The path to the file where data will be written.
  \param writeObjects - Vector containing pointers to serializable objects that will be stored.
  \return true if writing to disk was successful.
  */
  bool AGXPHYSICS_EXPORT write( const std::string& filename, const agxStream::SerializablePtrVector& writeObjects );

  /**
  Open and read serializable objects from a file on disk with previously stored objects.
  The type of file (binary/ascii) is determined from the file type.

  The vector \p readObjects contain pointers to restored objects, and it is the caller of this functions
  responsibility to handle the allocated memory.



  \param filename - The exact path to the file where data will be written, it will not be searched for using agxIO::Environment.
  \param readObjects - Vector containing pointers to restored objects.
  \return true if no error occurred during restoration.
  */
  bool AGXPHYSICS_EXPORT read( const std::string& filename, agxStream::SerializablePtrVector& readObjects );

}

