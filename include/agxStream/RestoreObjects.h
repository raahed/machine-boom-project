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


#include <agxStream/InputArchive.h>

namespace agxStream
{
  template<typename T>
  class RestoreObjects
  {
    public:
      typedef T ObjectType;
      typedef T* ObjectPtrType;
      typedef agx::Vector<T*> ObjectPtrVectorType;

      RestoreObjects(agxStream::InputArchive& archive, const char* collectionTitle, const char* elementTitle) :
          m_archive(archive),
          m_collectionTitle(collectionTitle),
          m_elementTitle(elementTitle)
      {
        m_archive.beginSection(m_collectionTitle);
        size_t size;
        m_archive.getAttribute("size", size);
        m_objects.resize(size);
        for(size_t i = 0; i < size; i++)
        {
          m_archive >> agxStream::in(m_elementTitle, m_objects[i]);
          if (m_archive.eof())
          {
            std::stringstream str;
            str << "Reached EOF while restoring " << m_collectionTitle << ". " <<
                   "There should be " << size << " RigidBodies, reached eof after restoring " << i;
            agxThrow agxStream::ArchiveException(str.str());
          }
        }
        m_archive.endSection(m_collectionTitle);
      }


      typename ObjectPtrVectorType::iterator begin()
      {
        return m_objects.begin();
      }

      typename ObjectPtrVectorType::iterator end()
      {
        return m_objects.end();
      }

      const agx::Vector<T*>& getObjects()
      {
        return m_objects;
      }

    protected:
      ObjectPtrVectorType m_objects;

      agxStream::InputArchive& m_archive;
      const char* m_collectionTitle;
      const char*m_elementTitle;


    private:
      RestoreObjects& operator =(const RestoreObjects&)
      {
        return *this;
      }
  };
}

