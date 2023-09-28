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


#include <agxStream/OutputArchive.h>

namespace agxStream
{
  template<typename T>
  class StoreObjects
  {
    public:
      StoreObjects(agxStream::OutputArchive& archive, const char* collectionTitle, const char* elementTitle ) :
          m_archive(archive),
          m_collectionTitle(collectionTitle),
          m_elementTitle(elementTitle)
      {
      }

      void push_back(const T* obj, bool allowNullptr)
      {
        if (obj != nullptr && obj->getEnableSerialization())
          m_objects.push_back(obj);
        else if (obj == nullptr && allowNullptr)
          m_objects.push_back(nullptr);
      }

      ~StoreObjects()
      {
        m_archive.beginSection(m_collectionTitle);
        m_archive.addAttribute("size", m_objects.size());
        for(size_t i = 0; i < m_objects.size(); i++)
          m_archive << agxStream::out(m_elementTitle, m_objects[i]);
        m_archive.endSection(m_collectionTitle);
      }

      agx::Vector<const T*> m_objects;

      agxStream::OutputArchive& m_archive;
      const char* m_collectionTitle, *m_elementTitle;

    private:
      StoreObjects& operator =(const StoreObjects&)
      {
        return *this;
      }
  };
}

