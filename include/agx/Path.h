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

#ifndef AGX_PATH_H
#define AGX_PATH_H

#include <agx/Name.h>
#include <agx/HashFunction.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agx
{
  /**
  Representation of a path, a list of name components.
  */
  class AGXCORE_EXPORT Path
  {
  public:
    Path();
    Path(const agx::String& path);
    Path(const std::string& path);
    Path(const char *path);
    // Path(const char *begin, const char *end);
    Path(const agx::Path& other);
    explicit Path(const Name& name);

    ~Path();

    Path& operator= (const Path& other);


    /**
    \return A string representation of the path.
    */
    const agx::String& str() const;

    /**
    \return A c-string representation of the path.
    */
    const char *c_str() const;

    /**
    \return The root name component.
    */
    agx::String root() const;

    /**
    \return True if the path is empty.
    */
    bool empty() const;

    /**
    \return True if the path has sub-paths.
    */
    bool hasTail() const;

    /**
    \return The subpath below the root, eg Foo.bar.hej -> bar.hej
    */
    agx::Path tail() const;

    /**
    \return The base of the path, eg Foo.bar.hej -> Foo.bar
    */
    agx::Path base() const;

    /**
    \return The last component of the path, eg Foo.bar.hej -> hej
    */
    agx::String back() const;


    bool contains(const Path& subpath);
    bool contains(const Name& component);

    // Hash function
    UInt32 hash() const;

    agx::Path operator+ (const agx::Path& extension) const;
    agx::Path operator+ (const agx::Name& extension) const;

    agx::Path& operator+= (const agx::Path& extension);
    agx::Path& operator+= (const agx::Name& extension);

    bool operator== (const agx::Path& other) const;
    bool operator!= (const agx::Path& other) const;


  public:
    class AGXCORE_EXPORT iterator
    {
    public:
      iterator();

      String operator *() const;

      iterator& operator++();
      iterator operator++(int);

      bool operator== (const iterator& other) const;
      bool operator!= (const iterator& other) const;

    private:
      friend class Path;
      iterator(const Path& path, size_t pos = 0);

    private:
      String m_path;
      size_t m_pos;
    };


    iterator begin() const;
    iterator end() const;

  private:
    void init(const char *begin, const char *end);
    void init(const std::string& str);

  private:
    String m_str;
  };


  AGXCORE_EXPORT std::ostream& operator << ( std::ostream& stream, const Path& path );

  /* Implementation */
  inline Path::Path()
  {}

  AGX_FORCE_INLINE const String& Path::str() const { return m_str; }

  inline Path::iterator Path::begin() const { return iterator(*this); }
  inline Path::iterator Path::end() const { return iterator(*this, String::npos); }

  AGX_FORCE_INLINE UInt32 Path::hash() const
  {
    return agx::hash(m_str);
  }

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* AGX_PATH_H */
