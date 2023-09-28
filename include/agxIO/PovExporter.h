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

#ifndef AGXIO_POV_EXPORTER_H
#define AGXIO_POV_EXPORTER_H

#include <agx/config.h>
#include <agx/agxPhysics_export.h>
#include <stdexcept>
#include <agx/AffineMatrix4x4.h>
#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>
#include <agx/Constraint.h>
#include <agxSDK/Assembly.h>
#include <sstream>
namespace agxSDK
{
  class Simulation;
}

namespace agx {
  class PropertyContainer;
  class Material;
  class ContactMaterial;
}

namespace agxCollide {
  class Shape;
}

namespace agxIO
{
  /**
  Class for exporting a simulation into AGX .pov format.
  */
  class AGXPHYSICS_EXPORT PovExporter
  {
  public:
    PovExporter();

    /**
    Write a simulation into the \p Pov-ray file format for rendering.
    \param filename - Path to the file which will be created
    \param simulation - The simulation that will be written to the scene file
    \param from - Camera from (eye) position
    \param at - Camera looking at position
    \param up - Camera up vector
    \param fovy - field of view value
    \param aspectRatio - aspect between the height/width of the window
    \return true if exporting to pov-ray file was successful.
    */
    bool write( const agx::String& filename, agxSDK::Simulation *simulation,
                agx::Vec3 from, agx::Vec3 at,
                agx::Vec3 up, double fovy, double aspectRatio );

    /**
    Write a simulation in .pov format to a stream
    \param outStream - Stream into which the data will be written
    \param simulation - The simulation that will be written to the pov-ray file
    \param from - Camera from (eye) position
    \param at - Camera looking at position
    \param up - Camera up vector
    \param fovy - field of view value
    \param aspectRatio - aspect between the height/width of the window
    \return true if exporting to pov-ray file was successful.
    */
    bool write( std::ostream& outStream, agxSDK::Simulation *simulation,
                agx::Vec3 from, agx::Vec3 at,
                agx::Vec3 up, double fovy, double aspectRatio );

    ~PovExporter();

    static void setFocalSettings( agx::Real distance, agx::Real aperture, int blur_samples, agx::Real variance );
    static void setEnableFocalBlur( bool enable );

  protected:

    /// Translate any non-alphanumeric letters into '_'
    agx::String cleanupString( const agx::String& in )
    {
      agx::String out;
      out.resize(in.size());

      for(size_t i=0; i < in.size(); i++)
      {
        if (isalnum(static_cast<unsigned char>(in[i])))
          out[i] = in[i];
        else
          out[i] = '_';
      }
      return "pov_"+out;
    }


    bool exportRigidBodies();
//    bool exportConstraints();
    bool exportAssemblies();
    bool exportAssembly( agxSDK::Assembly *assembly );
    bool exportGeometries( const agx::String& parentMaterial );
    bool exportShape( const agx::AffineMatrix4x4& t, const agxCollide::Shape *shape, const agx::String& parentMaterial );
    bool exportGeometry( agxCollide::Geometry *geometry, const agx::String& parentMaterial);
    bool exportRigidBody( agx::RigidBody *body);

    typedef agx::HashSet<agx::RigidBody *> RigidBodyPtrHashSet;
    typedef agx::HashSet<agxCollide::Geometry *> GeometryPtrHashSet ;
    typedef agx::HashSet<agxSDK::Assembly *> AssemblyPtrHashSet ;

    GeometryPtrHashSet m_geometryHasMaterial;
#ifdef GET_NAME
#undef GET_NAME
#endif
#define GET_NAME(TYPE, NAME) \
    typedef agx::HashTable<const TYPE *, agx::String> NameTable ##NAME;  \
    typedef agx::HashSet<agx::String> UniqueNameTable ##NAME;  \
    NameTable ##NAME  m_exportedName ##NAME;   \
    UniqueNameTable ##NAME  m_uniqueNames ##NAME;   \
     \
    inline agx::String getName( const TYPE *item )   \
    {   \
      agx::String name = item->getName();   \
      NameTable ##NAME::const_iterator nameIt = m_exportedName ##NAME.find(item);  \
      if (nameIt != m_exportedName ##NAME.end()) \
        return cleanupString(nameIt->second);    /* use the associated name */      \
      if (name.empty()) { \
        std::stringstream str;   \
        str << #NAME << "_" << item->getIndex();  \
        name = str.str();   \
        m_exportedName ##NAME.insert(item, name);  \
        m_uniqueNames ##NAME.insert(name);  \
        return cleanupString(name);  \
      }   \
      else {   /* Not empty, but is it unique? */ \
        UniqueNameTable ##NAME::const_iterator it = m_uniqueNames ##NAME.find(name);  \
        if (it == m_uniqueNames ##NAME.end() )  \
        {  \
          m_uniqueNames ##NAME.insert(name);  \
          m_exportedName ##NAME.insert(item, name);  \
          return cleanupString(name);  \
        }  \
         \
        std::stringstream str;   \
        str << name << "_" << item->getIndex();  /* Create a new unique name and use that for this item */ \
        name = str.str();   \
        m_exportedName ##NAME.insert(item, name);  \
        m_uniqueNames ##NAME.insert(name);  \
      }  \
      return cleanupString(name);   \
    }

    GET_NAME(agx::RigidBody, Bodies)
    GET_NAME(agxCollide::Geometry, Geometries)
    GET_NAME(agxSDK::Assembly, Assemblies)

    #undef GET_NAME

    AssemblyPtrHashSet m_exportedAssemblies;
    RigidBodyPtrHashSet m_exportedBodies;
    GeometryPtrHashSet m_exportedGeometries;

    agxSDK::Simulation *m_simulation;

    std::ostream *m_outStream;

    bool m_declareStarted;
    static agx::Real m_focalDistance;
    static agx::Real m_aperture;
    static agx::Real m_variance;
    static int m_blurSamples;
    static bool m_focalBlurEnable;
  };

}
#endif
