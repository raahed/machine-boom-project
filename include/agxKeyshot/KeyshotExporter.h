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

#ifndef AGXKEYSHOT_EXPORTER_H
#define AGXKEYSHOT_EXPORTER_H


#include <agx/config.h>
#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/String.h>
#include <agxSDK/Simulation.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide
{
  class RenderMaterial;
}

namespace agxKeyshot {

  /**
  */
  class AGXPHYSICS_EXPORT KeyshotExporter : public agx::Referenced
  {
  public:
    /**
    Constructor for a KeyshotExporter.
    \param binaryMode - if true, the vertex,normal,tex data will be binary to reduce size.
    */
    KeyshotExporter( bool binaryMode = true );

    bool open( const agx::String& path );

    /**
    Connect exporter to a journal and read transform data. Default behavior is to loop over the journal session frames
    and extract transformation data for the bodies in the simulation which is stored in the exporter. Can give optional
    argument for extracting data from a single session frame, instead of the whole journal.
    \param journalPath - Path to the Journal
    \param sessionName - Name of the specified session in the journal.
    \param singleFrameIndex - if set to >= 0, only extract data from the frame with the specified Index. Otherwise loop the journal normally.
    */
    bool connectToJournal(const agx::String& journalPath, const agx::String& sessionName, int singleFrameIndex=-1);

    size_t getNumParticles() const;
    agx::ParticleSystem * getParticleSystem();

    size_t getNumBodies() const;
    agx::RigidBody* getBody( size_t i );

    /**
    Specify whether the animation should be in slow motion (multiplier > 1)
    or speedup (<1). So having 4 seconds of data in 100Hz, setting multiplier to 2, will
    increase the actual time to 8, resulting in half speed in playback.
    1.0 - Original speed.
    2 - Half speed.
    0.5 - double speed.
    \param multiplier - Specifies the speedup/slowdown of the data.
    */
    void setTimeMultiplier( agx::Real multiplier = 1.0 );
    agx::Real getTimeMultiplier(  ) const;

    /**
    Set camera specifications. Currently ignored.
    */
    void setCamera( const agx::Vec3& position, const agx::Vec3& lookAt, const agx::Vec3& upDirection, float fov  );


    /**
    Set the frame rate for the keyshot rendering
    */
    void setFps( agx::UInt fps );

    /**
    \return current FPS fpr the keyshot rendering
    */
    agx::UInt getFps( ) const;

    /**
    Set the size of the renderable window
    \param width - Width of the window in pixels
    \param height - Height of the window in pixels
    */
    void setImage( unsigned int width, unsigned int height);


    /**
    Add a mesh to the fbx file from object path, the mesh can then be animated using its id
    \param id - The node id to be used when animating the mesh
    \param renderMaterial - render material specification
    \return true if successful
    */
    bool addMesh( agx::RigidBody* body, const agx::Vec3Vector& vertices,
      const agx::UInt32Vector& indices, const agx::Vec3Vector& normals,
      const agx::Vec2Vector& texCoordinates, const agx::AffineMatrix4x4& relative_transform,
      agxCollide::RenderMaterial *renderMaterial);

    bool addSpheres( const agx::Vec3Vector& centers,
      const agx::RealVector& radiuses,
      const agx::Vec3Vector& colors);

    bool writeParticleShadingMaterial();

    virtual ~KeyshotExporter();

    static agxCollide::RenderMaterial *parseKeyshotMaterial( const agx::String& material );


  protected:

    bool writeHeader();
    bool writeCamera();


    agx::observer_ptr<agxSDK::StepEventListener> m_listener;

    class TransformVector : public agx::Referenced {
    public:
      void push_back( const agx::AffineMatrix4x4& transform ) { m_vector.push_back(transform); }
      const agx::AffineMatrix4x4& operator()(size_t i) { return m_vector[i]; }
      void clear() { m_vector.clear(); }
      size_t size() const { return m_vector.size(); }
    protected:
       virtual ~TransformVector() {}

      agx::Vector<agx::AffineMatrix4x4> m_vector;
    };

    typedef agx::HashTable<agx::UInt, agx::ref_ptr<TransformVector> > BodyTransformations;
    BodyTransformations m_bodyTransforms;

  private:


    agx::ref_ptr<agx::Referenced> m_writer;
    agxSDK::SimulationRef m_simulation;
    int m_numFrames;
    agx::Real m_duration;
    agx::UInt m_fps;

    struct CameraSettings
    {
      CameraSettings() : position(0,10,10), direction(0,-1,-1), up(0,1,0), fov(45), valid(false) {}

      agx::Vec3 position;
      agx::Vec3 direction;
      agx::Vec3 up;
      float fov;
      bool valid;
    };

    CameraSettings m_camera;
    agx::Vec2u m_imageSize;

    agxCollide::BoundingAABB m_bounds;
    bool m_binaryMode;
    agx::Real m_timeMultiplier;

    FILE *m_file;

    typedef agx::HashTable<agx::UInt32, agxCollide::RenderMaterialRef> RenderMaterialTable;
    RenderMaterialTable m_renderMaterials;

  };

  typedef agx::ref_ptr<KeyshotExporter> KeyshotExporterRef;

}
DOXYGEN_END_INTERNAL_BLOCK()
#endif
