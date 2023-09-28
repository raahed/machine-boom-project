/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXOSG_PRESSURE_ATLAS
#define AGXOSG_PRESSURE_ATLAS

#include <agx/Referenced.h>

#include <agxCollide/Trimesh.h>

#include <agxOSG/export.h>
#include <agxOSG/PressureGenerator.h>
#include <agxOSG/PressureToColorConverter.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Geometry>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  class GeometryNode;
  AGX_DECLARE_POINTER_TYPES(PressureAtlas);


  /**
   * A pressure map for a single trimesh. Holds both the AGX trimesh used for
   * simulation and a corresponding OSG geometry used for rendering, as well as
   * a 32-bit floating-point image for the pressure data itself.
   */
  class AGXOSG_EXPORT PressureAtlas : public agx::Referenced
  {
  public:
    PressureAtlas(const agxCollide::Trimesh* mesh, agx::Real texelsPerMeter);

    /// \todo Make  vvv this vvv  private and friend PressureGenerator..

    /**
     * Let this PressureAtlas take over overship of the given pressure
     * generator. A generator can only be registered on one atlas at the time
     * and will be rejected (false returned) if it was already registered to
     * another atlas.
     *
     * \return True if the generator was registered. False otherwise.
     */
    bool registerPressureGenerator(agxOSG::PressureGenerator* generator);


    /**
     * Add a pressure sphere centered at 'point', given in the mesh's
     * coordinate system, The given 'magnitude' is the pressure at the given
     * 'point'. The added pressure will decrease linearly with distance from
     * 'point' to reaches zero for points at or beyond 'radius' distance from
     * 'point'.
     *
     * The point should be on the triangle with the given triangle index.
     *
     * Will mark the atlas as dirty and pressurized if the magnitude is non-zero.
     *
     * @param point The point, in mesh coordinates, where the pressure originates.
     * @param triangleIndex The index of the triangle on which the point is located.
     * @param magnitude The pressure at the point, given in Pa. Must be non-negative.
     * @param radius The radius of the sphere which should be given a pressure contribution. Must be positive.
     */
    void addPressure(const agx::Vec3& point, agx::UInt32 triangleIndex, agx::Real magnitude, agx::Real radius);

    /**
     * Fill the pressure map with zeros. Set the dirty flag and clear the
     * pressurized flag.
     */
    void clearPressure();


    /**
     * Check if the pressure map is dirty. It is dirty if 'clearPressure' or
     * 'addPressure' has been called with a non-zero magnitude since the last
     * call to 'updateColorImages'.
     *
     * @return True if the pressure map is dirty. False otherwise.
     */
    bool isDirty() const;


    /**
     * Check if the pressure map is pressurized. It is pressurized if 'addPressure' has
     * been called with a non-zero pressure since the last call to 'clearPressure'.
     *
     * @return True if the pressure map has at elast one non-zero element. False otherwise.
     */
    bool isPressurized() const;


    /// \todo Make  vvv this vvv  private and friend PressureToColorConverter.


    /**
     * Make the given color converter perform color conversion on this atlas'
     * pressure map when PressureAtlas::updateColorImages is called.
     *
     * Will hold a reference to the converter.
     *
     * @param colorConverter Color converter that should convert this atlas' pressure map.
     * @return True if the color converter was registered. False if it was already registered.
     */
    bool registerColorConverter(agxOSG::PressureToColorConverter* colorConverter);


    /**
     * Cause the given color converter to no longer receive update calls from
     * PressureAtlas::updateColorImages.
     *
     * Will release a reference to the color converter, so beware that the
     * given pointer may become dangling.
     *
     * @param colorConverter The color converter to unregister.
     * @return True if the color converter was unregistered. False if it wasn't registered.
     */
    bool unregisterColorConverter(agxOSG::PressureToColorConverter* colorConverter);


    /**
     * Trigger a color conversion on all converters, if the pressure map is dirty.
     */
    void updateColorImages();

    /**
     * Returns the pressure map. It is a 2D image containing the pressure on
     * the mesh mapped onto the mesh according to the texture coordinate found
     * in the osg::Geometry returned by getGraphicsMesh(). The image format is
     * one 32-bit floating point channel per texel, i.e., GL_RED, GL_FLOAT.
     */
    osg::Image* getPressureMap();

    const agxCollide::Trimesh* getPhysicsMesh();

    /**
     * Returns an OSG Geometry that can be used to render the collected
     * pressure. Contains texture coordinates into the pressure map.
     */
    osg::Geometry* getGraphicsMesh();

    /**
     * @return The texture coordinates for the vertices of the graphics mesh.
     */
    osg::Vec2Array& getTextureCoordinates();


    agxOSG::GeometryNode* createAtlasRendering();


    /** @return The number of triangles of the mesh. */
    size_t getNumTriangles() const;

    /**
     * @return The number of vertices in the graphics mesh. This is always
     * 3*getNumTriangles().
     */
     size_t getNumVertices() const;

     size_t getVertexIndex(size_t triangleIndex, size_t localVertexIndex) const;

  protected:
    virtual ~PressureAtlas() {}

  private:
    void createGraphicsMesh();
    osg::Geometry* createOsgMesh(const agxCollide::Trimesh* mesh);
    void createAtlas();
    // Hiding assignment
    PressureAtlas& operator=(const PressureAtlas&) {return *this;}


    agx::UInt32 nextTriangle();
    void queuePartnersInRange(agxCollide::Mesh::Triangle& triangle, const agx::Vec3& point, agx::Real radiusSquared);
    bool isInRange(const agx::Vec3& point, const agxCollide::Mesh::Triangle& triangle, agx::Real radiusSquared);
    bool colorTriangle(const agxCollide::Mesh::Triangle& triangle, const agx::Vec3& point,
                       const agx::Real radius, const agx::Real radiusSquared,
                       const agx::Real centerMagnitude);


  private:

    /** The AGX mesh on which we want to measure pressure. */
    const agxCollide::TrimeshConstRef m_physicsMesh;

    /** Triangle soup used for rendering. Carries texture coordinates into the atlas. */
    osg::ref_ptr<osg::Geometry> m_graphicsMesh;

    /** The resolution of the pressure atlas. */
    agx::Real m_texelsPerMeter;

    /**
     * The pressure itself. Holds texture data for all triangles. One 32-bit
     * floating-point channel per texel. Sized according to the total area of
     * the triangles and the specified resolution.
     */
    osg::ref_ptr<osg::Image> m_pressureMap;

    /**
     * The PressureGenerators that have been registered to supply pressure on
     * this atlas.
     */
     PressureGeneratorRefVector m_pressureGenerators;

     /**
      * The ColorGenerators that have been registered and will be notified when
      * 'updateColorImages' is called.
      */
     PressureToColorConverterRefVector m_colorGenerators;


     /** Used during trimesh traversal to avoid loops. Contains visited and queued ones.*/
     agx::HashSet<agx::UInt32> m_consideredTriangles;

     /** Used during trimesh traversal to keep track where we need to go. */
     agx::Vector<agx::UInt32> m_queuedTriangles;

    /**
     * Set when changes are made to the pressure map. Cleared when the color
     * generators are run.
     */
     bool m_dirty;

     /**
      * Set when 'addPressure' is called with a non-zero magnitude. Cleared when
      * 'clearPressure' is called.
      */
     bool m_pressurized;
  };

}

#endif

