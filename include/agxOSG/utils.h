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

#ifndef AGXOSG_UTILS_H
#define AGXOSG_UTILS_H

#include <agxOSG/export.h>
#include <agx/Vec4.h>

#include <agxUtil/agxUtil.h>
#include <agxUtil/Statistic.h>
#include <agxOSG/Node.h>
#include <agxOSG/ParticleSystemDrawable.h>
#include <agxOSG/RigidBodySystemDrawable.h>
#include <agxOSG/RigidBodyRenderCache.h>
#include <agxOSG/ScalarColorMap.h>
#include <agxSDK/GuiEventListener.h>
#include <agxRender/Color.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/observer_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agx
{
  class RigidBody;
  class ParticleSystem;

  class RigidBodyEmitter;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxCollide
{
  class Geometry;
}

namespace agxUtil
{
  class ExponentialMovingAverageStatistic;
}

namespace agxModel
{
  class FractureGenerator;
}

DOXYGEN_START_INTERNAL_BLOCK()


namespace osg {
  class Geometry;
  class Node;
  class Texture2D;
}

namespace osgSim
{
  class ColorRange;
}

namespace osgText
{
  class Text;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxOSG
{
  class GeometryNode;
  class FrameTrajectoryDrawable;

  enum TextureMode
  {
    DIFFUSE_TEXTURE=0,
    SHADOW_TEXTURE=1,
    SHADOW_JITTER_TEXTURE=2,
    NORMALMAP_TEXTURE=3,
    HEIGHTMAP_TEXTURE=4,
    SPECULARMAP_TEXTURE = 5,
    SPHEREMAP_TEXTURE = 6
  };

  /**
  Finds the geometry node associated to a geometry.
  \return the geometry node if the geometry has one, otherwise false
  */
  AGXOSG_EXPORT GeometryNode* findGeometryNode( const agxCollide::Geometry* geometry, osg::Group* rootNode );

  /**
  Assigns a Cubemap texture to the specified node.
  \param node - Node to which the texture will be assigned
  \param imagePath - Path to the initial part of the 6 files that build up the cubemap:

  [imagePath]/negx.[imageType]
  [imagePath]/posx.[imageType]
  [imagePath]/negy.[imageType]
  [imagePath]/posy.[imageType]
  [imagePath]/negz.[imageType]
  [imagePath]/posz.[imageType]

  \param imageType - The file type of the cubemap textures. default is ".dds"
  \returns true if files are found and the node has a cube map assigned
  */
  AGXOSG_EXPORT bool setCubeMapTexture(osg::Node *node, const std::string& imagePath, const std::string& imageType = ".dds");

  /**
  Specifies the amount of reflection that should be used (if CubeMap or SphereMap is assigned to the node).
  0 - No reflection
  1 - Only reflection
  */
  AGXOSG_EXPORT void setEnvMapReflectionCoefficient(osg::Node* node, float coeff);

  /**
  Read an image from disk and apply to the specified geometry as a 2D Texture in OVERRIDE|PROTECTED mode.
  \note For this to work, geometry has to have an agxOSG::GeometryNode in the root. createVisual( geometry, root )

  \param geometry - geometry to search for
  \param rootNode - Parent node in which the GeometryNode associated to geometry should be found
  \param imagePath - Path to the image to be used as a texture
  \param repeat - If true the texture will use WRAP mode REPEAT
  \param mode - Texture mode
  \return false if image cannot be read.
  */
  AGXOSG_EXPORT bool setTexture(const agxCollide::Geometry* geometry, osg::Group* rootNode, const std::string& imagePath, bool repeat = false, TextureMode mode = DIFFUSE_TEXTURE);

  /**
  For all geometries in the rigid body, this function will set a 2D texture in OVERRIDE|PROTECTED mode.
  \note For this to work, the geometries have to have an agxOSG::GeometryNode in the root. createVisual( rigidBody, root )

  \param rigidBody - body to search for
  \param rootNode - Parent node in which the GeometryNode associated to geometry should be found
  \param imagePath - Path to the image to be used as a texture
  \param repeat - If true the texture will use WRAP mode REPEAT
  \param mode - Texture mode
  \return false if image cannot be read.
  */
  AGXOSG_EXPORT bool setTexture(
    const agx::RigidBody* rigidBody, osg::Group* rootNode, const std::string& imagePath,
    bool repeat = false, TextureMode mode = DIFFUSE_TEXTURE);

  /**
  Read an image from disk and apply to the specified node as a 2D Texture in OVERRIDE|PROTECTED mode.
  \param node - Node to attach texture to
  \param imagePath - Path to the image to be used as a texture
  \param repeat - If true the texture will use WRAP mode REPEAT
  \param mode - Texture mode
  \param scaleU - Scaling of texture in U
  \param scaleV - Scaling of texture in V
  \return false if image cannot be read.
  */
  AGXOSG_EXPORT bool setTexture(
    osg::Node* node, const std::string& imagePath, bool repeat = false, TextureMode mode = DIFFUSE_TEXTURE,
    float scaleU = 1.0f, float scaleV = 1.0f);


  /**
  Create a Texture2D object that can be assigned to nodes
  \param filename - Path to the image file
  */
  AGXOSG_EXPORT osg::Texture2D* createTexture(const std::string& filename);


  /**
  Apply a 2D Texture to the specified node in OVERRIDE|PROTECTED mode.
  \param node - Node to attach texture to
  \param texture - The texture to apply
  \param repeat - If true the texture will use WRAP mode REPEAT
  \param mode - Texture mode
  \param scaleU - Scaling of texture in U
  \param scaleV - Scaling of texture in V
  \return false if image cannot be read.
  */
  AGXOSG_EXPORT bool setTexture(
    osg::Node *node, osg::Texture2D *texture, bool repeat = false, TextureMode mode = DIFFUSE_TEXTURE,
    float scaleU = 1.0f, float scaleV = 1.0f);

  /**
  Apply a 2D Texture to the specified node in OVERRIDE|PROTECTED mode.
  \param node - Node to attach texture to
  \param texture - The texture to apply
  \param repeat - If true the texture will use WRAP mode REPEAT
  \param mode - Texture mode
  \param scaleU - Scaling of texture in U
  \param scaleV - Scaling of texture in V
  \return false if image cannot be read.
  */
  AGXOSG_EXPORT bool setTexture(osg::Node *node, agxOSG::Texture2D *texture, bool repeat = false, TextureMode mode = DIFFUSE_TEXTURE, float scaleU = 1.0f, float scaleV = 1.0f);


  /**
  Apply a 2D Texture to the specified node in OVERRIDE|PROTECTED mode.
  \param node - Node to attach texture to
  \param texture - The texture to apply
  \param repeat - If true the texture will use WRAP mode REPEAT
  \param mode - Texture mode
  \param scaleU - Scaling of texture in U
  \param scaleV - Scaling of texture in V
  \return false if image cannot be read.
  */
  AGXOSG_EXPORT bool setTexture(agxOSG::GeometryNode *node, agxOSG::Texture2D *texture, bool repeat = false, TextureMode mode = DIFFUSE_TEXTURE, float scaleU = 1.0f, float scaleV = 1.0f);

  template <typename T>
  void setUniform( osg::Node *node, const std::string& name, T value)
  {
    if ( node == nullptr )
      return;

    node->getOrCreateStateSet()->addUniform( new osg::Uniform(name.c_str(), value));
  }

  /**
  Add text relative to an geometry. The text object will follow the geometry if it moves.
  \note The Geometry must have a visual representation created before this call.
  \param geometry - subject geometry
  \param root - root node
  \param text - the text to render
  \param relPosition - relative position to the geometry
  \param color - color of the text
  \param size - size of the text
  \return false if the geometry does not have a visual representation in the root parent
  */
  AGXOSG_EXPORT bool addText(const agxCollide::Geometry* geometry,
                             osg::Group* root,
                             const std::string& text,
                             const agx::Vec3& relPosition = agx::Vec3(0, 0, 0),
                             const agx::Vec4f& color = agxRender::Color::Yellow(),
                             agx::Real size=0.2);

  /**
  Create a text object
  \param text - the text to render
  \param relPosition - relative position to the geometry
  \param color - color of the text
  \param size - size of the text
  \return The text node
  */
  AGXOSG_EXPORT osg::Geode* createText(const std::string& text,
                                       const agx::Vec3& relPosition = agx::Vec3(0,0,0),
                                       const agx::Vec4f& color = agxRender::Color::Yellow(),
                                       agx::Real size = 0.2);


  /**
  Remove text from a geometry. If several text objects have been added to the geometry,index will find you the i'th
  text object added.
  \param geometry - subject geometry
  \param root - root node
  \param index - index of the text object to remove
  \return true if text object found, false otherwise
  */
  AGXOSG_EXPORT bool removeText(const agxCollide::Geometry* geometry, osg::Group * root, size_t index = 0);

  /**
  Finds a text object already added to the geometry (so the text can be changed). If several text objects have been added
  to the geometry, index will find you the i'th text object added.
  \param geometry - geometry with agxOSG::GeometryNode and text
  \param root - root node
  \param index - index of the text object to find (default 0)
  \return text object if present - otherwise false
  */
  AGXOSG_EXPORT osgText::Text* findText( const agxCollide::Geometry* geometry, osg::Group* root, size_t index = 0 );

  /**
  Create a uniform grid.
  This function might be a little hard to use, since it uses osg-primitives.
  There is an easier to use variant of this function.
  \param resU The resolution of the grid (in number of lines) in x.
  \param resV The resolution of the grid (in number of lines) in y.
  \param origin The origin of the grid in world coordinates
  \param endPointX The end point of the grid in x coordinates.
  The distance between this point and the origin gives the length of the grid in x.
  \param endPointY The end point of the grid in y coordinates.
  The distance between this point and the origin gives the length of the grid in y.
  \param wireFrameMode unused
  \retval the osg::Geometry* containing the grid.
  */
  AGXOSG_EXPORT osg::Geometry* createGrid(
    unsigned int resU,
    unsigned int resV,
    const osg::Vec3& origin,
    const osg::Vec3& endPointX,
    const osg::Vec3& endPointY,
    bool wireFrameMode = false);

  /**
  Create a uniform grid and adds it to a root.
  Check return value to see if everything was successful.
  \param resU The resolution of the grid (in number of lines) in x.
  \param resV The resolution of the grid (in number of lines) in y.
  \param origin The origin of the grid in world coordinates
  \param endPointX The end point of the grid in x coordinates.
  The distance between this point and the origin gives the length of the grid in x.
  \param endPointY The end point of the grid in y coordinates.
  The distance between this point and the origin gives the length of the grid in y.
  \param root The root that the grid should be added to. Non-zero.
  \retval The geode which is part of root now.
  */
  AGXOSG_EXPORT osg::Geode* createGrid(
    unsigned int resU,
    unsigned int resV,
    const agx::Vec3& origin,
    const agx::Vec3& endPointX,
    const agx::Vec3& endPointY,
    osg::Group* root);

  /**
  Creates osg axes relative a geometry given relative transform. If the relative transform is zero, the
  axes will be at model center of the geometry. If geometry = 0 the root node will be used as parent.
  */
  AGXOSG_EXPORT osg::MatrixTransform* createAxes(agxCollide::Geometry* geometry, agx::Frame* relativeTransform, osg::Group* root, float scale = 1, const agx::Vec4f& color = agx::Vec4f());
  AGXOSG_EXPORT osg::MatrixTransform* createAxes(agxCollide::Geometry* geometry, agx::AffineMatrix4x4 relativeTransform, osg::Group* root, float scale = 1, const agx::Vec4f& color = agx::Vec4f());

  /**
  Creates osg axes relative a rigid body given relative transform. If the relative transform is zero, the
  axes will be at model center of the rigid body. If rigid body = 0 the root node will be used as parent.
  */
  AGXOSG_EXPORT osg::MatrixTransform* createAxes(agx::RigidBody* rb, agx::Frame* relativeTransform, osg::Group* root, float scale = 1.f, const agx::Vec4f& color = agx::Vec4f());
  AGXOSG_EXPORT osg::MatrixTransform* createAxes(agx::RigidBody* rb, agx::AffineMatrix4x4 relativeTransform, osg::Group* root, float scale = 1.f, const agx::Vec4f& color = agx::Vec4f());


  AGXOSG_EXPORT osg::MatrixTransform* createAxes(agx::AffineMatrix4x4 relativeTransform, osg::Group* root, float scale = 1.f, const agx::Vec4f& color = agx::Vec4f());

  /**
  Create axes at the constraint attachments.
  */
  AGXOSG_EXPORT void createAxes( agx::Constraint* constraint, osg::Group* root, float scale = 1.f, const agx::Vec4f& color = agx::Vec4f() );

  /**
  Set the specular part of a material for a node. If the node  does not have a material, a new one
  will be created and assigned to the node.
  */
  AGXOSG_EXPORT void setSpecularColor( osg::Node * node, const agx::Vec4f& color );

  /**
  Set the shininess exponent for the Phong specular model
  */
  AGXOSG_EXPORT void setShininess( osg::Node *node, float shininess );

  /**
  Set the ambient part of a material for a node. If the node  does not have a material, a new one
  will be created and assigned to the node.
  */
  AGXOSG_EXPORT void setAmbientColor( osg::Node * node, const agx::Vec4f& color );

  /**
  Set the alpha part of the material for a node. 0 completely transparent, 1 - opaque
  */
  AGXOSG_EXPORT void setAlpha( osg::Node * node, float alpha );

  /**
  Set the diffuse part of a material for a node. If the node  does not have a material, a new one
  will be created and assigned to the node.
  */
  AGXOSG_EXPORT void setDiffuseColor( osg::Node * node, const agx::Vec4f& color );


  /**
  Set the diffuse part of a material for a rigid body (all geometries). Only geometries with geometry node in the scene
  graph will have their color changed.
  */
  AGXOSG_EXPORT void setDiffuseColor( agx::RigidBody* rb, const agx::Vec4f& color, osg::Group* root );

  /**
  Set the diffuse part of a material for a geometry. If the geometry doesn't have a geometry node (in the scene graph),
  this call will be ignored.
  */
  AGXOSG_EXPORT void setDiffuseColor( agxCollide::Geometry* geometry, const agx::Vec4f& color, osg::Group* root );

  /**
  Given a geometry, this function create visual representation of the shape.
  \param geometry . The geometry for which a visual node should be created
  \param root - The root node for the visual node (will be inserted)
  \param detailRatio - The tessellation count, the higher, the more detailed
  \param createAxes - If true, xyz axes are created at the center
  \param evenIfSensor - Create visual even if the geometry is a sensor
  */
  AGXOSG_EXPORT agxOSG::GeometryNode *createVisual( const agxCollide::Geometry* geometry, osg::Group* root, float detailRatio = 1.0f, bool createAxes = false, bool evenIfSensor=true );

  /**
  Given a particle system, create a visual representation.
  */
  AGXOSG_EXPORT osg::Geode *createVisual( agx::ParticleSystem* particleSystem, osg::Group* root, agxOSG::ParticleSystemDrawable::ParticleRenderingMode mode = DEFAULT_PARTICLE_RENDERING_MODE, agxOSG::ParticleSystemDrawable::ParticleShaderMode particleShaderMode = agxOSG::ParticleSystemDrawable::ROTATION_SPRITES);

  AGXOSG_EXPORT agxOSG::ParticleSystemDrawable *findParticleSystemDrawable(osg::Group* root);

  AGXOSG_EXPORT agxOSG::RigidBodySystemDrawable* findRigidBodySystemDrawable( osg::Group* root );

  AGXOSG_EXPORT osg::Group *createVisual( agx::RigidBodyEmitter* emitter, osg::Group* root);

  AGXOSG_EXPORT osg::Group *createVisual(agx::Emitter* emitter, osg::Group* root);

  /**
  Create a visual representation of an ObserverFrame (axis)
  */
  AGXOSG_EXPORT osg::MatrixTransform *createVisual(agx::ObserverFrame *frame, osg::Group* root, float scale = 1);


  /**
  Creates a visual of the particle contact network, which colors contacts depending on contact force.
  */
  AGXOSG_EXPORT osg::Geode *createParticleContactGraph(agx::ParticleSystem* particleSystem, osg::Group* root, agx::Real minForce, agx::Real maxForce);

  /**
  Creates a visual of the particle position trajectories.
  \param sampling - Sampling fraction of particles that will be used for rendering trajectories. 0 (none) - 1.0 (all).
  \param numPositions - Number of positions that will be used for each trajectory.
  */
  AGXOSG_EXPORT osg::Geode *createParticleTrajectoriesVisual(agx::ParticleSystem* particleSystem, osg::Group* root, agx::Real sampling, agx::UInt numPositions);

  /**
  Creates a visual of the rigid body position trajectories.
  \param sampling - Sampling fraction of rigidBodies that will be used for rendering trajectories. 0 (none) - 1.0 (all).
  \param numPositions - Number of positions that will be used for each trajectory.
  */
  AGXOSG_EXPORT osg::Geode *createRigidBodyTrajectoriesVisual(agxSDK::Simulation * simulation, osg::Group* root, agx::Real sampling, agx::UInt numPositions, ScalarColorMap* scalarColorMap = nullptr );

  AGXOSG_EXPORT osg::Geode *createFrameTrajectoryDrawable(agxSDK::Simulation * simulation, osg::Group* root, agx::UInt numPositions, const agx::Vec4f& defaultColor = agx::Vec4f(0.5, 0.5, 0.5, 1.0));

  AGXOSG_EXPORT agxOSG::FrameTrajectoryDrawable* findFrameTrajectoryDrawable(osg::Group* root);

  /**
  Given a rigid body, this function creates visual representation of all the geometries.
  */
  AGXOSG_EXPORT osg::Group *createVisual( agx::RigidBody* rb, osg::Group* root, float detailRatio = 1.0f, bool createAxes = false );

  /**
  Given an Assembly, this function creates visual representation of all the geometries.
  */
  AGXOSG_EXPORT osg::Group *createVisual( agxSDK::Assembly *parent, osg::Group* root, float detailRatio = 1.0f, bool createAxes = false );

  /**
  Given an Assembly, this function creates visual representation of all the geometries.
  */
  AGXOSG_EXPORT osg::Group* createVisualTemplateCaching( agxSDK::Assembly* parent, osg::Group* root, agxOSG::RigidBodyRenderCache* cache = nullptr, float detailRatio = 1.0f, bool createAxes = false );

  /**
  Given some specific classes inheriting from EventListener, this function can handle custom visual representations and effects.
  \param listener - an EventListener which hold some information about how to update visuals
  \param node - a pointer to the osg::Group where the visuals to update are expected to be
  \returns - a new listener which will update the visual, null on fail.
  */
  AGXOSG_EXPORT agxSDK::EventListener* createVisualUpdater(agxSDK::EventListener* listener, osg::Group* node);

  /**
  Given a simulation, this function creates visual representation of all the contents in the simulation, such as geometries and particle systems.
  */
  AGXOSG_EXPORT osg::Group *createVisual(agxSDK::Simulation *simulation, osg::Group* root, float detailRatio = 1.0f, bool createAxes = false);

  AGXOSG_EXPORT bool setEnableOutline(osg::Node *node, bool enabled, const agxRender::Color& color=agxRender::Color(1,1,0,1), float width = 2.0);
  AGXOSG_EXPORT bool getEnableOutline(osg::Node *node);



  AGXOSG_EXPORT void setEnableLight( bool enable, int lightNum, osg::Node* node  );

  /**
  Get the near and far points of the current window, given a camera and x and y position.
  Note that buffers for the near and far points have to be supplied.
  \param camera - A camera object
  \param x - The x position in the camera window
  \param y - The y position in the camera window
  \param near - The near point
  \param far - The far point
  */
  AGXOSG_EXPORT void getNearFarPoints(const osg::Camera* camera, float x, float y, agx::Vec3& near, agx::Vec3& far);

  /**
  Get a geometry contact given the current camera and a position in the camera window.
  If the ray passes through more than one geometry, the closest contact is returned.
  \param camera - A camera object (from e.g. agxOSG::ExampleApplication::getCamera())
  \param space - A space object (from e.g. agxSDK::Simulation::getSpace())
  \param x - The x position in the camera window (from e.g. agxSDK::GuiEventListener::update(x, y))
  \param y - The y position in the camera window (from e.g. agxSDK::GuiEventListener::update(x, y))
  */
  AGXOSG_EXPORT agxCollide::LocalGeometryContact getHoveredGeometryContact(const osg::Camera * camera, agxCollide::Space * space, float x, float y);

  AGXOSG_EXPORT bool setEnableWireFrameMode(osg::Node* node, bool flag);

  AGXOSG_EXPORT bool forceWireFrameModeOn( osg::Node* node );

  AGXOSG_EXPORT bool toggleWireFrameMode(osg::Node* node);

  AGXOSG_EXPORT bool setOrbitCamera(osgViewer::Viewer* viewer, agxOSG::GeometryNode* node,
                                    const agx::Vec3& eye, const agx::Vec3& center, const agx::Vec3& up,
                                    int trackerMode);



  /**
  Internal function to load images into an array
  */
  AGXOSG_EXPORT bool  loadCubeMapImages(const std::string& imagename, const std::string& filetype, agx::Vector<osg::ref_ptr<osg::Image> >& images);

  /**
  Internal functions for creating texture cube maps
  */
  AGXOSG_EXPORT osg::TextureCubeMap * createTextureCubeMap(agx::Vector<osg::ref_ptr<osg::Image> >& images);
  AGXOSG_EXPORT osg::TextureCubeMap * createTextureCubeMap(const std::string& imagename, const std::string& filetype);

  /**
  Adds createVisual callback to a FractureGenerator when new fragments are created that sets a specific color
  */
  AGXOSG_EXPORT void addRenderColorCallbackToFracture( agxModel::FractureGenerator * listener, osg::Group * root, const agx::Vec4f& color );

  /**
  Adds default createVisual callback to a FractureGenerator when new fragments are created that copies RenderData
  from the fractured geometry shape if it exists.
  */
  AGXOSG_EXPORT void addRenderCallbackToFracture( agxModel::FractureGenerator * listener, osg::Group * root );

  /**
  Extracts mesh data from an osg::Node, creates an agxCollide::Trimesh from it and returns the result.
  All triangle data read from the Node will be merged into one TriangleMesh in the world coordinate system.
  */
  AGXOSG_EXPORT agxCollide::Trimesh *createTrimesh(osg::Node *node);

  /**
  Helper vector's for ForceArrowListener
  */
  typedef agx::Vector<agx::Constraint1DOFObserver> Constraint1DOFObserverVector;
  typedef agx::Vector< std::pair<Constraint1DOFObserverVector, agx::Vec3>> Constraint1DOFObserverVectorVec3PairVector;
  typedef agx::HashVector< agx::RigidBodyRef,agx::Vec3> RigidBodyRefVec3PairTable;
  typedef agx::Vector<agxUtil::ExponentialMovingAverageStatistic> ExponentialMovingAverageStatisticVector;
  typedef agx::Vector<ExponentialMovingAverageStatisticVector> calculateTotalContactForceVectorVector;

  /**
  Contact force directions
  */
  enum ContactForceStatistics
  {
    NORMAL_X = 0,
    NORMAL_Y,
    NORMAL_Z,
    FRICTION_X,
    FRICTION_Y,
    FRICTION_Z,
    NUM_FORCES
  };

  /**
  Creates an arrow visualizing force,
  either contact force (for added bodies),
  or motor force for added Constraint1DOF (Hinge,Prismatic,..)

  The arrow is positioned on an local offset rel. the body associated.
  For constraints it is the first body of the first constraint.

  The size of the arrow is possible to scale, and the looks are tuned
  with cone length and arrow radius.
  */
  class AGXOSG_EXPORT ForceArrowRenderer : public agxSDK::GuiEventListener
  {
  public:
    /* Create a force arrow renderer*/
    ForceArrowRenderer();

    /*
    Add body for contact force visualization, normal and friction force are visualized with one arrow each.
    \param body - a rigid body.
    \param localOffsetRelBody - the start point of the arrow relative the rigid body.
    */
    void add(agx::RigidBody* body, const agx::Vec3& localOffsetRelBody);

    /*
    Remove a body from the contact force arrow visualization.
    */
    agx::Bool remove(agx::RigidBody* body);

    /*
    Add a vector of 1DOF constraints. The sum of the motor forces are visualized with one arrow.
    \param constraintVector - a vector with Constraint1DOF
    \param localOffsetRelBody1 - the start point of the arrow relative the first rigid body of the first constraint.
    */
    void add(Constraint1DOFObserverVector& constraintVector, const agx::Vec3& localOffsetRelBody1);

    /*
    Remove a constraints from the constraint force arrow visualization.
    */
    agx::Bool remove(agx::Constraint1DOF* constraint);
    agx::Bool remove(Constraint1DOFObserverVector& constraintVector);

    /*
    A smooth factor for a filter, smoothing the force arrow size.
    \param smoothFactor - factor to be used with a agxUtil::ExponentialMovingAverageStatistic
    */
    void setSmoothFactor(const agx::Real& smoothFactor);
    agx::Real getSmoothFactor() const;

    /*
    Look and feel of the arrow
    \param length - cone length
    */
    void setConeLength(const agx::Real& length);
    agx::Real getConeLength() const;
    /*
    Look and feel of the arrow.
    \param radius - arrow radius
    */
    void setArrowRadius(const agx::Real& radius);
    agx::Real getArrowRadius() const;

    /*
    Scale the size of the arrow (scale 1 would result in 1 Newton giving length of 1 l.u. )
    \param scale - a scale factor from Newton to length units (l.u.).
    */
    void setScale(const agx::Real& scale);
    agx::Real getScale() const;

    /**
    Called once per simulation frame
    \param x,y coordinate in normalized screen coordinates of the mouse pointer
    */
    virtual void update(float x, float y);

  protected:
    virtual ~ForceArrowRenderer();

    void drawArrow(const agx::Vec3& start, const agx::Vec3& force);

    agx::Bool m_enabled;
    agx::Real m_smoothFactor;
    agx::Real m_coneLength;
    agx::Real m_arrowRadius;
    agx::Real m_scale;
    RigidBodyRefVec3PairTable m_bodies;
    Constraint1DOFObserverVectorVec3PairVector m_constraint1DOFVectors;
    calculateTotalContactForceVectorVector m_bodyStatistics;
    ExponentialMovingAverageStatisticVector m_constraintStatistics;
  };
}

#endif /* _AGXOSG_UTILS_H_ */
