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

#include <agx/agx.h>
#include <agxUtil/agxUtil.h>

namespace agxCollide {
  class Trimesh;
}

namespace agxUtil {

  class ParallelTrimeshDeformer;

  AGX_DECLARE_POINTER_TYPES(BodyLocalOffset);
  AGX_DECLARE_VECTOR_TYPES(BodyLocalOffset);

  class AGXPHYSICS_EXPORT BodyLocalOffset : public virtual agx::Referenced, public virtual agxStream::Serializable
  {
    public:
      BodyLocalOffset(const agx::RigidBody* relativeBody,
                      const agx::RigidBody* body, 
                      agx::Real offsetReach,
                      agx::Real fullOffsetPercent,
                      agx::Vec3* relativePosition);
      agx::Vec3 getWorldOffsetVector() const;
      const agx::RigidBody* getBody() const;
      agx::Real getOffsetReach() const;
      void setFullOffsetPercent(agx::Real percent);
      agx::Real getFullOffsetPercent();
      agx::Real getFullOffsetReach() const;
      agx::Vec3 calculateRelativeTranslate() const;
      agx::Vec3 getOriginalRelativeTranslate() const;
      
      typedef agx::HashVector<size_t, agx::Vec3> IndexVec3HashVector;

      const IndexVec3HashVector& getAffectedVertices() const;
      void collectAffectedVertices(const agxData::LocalVector<agx::Vec3>& worldVertices);

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::BodyLocalOffset);

    protected:
      BodyLocalOffset();
      virtual ~BodyLocalOffset();
      const agx::RigidBody* m_relativeBody;
      const agx::RigidBody* m_body;
      agx::Real m_offsetReach;
      agx::Real m_fullOffsetPercent;
      agx::Vec3 m_relativeRelativeTranslate;

      IndexVec3HashVector m_affectedVertices;
  };


  /**
  Class for deforming a Trimesh shape given one or many operations
  */
  AGX_DECLARE_POINTER_TYPES(TrimeshDeformer);
  class AGXPHYSICS_EXPORT TrimeshDeformer : public agxSDK::StepEventListener
  {
  public:

    /**
    \param trimesh         - an trimesh shape, which we like to update each timestep
    \param trimeshGeometry - the geometry which include the trimesh shape. There should be no relative shape transform.
    */
    TrimeshDeformer(agxCollide::Trimesh* trimesh,
      agxCollide::Geometry* trimeshGeometry);

    /**
    The dual frame interpolation will move all verticies of the trimesh in between the
    x-y planes of interpolationFrame1 and interpolationFrame2.
    The relative transform between the interpolation frames at creation is considered the relaxed state.
    When the simulation forces the frames to move individually, all vertices get updated positions.
    The rule is that the verticies in the local x-y planes are not transformed, but then given the
    distance to the respective planes, the position for each vertex gets a weighted average depending on distance
    given where they would be if the shape was located at either interpolation frame, un-transformed.

    This method is useful when visualizing the deformation of springs or suction cups,
    where the dynamics is represented by one constraint.

    Initial state                                Transformed state
    ____________ x-y plane interpolationFrame2        _ \
       /  /                                         /    \
      /  /                                        /     _/\ rotated x-y plane interpolationFrame2
    _/__/________ x-y plane interpolationFrame1  /    /
                                                /___/____x-y plane interpolationFrame1

    \param interpolationFrame1 - a frame from where the trimesh shape starts
    \param interpolationFrame2 - a frame from where the trimesh shape ends
    \param localMeshDir        - the mesh local direction defining trimesh start and end
    */
    bool activateDualFrameInterapolation( agx::Frame* interpolationFrame1,
                                          agx::Frame* interpolationFrame2,
                                          agx::Vec3 localMeshDir);

    /**
     * @brief Prepare all local offsets, and calculate weight for each effected vertex
     * 
     */
    void activateLocalOffsetInterpolation();

    /**
     * @brief For simulation models that consist of several rigid bodies, but we like to have just one
    visual but it is interesting to visualize local deformations.
    Could be a tire, bending steel plate or a flexible lip of a suction cup which should deform non uniformly.
    \param relativeBody - a central body which is the owner of the trimeshGeometry that will deform
    \param body         - a body which transform relative the relativeBody we like to visualize by trimesh deformations
    \param offsetReach  - define within which distance from the body should vertices be affected
    \param fullOffsetPercent - between (0,1) how big fraction within the offsetReach should vertices fully reflect the offset, outside it will decay linearly
    \param relativePosition - specify if there is a known wanted realtive position, if not specified the current position will be used.
    */
    bool addLocalDeformation(const agx::RigidBody* relativeBody, 
                             const agx::RigidBody* body,
                             agx::Real offsetReach, 
                             agx::Real fullOffsetPercent,
                             agx::Vec3* relativePosition = nullptr);

    /**
    Update the trimesh given the operations activated
    */
    void update();

    void last(const agx::TimeStamp&);

    /**
    \returns pointer to the geometry of the trimesh being deformed
    */
    agxCollide::Geometry* getGeometry();

    /**
    \returns pointer to the trimesh being deformed
    */
    agxCollide::Trimesh* getTrimesh();

    /**
    \returns true if the trimesh and geometry instances exist.
    */
    bool isValid() const;

    /**
    Specify if the mesh collision properties, including mass properties
    and bounding box calculations needed for collision detection
    should be calculated each time step. Not needed if the trimesh is only for the visual.
    */
    void setEnableUpdateCollisionGeometry(bool enable);

    /**
    \returns true if the mesh collision properties, including mass properties
    and bounding box calculations needed for collision detection
    should be calculated each time step.
    */
    bool getEnableUpdateCollisionGeometry() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::TrimeshDeformer);


    void updateCollisionGeometry();


  protected:
    TrimeshDeformer();
    virtual ~TrimeshDeformer();
    
    friend class ParallelTrimeshDeformer;
    void setOwner(ParallelTrimeshDeformer* deformer);

    void removeNotification();


    void updateDualFrameInterpolation();
    void updateLocalOffset(const BodyLocalOffsetRef offset);
    agxCollide::TrimeshObserver m_trimesh;
    agxCollide::GeometryObserver m_trimeshGeometry;
    agx::FrameRef m_interpolationFrame1;
    agx::FrameRef m_interpolationFrame2;
    agx::Real m_fullInterpolationLength;
    agx::Vec3 m_localMeshNormal;
    bool m_enableUpdate;
    agx::AffineMatrix4x4 m_localFrame1VertexRotation;
#ifndef SWIG    
    agx::Vector<agx::Vec3> m_originalVertices;
    agx::Vector<agx::Vec3> m_verticesRelative1;
    agx::Vector<agx::Vec3> m_verticesRelative2;
    BodyLocalOffsetRefVector m_localOffsets;

    agx::observer_ptr<ParallelTrimeshDeformer> m_owner;
#endif

  };



  /**
  
  */
  AGX_DECLARE_POINTER_TYPES(ParallelTrimeshDeformer);
  class AGXPHYSICS_EXPORT ParallelTrimeshDeformer : public agxSDK::StepEventListener
  {
    public:

      ParallelTrimeshDeformer();
      void add(TrimeshDeformer* deformer);
      bool remove(TrimeshDeformer* deformer);

      void setParallel(bool flag);
      bool getParallel() const;

  protected:

    void last(const agx::TimeStamp& time);

    void addNotification();
    void removeNotification();


    typedef agx::SetVector<TrimeshDeformerRef> TrimeshDeformerSetVector;
    TrimeshDeformerSetVector m_deformers;

    virtual ~ParallelTrimeshDeformer();

    bool m_parallel;
  };
}

