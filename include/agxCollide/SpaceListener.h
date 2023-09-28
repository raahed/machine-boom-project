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


namespace agxCollide
{
  class Shape;
  class Space;
  class Geometry;

  /**
  Class for listening for add/remove geometry events for an associated Space.
  */
  class AGXPHYSICS_EXPORT SpaceListener : public agx::Referenced
  {
  public:

    /// Default constructor
    SpaceListener();

    /**
    Called when this SpaceListener is added to a space
    \param space - The space which this SpaceListener is added to.
    */
    virtual void addNotification(agxCollide::Space* space);

    /**
    Called when this SpaceListener is removed from a space
    \param space - The space which this SpaceListener is removed from.
    */
    virtual void removeNotification(agxCollide::Space* space);

    /**
    Called whenever a Geometry is added to the associated Space.
    \param space - The space for which a geometry is added
    \param geometry - The added geometry
    **/
    virtual void addGeometryNotification(agxCollide::Space* space, agxCollide::Geometry* geometry);
    /**
    Called whenever a Geometry is removed from the associated Space.
    \param space - The space for which a geometry is removed
    \param geometry - The removed geometry
    **/
    virtual void removeGeometryNotification(agxCollide::Space* space, agxCollide::Geometry* geometry);

    /**
    Called whenever a Shape is added to a Geometry, where the geometry belongs to a Space.
    \param space - The Space in which the Geometry is associated.
    \param geometry - The Geometry which is being modified.
    \param shape - The Shape which is added to a Geometry
    **/
    virtual void addShapeNotification(agxCollide::Space* space, agxCollide::Geometry* geometry, agxCollide::Shape* shape);

    /**
    Called whenever a Shape is removed from a Geometry, where the geometry belongs to a Space.
    \param space - The Space in which the Geometry is associated.
    \param geometry - The Geometry which is being modified.
    \param shape - The Shape which is removed from a Geometry
    **/
    virtual void removeShapeNotification(agxCollide::Space* space, agxCollide::Geometry* geometry, agxCollide::Shape* shape);

  protected:

    /// Destructor
    virtual ~SpaceListener();

  private:
    friend class Space;

    typedef agx::Event2<Space*, Geometry*> GeometryEvent;
    typedef agx::Event3<Space*, Geometry*, Shape*> ShapeEvent;



    GeometryEvent::CallbackType m_addGeometryCallback;
    GeometryEvent::CallbackType m_removeGeometryCallback;
    ShapeEvent::CallbackType m_addShapeCallback;
    ShapeEvent::CallbackType m_removeShapeCallback;
  };

  AGX_DECLARE_POINTER_TYPES(SpaceListener);

}
