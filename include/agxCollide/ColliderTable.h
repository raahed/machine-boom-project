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

#ifndef AGX_COLLIDERTABLE_H
#define AGX_COLLIDERTABLE_H

#include <agxCollide/Shape.h>
#include <agxCollide/ShapeCollider.h>
#include <agx/Initializer.h>

namespace agxCollide
{


  class ColliderTableCleanup {
    public:

    private:
      friend class ColliderTable;

      ColliderTableCleanup() {}

      ~ColliderTableCleanup();
  };


  class AGXPHYSICS_EXPORT ColliderTable
  {
  public:
    /**
    Get a shape collider for testing two shapes against each other.
    \return The collider, or 0 if no collider is registered for the shape types.
    */
    static ShapeCollider *getCollider( const Shape *a, const Shape *b );
    static ShapeCollider *getCollider( unsigned typeA, unsigned typeB );

    /**
    Set a shape collider, normally handled by the PluginManager.
    */
    static void setCollider( unsigned typeA, unsigned typeB, ShapeCollider *collider );
    static void setCollider( Shape *a, Shape *b, ShapeCollider *collider );

    /// Initialize the collider table
    static void init();


  private:
    friend class ColliderTableCleanup;

    static void wipe();

    static ShapeCollider *s_colliders[Shape::NUM_TYPES * Shape::NUM_TYPES];
    static bool s_collidersInitialized;
    static ColliderTableCleanup s_colliderTableCleanup;
  };


  /* Implementation */

  #define CalcPairId(id1, id2) (std::min(id1, id2) * Shape::NUM_TYPES + std::max(id1, id2))


  AGX_FORCE_INLINE ShapeCollider *ColliderTable::getCollider( const Shape *a, const Shape *b )
  {
    return s_colliders[CalcPairId(a->getType(), b->getType())];
  }

  AGX_FORCE_INLINE ShapeCollider *ColliderTable::getCollider( unsigned typeA, unsigned typeB )
  {
    return s_colliders[CalcPairId(typeA, typeB)];
  }

}

#endif /* _COLLIDERTABLE_H_ */
