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

#ifndef AGXMODEL_TREE_H
#define AGXMODEL_TREE_H

#include <agxModel/export.h>
#include <agxSDK/Assembly.h>
#include <agxSDK/ContactEventListener.h>
#include <agxSDK/StepEventListener.h>

#include <agx/LockJoint.h>
#include <agx/HashVector.h>

#include <agxCollide/Cylinder.h>
#include <agxCollide/Capsule.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES(Tree);
  AGX_DECLARE_VECTOR_TYPES(Tree);
  /**
  Tree is a structure of Branch objects in a mathematical Tree structure where a Branch can have several
  children Branches an so forth.
  Tree holds functions to cut, split and in other ways change the Tree structure.
  */
  class AGXMODEL_EXPORT Tree : public agxSDK::Assembly
  {
    public:
      class Branch;

      typedef agx::ref_ptr< Branch >                                                              BranchRef;
      typedef agx::observer_ptr< Branch >                                                         BranchObs;
      typedef agx::Vector< BranchRef >                                                            BranchContainer;
      typedef BranchContainer::iterator                                                           BranchIterator;
      typedef BranchContainer::const_iterator                                                     BranchConstIterator;
      typedef agx::HashVector< agxCollide::Geometry*, agx::observer_ptr< agxCollide::Geometry > > DisabledCollisionsGeometryContainer;
      typedef agx::SetVector< agx::UInt32 >                                                       GroupIdContainer;

      /**
      Branch is a part of the Tree, with a body and a constraint to its parent
      */
      class AGXMODEL_EXPORT Branch : public agx::Referenced, public agxStream::Serializable
      {
        public:
          enum MaxLoadType { TRANSLATIONAL = 0, ROTATIONAL };

        public:
          /// Default constructor
          Branch();
          
          /**
          Construct branch given start- and end point in world. The rigid body model frame will
          be located at \p start, and rotated so that z-axis points toward end.
          \param start - start position, in world, for this branch
          \param end - end position, in world, for this branch
          */
          Branch( const agx::Vec3& start, const agx::Vec3& end );

          /**
          Construct branch given transform. The rigid body will initially have this transform,
          i.e., this is the model frame during initialize.
          \param transform - model frame transform
          */
          Branch( const agx::AffineMatrix4x4& transform );

          /**
          Add child branch to this branch. This branch will be parent
          to \p branch and later attached to it with a constraint.
          \param branch - child branch
          \return true if add was successful - otherwise false (e.g., loops)
          */
          bool addBranch( agxModel::Tree::Branch* branch );

          /**
          Remove a child branch. Parent will be set to 0 for \p branch.
          \param branch - child branch
          \return true if the branch was successfully removed
          */
          bool removeBranch( agxModel::Tree::Branch* branch );

          /**
          Remove child given index. Parent will be set to 0 for branch at \p index.
          \param index - index of branch to remove
          \return true if the branch, at index, was successfully removed
          */
          bool removeBranch( size_t index );

          /**
          \return the number of child branches for this branch
          */
          size_t getNumBranches() const;

          /**
          \return child branch \p i
          */
          agxModel::Tree::Branch* getBranch( size_t i ) const;

          /**
          Assign custom/user data of any type. Only constraint is that the class
          inherits agx::Referenced for easier memory managment.
          \param customData - custom/user data for this branch
          */
          void setCustomData( agx::Referenced* customData );

          /**
          \return the custom data assigned to this branch (0 if none)
          */
          agx::Referenced* getCustomData() const;

          /**
          \return parent branch to this branch
          */
          agxModel::Tree::Branch* getParent() const;

          /**
          \return the rigid body of this branch
          */
          agx::RigidBody* getRigidBody() const;

          /**
          \return the constraint of this branch
          */
          agx::Constraint* getConstraint() const;

          /**
          \return direction of this branch
          */
          agx::Vec3 getDirection() const;

          /**
          \return the local direction of this branch
          */
          agx::Vec3 getLocalDirection() const;

          /**
          Assign max load for this branch (connection at start). Maximum load is
          divided into two parts, translational and rotational, units force.
          When/if the constraint exceeds one of these values an event is fired.
          \param translational - translational max load, unit force (default: infinity = disabled)
          \param rotational - rotational max load, unit force (default: infinity = disabled)
          */
          void setMaxLoad( agx::Real translational, agx::Real rotational );

          /**
          Assign translational max load for this branch.
          \param maxLoad - translational max load, unit force (default: infinity = disabled)
          */
          void setMaxTranslationalLoad( agx::Real maxLoad );

          /**
          Assign rotational max load for this branch.
          \param maxLoad - rotational max load, unit force (default: infinity = disabled)
          */
          void setMaxRotationalLoad( agx::Real maxLoad );

          /**
          \return both translational and rotational max load (index with Branch::TRANSLATIONAL and
          Branch::ROTATIONAL)
          */
          agx::Vec2 getMaxLoad() const;

          /**
          \return translational max load for this branch
          */
          agx::Real getMaxTranslationalLoad() const;

          /**
          \return rotational max load for this branch
          */
          agx::Real getMaxRotationalLoad() const;

          /**
          \return both translational and rotational load (current) (index with Branch::TRANSLATIONAL
          and Branch::ROTATIONAL)
          */
          agx::Vec2 getCurrentLoad() const;

          /**
          \return current translational load in this branch
          */
          agx::Real getCurrentTranslationalLoad() const;

          /**
          \return current rotational load in this branch
          */
          agx::Real getCurrentRotationalLoad() const;

        public:
          /**
          Calculates and returns an estimated length of this branch. Default this method can estimate
          the length of the first shape in \p geometry of shape types: agxCollide::Shape::CYLINDER
          and agxCollide::Shape::CAPSULE.
          \return the length, or at least estimated length, of this branch
          */
          virtual agx::Real getLength( const agxCollide::Geometry* geometry ) const;

          /**
          This branch is about to be divided into two branches (cut)!

          Split a shape in this branch (\p shapeInBranch is part of a geometry of this branch rigid body) at a distance along
          the shape. Creates a new shape and also returns the new length from start to \p shapePosition, \p newBelowLength, and
          new length from \p newBelowLength to end, \p newAboveLength. The new shape will belong to \p newBranch after this call.

          Default this method supports shape types: agxCollide::Shape::CYLINDER and agxCollide::Shape::CAPSULE.
          \param shapeInBranch - a shape in a geometry in this branch rigid body
          \param shapePosition - a distance along shapeInBranch y-axis (because normally cylinder or capsule)
          \param newBranch - the branch the new shape will belong to
          @param[out] newBelowLength - new length of this branch (along y-axis)
          @param[out] newAboveLength - new length of the new shape
          \return a new shape where the length of the new shape should be round \p newAboveLength of length
          */
          virtual agxCollide::Shape* splitShape( agxCollide::Shape* shapeInBranch, agx::Real shapePosition, agxModel::Tree::Branch* newBranch, agx::Real& newBelowLength, agx::Real& newAboveLength );

          /**
          Creates the constraint with this branch's parent. Default is lock joint along this branch direction.
          \note It's not defined to call this method if the parent is the root.
          */
          virtual agx::ConstraintRef createConstraintWithParent() const;

          /**
          \return clone of this branch
          */
          virtual agxModel::Tree::Branch* clone();

          AGXSTREAM_DECLARE_SERIALIZABLE(Branch);

        protected:
          virtual ~Branch();

          /**
          Assign parent to this branch.
          \param parent - parent branch
          */
          void setParent( agxModel::Tree::Branch* parent );

          friend class Tree;
          /**
          Assign constraint attached at start point of this branch.
          \param constraint - constraint attached at start point of this branch
          */
          void setConstraint( agx::Constraint* constraint );

        protected:
          BranchObs m_parent;
          BranchContainer m_branches;
          agx::ref_ptr< agx::Referenced > m_customData;

          agx::RigidBodyRef m_rb;
          agx::ConstraintRef m_constraint;
          agx::Vec2 m_maxLoad;
      };

      /**
      BranchEventListener handles certain events happening to a Tree
      */
      class AGXMODEL_EXPORT BranchEventListener : public agx::Referenced, public agxStream::Serializable
      {
        public:
          /**
          Construct branch event listener to listen to certain events happening in a Tree.
          Not valid to use this branch event listener in more than one Tree.
          */
          BranchEventListener();

          /**
          Event when branch \p branch has been configured with position,
          orientation and constraint to parent (if present).
          \param branch - the newly configured branch
          */
          virtual void onCreate( agxModel::Tree::Branch* branch );

          /**
          Event when branch \p branch connection is under high load.
          \param branch - branch under high load
          */
          virtual void onHighLoad( agxModel::Tree::Branch* branch );

          /**
          Clone method used by cut.
          \return a clone of this branch event listener
          */
          virtual agxModel::Tree::BranchEventListener* clone();

          AGXSTREAM_DECLARE_SERIALIZABLE(BranchEventListener);

        protected:
          virtual ~BranchEventListener();

          /**
          \return the tree this listener is assigned to
          */
          Tree* getTree() const;

        private:
          friend class Tree;
          void setTree( Tree* tree );

        private:
          agx::observer_ptr< Tree > m_tree;
      };

      typedef agx::ref_ptr< BranchEventListener > BranchEventListenerRef;

      /**
      TreeExecuteFilter allows to catch contacts with the Tree.
      Note that TreeExecuteFilter is not serializable. A tree that has been made static and 
      to wake on contact, using a TreeExecuteFilter, will remain in the same state, but not 
      able to wake up after being restored. Use Tree::setTreeExecuteFilter to set the filter
      again in that case.
      */
      class AGXMODEL_EXPORT TreeExecuteFilter : public agxSDK::ExecuteFilter
      {
        public:
          /// Default constructor
          TreeExecuteFilter();

          /**
          Find contacts where one geometry is the Tree
          \return true if contact is with Tree
          */
          virtual bool match( const agxCollide::GeometryContact& contact ) const override;

          /**
          Find contacts where one geometry is the Tree
          \return true if one geometry in the pair is part of the Tree
          */
          virtual bool match( const agxCollide::GeometryPair& pair ) const override;
          using agxSDK::ExecuteFilter::match;

          /**
          Clone method of the TreeExecuteFilter
          */
          virtual agxModel::Tree::TreeExecuteFilter* clone();

        protected:
          virtual ~TreeExecuteFilter();
          agx::UInt32 getTreeID() const;

        private:
          friend class Tree;
          agx::UInt32 m_treeID;
      };

      typedef agx::ref_ptr< TreeExecuteFilter > TreeExecuteFilterRef;

      enum State { INITIALIZED = (1<<0), STATIC = (1<<1), WAKE_ON_CONTACT = (1<<2), RESTORING = (1<<3) };

    public:
      /// Default constructor
      Tree();

      /**
      Clone call when new tree is needed from internal methods. Implement
      this if when you have your own implementation of this tree class.
      \return new tree
      */
      virtual Tree* clone();

      /**
      \return the root branch (created with this tree)
      */
      Branch* getRoot() const;

      /**
      \return true if this tree is initialized - otherwise false
      */
      bool initialized() const;

      /**
      Explicit call to this method to check return value. This tree will
      initialize it self when added to a simulation.
      \return true if initialize procedure was successful - otherwise false
      */
      virtual bool initialize();

      /**
      \return true if this tree is simplified - otherwise false
      */
      bool isStatic() const;

      /**
      \return true if this tree should wake on contact in state STATIC
      */
      bool wakeOnContact() const;

      /**
      Simplify this tree. Make all rigid bodies static and remove all constraints
      from the simulation. It's possible to flag if this tree should wake on
      contact with other objects. If a contact wakes this tree, it will not automatically
      become static again. This has to be monitored by the user.

      Default contact execute filter will trigger on all other contacts with this tree's
      geometries.
      \param wakeOnContactEvent - true to enable this tree to wake up upon contact
      \param filter - default all contacts with this tree
      \return true if the state was changed
      */
      bool makeStatic( bool wakeOnContactEvent, agxModel::Tree::TreeExecuteFilter* filter
#ifndef SWIG
        // SWIG cannot handle default arguments with the new keyword, at the moment.
        = new agxModel::Tree::TreeExecuteFilter()
#endif
        );

      /**
      Make all rigid bodies dynamic and add all constraints to the simulation. This is
      the default state.
      \return true if the state was changed
      */
      bool makeDynamic();

      /**
      Set or change the TreeExecuteFilter on a tree that has been made static and set to
      wake on contact. Mainly to be used on a restored tree, since TreeExecuteFilters are not
      serializable. Note that a filter cannot be set on a tree that is not set to static and
      to wake on contact. Use Tree::makeStatic to do that.
      \param filter - the wanted TreeExecuteFilter
      \return true if filter was set, false otherwise
      */
      virtual bool setTreeExecuteFilter(agxModel::Tree::TreeExecuteFilter* filter);

      /**
      Assign branch event listener. Unique for this tree.
      \param branchListener - new listener unique for this tree
      */
      virtual void setBranchEventListener( agxModel::Tree::BranchEventListener* branchListener );

      /**
      \return the branch event listener assigned to this tree - 0 if none
      */
      agxModel::Tree::BranchEventListener* getBranchEventListener() const;

      /**
      Attach bottom branch to an object given attachment frame. Make sure
      this tree has the correct transform when calling this method.
      \param rb - attach to rigid body (if 0, attached to world)
      \param frame - attachment frame, will be calculated in this method (invalid if 0)
      \return true if attach is successful - otherwise false (not initialized, frame = 0, root->getNumBranches() = 0)
      */
      virtual bool attach( agx::RigidBody* rb, agx::Frame* frame );

      /**
      Add group id to all geometries in this tree.
      \param id - id to add to all geometries in this tree
      \return true if valid add
      */
      bool addGroupID( agx::UInt32 id );

      /**
      Removes group id from all geometries in this tree.
      \param id - id to remove
      \return true if valid remove
      */
      bool removeGroupID( agx::UInt32 id );

      /**
      \return true if this tree has \p id
      */
      bool hasGroupID( agx::UInt32 id ) const;

      /**
      Enable or disable collisions between this tree and a rigid body.
      \param rb - rigid body
      \param flag - true to enable, false to disable
      \return true if collision flag was set - otherwise false
      */
      bool setEnableCollisions( agx::RigidBody* rb, bool flag );

      /**
      Enable or disable collisions between this tree and a geometry.
      \param geometry - the geometry
      \param flag - true to enable, false to disable
      \return true if collision flag was set - otherwise false
      */
      bool setEnableCollisions( agxCollide::Geometry* geometry, bool flag );

      /**
      \return true if collisions is enabled between this tree and all geometries in \p rb
      */
      bool getEnableCollisions( const agx::RigidBody* rb ) const;

      /**
      \return true if collisions is enabled between this tree and \p geometry
      */
      bool getEnableCollisions( const agxCollide::Geometry* geometry ) const;

      /**
      Assign custom data.
      \param customData - custom data
      */
      void setCustomData( agx::Referenced* customData );

      /**
      \return any custom data assigned to this tree
      */
      agx::Referenced* getCustomData() const;

      /**
      Assign new material. All geometries in this tree will have this material.
      \param material - new material
      */
      virtual void setMaterial( agx::Material* material );

      /**
      \return current material
      */
      agx::Material* getMaterial() const;

      /**
      Set new solve type for the branch connections. Only valid for initialized trees.
      Default: DIREKT_AND_ITERATIVE.
      \param solveType new solve type
      */
      virtual void setConstraintSolveType( agx::Constraint::SolveType solveType );

      /**
      Cut geometry (only one shape per geometry supported) given point in world and
      minimum shape height, after cut. If cut is too close to the shape ends
      (minShapeHeight), the cut is made between two geometries or directly
      at a branch connection.

      This method first splits this tree (see split method) and later cuts given
      the branch returned from split.

      \note Split and cut assumes that the geometries in a branch are ordered in position with first geometry furthest
            down in branch body coordinates.
      \param geometry - geometry to cut (only one shape per geometry supported)
      \param point - point in world
      \param minShapeHeight - resulting geometry/shape minimum height (default: 1 cm)
      \return new tree "above" branch with geometry \p geometry
      */
      virtual Tree* cut( agxCollide::Geometry* geometry, const agx::Vec3& point, agx::Real minShapeHeight = agx::Real( 1E-2 ) );

      /**
      Cut at branch and get a new tree. Branch \p branch will be member (first child of root)
      in the new tree.
      \param branch - cut position in tree
      \return new tree
      */
      virtual Tree* cut( agxModel::Tree::Branch* branch );

      /**
      Splits this tree given geometry and point in world. During this split a new branch
      is created and will be child to branch the geometry belongs to. Parameters are
      inherited from the old branch.

      With minShapeHeight it's possible to control how the split is made. If the split results in a
      shape with height less than that, the split will take place between two geometries (if present)
      or returns 0 if the split is too close to a branch connection (i.e., split unnecessary).
      \note Split and cut assumes that the geometries in a branch are ordered in position with first geometry furthest
            down in branch body coordinates.
      \param geometry - geometry to split (only one shape per geometry supported)
      \param point - point in world
      \param minShapeHeight - resulting geometry/shape minimum height (default: 1 cm)
      \return the new branch which is a child branch to the one holding \p geometry
      */
      virtual Branch* split( agxCollide::Geometry* geometry, const agx::Vec3& point, agx::Real minShapeHeight = agx::Real( 1E-2 ) );

      /**
      Sets default material parameters for a tree.
      */
      static void setDefaultParameters( agx::Material* material );

      /**
      Search the tree for rigid body \p rb.
      \param rb - rigid body to search for
      \return branch with rigid body \p rb (0 if not found)
      */
      Branch* findBranch( const agx::RigidBody* rb ) const;

      /**
      Search the tree for rigid body \p rb given start branch (i.e., whole tree if root). Recursive.
      \param fromBranch - branch to start from
      \param rb - rigid body to search for
      \return branch with rigid body \p rb (0 if not found)
      */
      Branch* findBranch( const agxModel::Tree::Branch* fromBranch, const agx::RigidBody* rb ) const;

      /**
      Remove all branches from this tree.
      */
      virtual void clear();

      /**
      Utility method to set linear velocity damping for all branches. I.e.,
      Tree::setLinearVelocityDamping( aTree->getRoot(), linDamping ). Recursive.

      \note Only utility method, i.e., a tree does NOT store or keep track of
            damping information.
      \param branch - start branch
      \param linearVelocityDamping - linear velocity damping
      */
      static void setLinearVelocityDamping( agxModel::Tree::Branch* branch, const agx::Vec3& linearVelocityDamping );

      /**
      Utility method to set angular velocity damping for all branches. I.g.,
      Tree::setAngularVelocityDamping( aTree->getRoot(), angDamping ). Recursive.

      \note Only utility method, i.e., a tree does NOT store or keep track of
            damping information.
      \param branch - start branch
      \param angularVelocityDamping - angular velocity damping
      */
      static void setAngularVelocityDamping( agxModel::Tree::Branch* branch, const agx::Vec3& angularVelocityDamping );

      static Tree* find(agxSDK::Simulation* simulation, const agx::Name& name);

      static Tree* find(agxSDK::Simulation* simulation, const agx::Uuid& uuid);

      static TreePtrVector findAll(agxSDK::Simulation* simulation, const agx::Name& name);

      static TreePtrVector getAll(agxSDK::Simulation* simulation);

      AGXSTREAM_DECLARE_SERIALIZABLE(Tree);

    protected:
      Tree( agxModel::Tree::Branch* root );
      virtual ~Tree();

      virtual void addNotification( agxSDK::Simulation* simulation ) override;
      using agxSDK::Assembly::addNotification;
      virtual void removeNotification( agxSDK::Simulation* simulation ) override;
      using agxSDK::Assembly::removeNotification;

      /**
      Pre-collide event callback.
      */
      virtual void preCollide();

      /**
      Pre event callback.
      */
      virtual void pre();

      /**
      Post step event callback. Check max load and fire events.
      */
      virtual void post();

      /**
      Traverse this tree and configures the branches. Recursive.
      \param branch - current branch
      */
      void configure( Branch* branch );

      /**
      Adds the id to the branch geometries. Recursive.
      \param branch - current branch
      \param id - id to add
      */
      void addGroupID( Branch* branch, agx::UInt32 id );

      /**
      Removes the id from the branch geometries. Recursive.
      \param branch - current branch
      \param id - id to remove
      */
      void removeGroupID( Branch* branch, agx::UInt32 id );

      /**
      Changes motion control and adds/removes constraints depending on flag. Recursive.
      \param branch - current branch
      \param flag - true for STATIC and remove
      */
      void simplify( Branch* branch, bool flag, agxSDK::Simulation* simulation );

      /**
      Checks current load. If it exceeds the maximum load an event is fired. Recursive.
      \param branch - current branch
      */
      virtual void checkLoad( Branch* branch );

      /**
      Move from source to destination.
      */
      void split( Branch* source, Branch* destination ) const;

      using agxSDK::Assembly::add;
      using agxSDK::Assembly::remove;

      /**
      Removes rigid body, constraint etc associated to this tree. Recursive.
      */
      void remove( Branch* branch );

      /**
      Adds rigid body, constraint etc to this tree. Recursive.
      \param branch - branch with rigid body and constraint to add
      */
      void add( Branch* branch );

      /**
      Assign material to all geometries for branch. Recursive.
      \param branch - current branch
      \param material - new material
      */
      void setMaterial( Branch* branch, agx::Material* material );

      /**
      Assign constraint solve type. Recursive.
      \param branch - current branch
      \param solveType - new constraint solve type
      */
      void setConstraintSolveType( Branch* branch, agx::Constraint::SolveType solveType );

      /**
      Copy branch parameters from source to destination.
      */
      void copyParameters( const Branch* source, Branch* destination ) const;

      /**
      Copy constraint parameters from source to destination.
      */
      void copyParameters( const agx::Constraint* source, agx::Constraint* destination ) const;

      /**
      Searches branch children and examines orientation etc to "guess" which of the children
      that is along the main/current tree branch.
      */
      Branch* guessConnectingChild( Branch* branch ) const;

    protected:
      class TreeContactEventListener : public agxSDK::ContactEventListener
      {
        public:
          TreeContactEventListener( Tree* tree )
            : m_tree( tree ) { setMask( IMPACT ); }

        protected:
          virtual ~TreeContactEventListener() {}

          virtual KeepContactPolicy impact( const agx::TimeStamp&, agxCollide::GeometryContact* );

        protected:
          agx::observer_ptr< Tree > m_tree;
      };

      typedef agx::ref_ptr< TreeContactEventListener > TreeContactEventListenerRef;

      class TreeStepEventListener : public agxSDK::StepEventListener
      {
        public:
          TreeStepEventListener( Tree* tree )
            : m_tree( tree ) { setMask( PRE_COLLIDE | PRE_STEP | POST_STEP ); }

        protected:
          virtual ~TreeStepEventListener() {}

          virtual void preCollide( const agx::TimeStamp& ) override;
          virtual void pre( const agx::TimeStamp& ) override;
          virtual void post( const agx::TimeStamp& ) override;

        protected:
          agx::observer_ptr< Tree > m_tree;
      };

      typedef agx::ref_ptr< TreeStepEventListener > TreeStepEventListenerRef;

    protected:
      BranchRef                           m_root;
      int                                 m_state;
      agx::UInt32                         m_geometriesID;
      agx::UInt32                         m_disabledCollisionsID;
      DisabledCollisionsGeometryContainer m_externalDisabledGeometries;
      GroupIdContainer                    m_geometryIDs;
      agx::ref_ptr< agx::Referenced >     m_customData;
      agx::MaterialRef                    m_material;
      BranchEventListenerRef              m_branchListener;
      TreeContactEventListenerRef         m_wakeOnContactListener;
      TreeStepEventListenerRef            m_stepListener;
  };

  template< typename T >
  T* cutShape( T* shape, agx::Real position, agx::Real& newBelowLength, agx::Real& newAboveLength )
  {
    agx::Real oldLength = shape->getHeight();
    newBelowLength = agx::Real( 0.5 ) * oldLength + position;
    newAboveLength = oldLength - newBelowLength;
    if ( newBelowLength < 0 || newAboveLength < 0 )
      return nullptr;
    shape->setHeight( newBelowLength );
    return new T( shape->getRadius(), newAboveLength );
  }

  AGXMODEL_EXPORT agxCollide::Shape* cutShape( agxCollide::Shape* shape, agx::Real position, agx::Real& newBelowLength, agx::Real& newAboveLength );
  AGXMODEL_EXPORT agx::Real getShapeLength( const agxCollide::Shape* shape );
  AGXMODEL_EXPORT agx::Real getShapeRadius( const agxCollide::Shape* shape );

  AGX_DECLARE_POINTER_TYPES(GenericBranchEventListener);

  /**
  Generic BranchEventListener throwing exceptions for non-implemented virtual methods.
  This model is the model to inherit from in, e.g, Python or C#.
  */
  class AGXMODEL_EXPORT GenericBranchEventListener : public Tree::BranchEventListener
  {
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Construct branch event listener to listen to certain events happening in a Tree.
      Not valid to use this branch event listener in more than one Tree.
      */
      GenericBranchEventListener();

      /**
      Event when branch has been configured with position, orientation and constraint 
      to parent (if present). GenericBranchEventListener::onCreate will throw an error 
      if called - this method has to be overridden.
      \param branch - the newly configured branch
      */
      virtual void onCreate(agxModel::Tree::Branch* branch) override;

      /**
      Event when branch connection is under high load. 
      GenericBranchEventListener::onHighLoad will throw an error if called - this method
      has to be overridden.
      \param branch - branch under high load
      */
      virtual void onHighLoad(agxModel::Tree::Branch* branch) override;

      /**
      Clone method used by cut. GenericBranchEventListener::clone will throw an error if 
      called - this method has to be overridden.
      \return a clone of this branch event listener
      */
      virtual agxModel::GenericBranchEventListener* clone() override;

      // For getTree to be accessible in python, we need to make it public
      using Tree::BranchEventListener::getTree;

    protected:
      virtual ~GenericBranchEventListener();

      DOXYGEN_END_INTERNAL_BLOCK()
  };
}

#endif // AGXMODEL_TREE_H
