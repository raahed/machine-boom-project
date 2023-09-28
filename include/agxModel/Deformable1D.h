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

#ifndef AGXMODEL_DEFORMABLE1D_H
#define AGXMODEL_DEFORMABLE1D_H

#include <agxModel/Deformable1DNode.h>
#include <agxModel/Deformable1DRoute.h>
#include <agxModel/Deformable1DIterator.h>
#include <agxModel/Deformable1DGeometryFactory.h>
#include <agxModel/Deformable1DBulkProperties.h>
#include <agxModel/Deformable1DComponent.h>


namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( Deformable1D );

  /**
  One dimensional deformable construction, i.e., objects attached successively. A
  Deformable1D may be used to simulate elasto-plastic cables, wires, beams etc.
  */
  class AGXMODEL_EXPORT Deformable1D : public agxModel::Tree
  {
    public:
      /**
      Construct given route.
      \param route - route to initialize this object
      */
      Deformable1D( agxModel::Deformable1DRoute* route, agxModel::Deformable1DGeometryFactory* geometryFactory );

      /**
      \return the begin node, i.e., the first element in the structure
      */
      agxModel::Deformable1DNode* getBegin() const;

      /**
      \return the begin iterator
      */
      agxModel::Deformable1DIterator getBeginIterator() const;

      /**
      \return Iterator to the last element in the structure.
      */
      agxModel::Deformable1DIterator findLastIterator() const;

      /**
      Attempt an initialization with the current settings.
      If the Deformable1D has been initialized already then nothing is done and
      the old initialization report is returned.
      \return Initialization report describing the quality of the found route.
      */
      agxModel::Deformable1DInitializationReport tryInitialize();

      /**
      \return the initialization report
      */
      agxModel::Deformable1DInitializationReport getInitializationReport() const;

      /**
      \return the route algorithm
      */
      agxModel::Deformable1DRoute* getRoute() const;

      /**
      Assign new bulk properties object. If null, a default bulk properties
      object will be created.
      \param bulkProperties - new bulk properties
      */
      void setBulkProperties( agxModel::Deformable1DBulkProperties* bulkProperties );

      /**
      \return the bulk properties
      */
      agxModel::Deformable1DBulkProperties* getBulkProperties() const;

      /**
      \return geometry factory used to create the geometries
      */
      agxModel::Deformable1DGeometryFactory* getGeometryFactory() const;


      /**
      Add another component governing the behavior of this Deformable1D.
      \param component - The component that this Deformable1D should have.
      */
      void addComponent(agxModel::Deformable1DComponent* component);

      /**
      Removed the given component from this Deformable1D. Has no effect if the
      given component is not part of this Deformable1D.

      \param component - The component to remove.
      \return True if the component was removed. False otherwise.
      */
      bool removeComponent(agxModel::Deformable1DComponent* component);

      /**
      \return The number of components held by this Deformable1D.
      */
      agx::UInt getNumComponents() const;

      /**
      Gives access to the Deformable1DComponent held by this Deformable1D.
      \return The Deformable1DComponent stored at the given index, or nullptr.
      */
      agxModel::Deformable1DComponent* getComponent(agx::UInt index);
      const agxModel::Deformable1DComponent* getComponent(agx::UInt index) const;

      /**
      Calculates the number of nodes present in current structure. If not initialized
      the number of nodes is 0 (i.e., the route nodes aren't included).
      \return the number of nodes
      */
      agx::UInt calculateNumberOfNodes() const;


      /**
      Make the current relative positions of all nodes the rest state. That is,
      when this call returns no constraint in the Deformable1D will be violated.
      Constraints that are part of an attachment on a node in the cable are not
      affected.

      Should only be called after the Deformable1D has been initialized.

      \return True if the Deformable1D was initialized. False otherwise.
      */
      bool rebind() const;


      virtual void clear() override;


      /**
      \param node - node to find iterator for
      \return iterator to the given node
      */
      static agxModel::Deformable1DIterator getIterator( agxModel::Deformable1DNode* node );

      /**
      \return true if the iterator is begin of a Deformable1D structure
      */
      static agx::Bool isBeginIterator( agxModel::Deformable1DIterator it );


      /**
      \return The Deformable1D that the given body is part of, or nullptr if not part of any Deformable1D.
      */
      static agxModel::Deformable1D* getDeformable(agx::RigidBody* body);

    public:
      /**
      Initialize this object given the current route. This method will be called
      automatically when this Deformable1D is added to a simulation - so there's
      no need to explicitly call this method.
      \return true if initialization process were successful
      */
      virtual bool initialize() override;

      void attachmentAdded(agxModel::Deformable1DNode* node, agxModel::NodeAttachment* attachment);

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Deformable1D );

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1D();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1D();

      /**
      \return a clone of this object
      */
      virtual agxModel::Tree* clone() override;

      /**
      Removes all constraints, rigid bodies, geometries etc added during initialize.
      */
      virtual void removeNotification( agxSDK::Simulation* simulation ) override;
      using agxSDK::Assembly::removeNotification;

      virtual void pre() override;
      virtual void post() override;

      friend class NodeListener;
      /**
      Called when a node is created.
      */
      virtual void onCreate( agxModel::Deformable1DNode* node );

    protected:
      void prepareForSolve( agxModel::Deformable1DNode* node ) const;

    protected:
      Deformable1DRouteRef m_route;
      Deformable1DGeometryFactoryRef m_geometryFactory;
      Deformable1DBulkPropertiesRef m_bulkProperties;
      Deformable1DInitializationReport m_initializationReport;

      Deformable1DComponentRefVector m_components;
  };
}

#endif // AGXMODEL_DEFORMABLE1D_H
