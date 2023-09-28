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

/*
Add new modification:
  1. Execute "agxArchive -g" to generate a random string that should be used here.
  2. Add the new modification name to OutputArchive::addModifications.

To save space, it is only 9 letters long. Should be plenty, but to be sure, do
a search in this file first to ensure there are no collisions.
*/

//#define AGXSTREAM_ARCHIVE_MODIFICATION_ATESTTHING "864419af" // Some random generated string

#define AGXSTREAM_ARCHIVE_MODIFICATION_WIRE_CABLE_HYDRO "6d403ff8" // Serialization of Wire and Cable data in WindAndWaterController
#define AGXSTREAM_ARCHIVE_MODIFICATION_CABLE_RESOLUTION_IN_INIT_REPORT "a3479399" // Serialization of selected cable resolution in initialization report.
#define AGXSTREAM_ARCHIVE_MODIFICATION_ELASTIC_CONTACT_MATERIAL "90446b72" // Serialization of Elastic Rest Length in Material (Bulk and Contact)
#define AGXSTREAM_ARCHIVE_MODIFICATION_MEASUREMENTSENSOR_STATE "614a9c18" // Serialization of MeasurementSensor state
#define AGXSTREAM_ARCHIVE_MODIFICATION_OBSERVERSTORAGE_REMOVED "d148f4e3" // Removed observer storage and it will no longer be serialized, nor can it be read.
#define AGXSTREAM_ARCHIVE_MODIFICATION_PACKED_POWERLINE_BODIES "ab481cbf" // Serialization changes dues to support for body packing of power line dimensions.
#define AGXSTREAM_ARCHIVE_MODIFICATION_WATER_FLOW_GENERATOR "6040ea6d" // Serialization of water flow generator
#define AGXSTREAM_ARCHIVE_MODIFICATION_CLUTCH_STORE_RESOLUTION "59491598" // Clutch was stored wrongly and had to be fixed
#define AGXSTREAM_ARCHIVE_MODIFICATION_TRANSLATIONAL_ACTUATOR_CONNECTOR "b84edcb2" // Serialization of TranslationalActuators' connector.
#define AGXSTREAM_ARCHIVE_MODIFICATION_ACTUATOR_CONSTRAINT_OWNING_ACTUATOR "2544acf1" // Serialization of Actuator owning an ActuatorConstraintImplementation.
#define AGXSTREAM_ARCHIVE_MODIFICATION_MERGE_SPLIT_THRESHOLDS "fe4dee6d" // Introduced thresholds in merge split algorithms.
#define AGXSTREAM_ARCHIVE_MODIFICATION_RIGIDBODYEMITTER_INTERNALDATA "3847cd80" // Added internal data to RigidBodies from RigidBodyEmitter
#define AGXSTREAM_ARCHIVE_MODIFICATION_DEFORMABLE1D_COMPONENT "cf43dabe" // Added Deformable1DComponent.
#define AGXSTREAM_ARCHIVE_MODIFICATION_PARTICLECONTACTDUMPER_EXPORTONCE "6f474e7b" // Added exportOnce flag to ParticleContactDumper.
#define AGXSTREAM_ARCHIVE_MODIFICATION_OBSERVERFRAME_ENABLED "0343cea9" // Renamed serialized field because mismatch between what is loaded and stored
#define AGXSTREAM_ARCHIVE_MODIFICATION_WIRE_CONTACTS "ad4b3a6c" // Serialization changes due to introduction of the ShapeContactNode and the WireShapeContactController.
#define AGXSTREAM_ARCHIVE_MODIFICATION_WIRE_FRICTION_COEFFICIENTS "3b4c3a2f" // Wire friction coefficients introduced in contact material.
#define AGXSTREAM_ARCHIVE_MODIFICATION_LINKED_CABLE "4f49f119" // Cable based on LinkedStructure instead of Deformable1D.
#define AGXSTREAM_ARCHIVE_MODIFICATION_MULTI_GEAR "9143075d" // Serialization for MultiGear.
#define AGXSTREAM_ARCHIVE_MODIFICATION_CABLE_DAMAGE "7a4a96ed" // Added cable damage.
#define AGXSTREAM_ARCHIVE_MODIFICATION_EC_NAME "5d486aad" // Elementary constraint store/restore name.
#define AGXSTREAM_ARCHIVE_MODIFICATION_OFM_FLAGS "64492057" // Flags in ConstantNormalForceOrientedBoxFrictionModel.
#define AGXSTREAM_ARCHIVE_MODIFICATION_WATER_WRAPPER "994f6ef1" // Introduced custom water wrappers in the WindAndWaterController.
#define AGXSTREAM_ARCHIVE_MODIFICATION_WATER_WRAPPER_DATA "264d9279" // Serialize geometry and water flow generator of WaterWrapper.
#define AGXSTREAM_ARCHIVE_MODIFICATION_OBSERVERFRAME_RIGIDBODY "48409c21" // Serialize m_rigidBody in ObserverFrame
#define AGXSTREAM_ARCHIVE_MODIFICATION_AMOR_PARALLELIZATION "de4d4203" // Merge split refactorization.
#define AGXSTREAM_ARCHIVE_MODIFICATION_TRACK_HINGE_PARAMETERS "794fc7e1" // agxVehicle::Track - possible to set hinge parameters per DOF.
#define AGXSTREAM_ARCHIVE_MODIFICATION_WIRE_COLLISION_GROUPS "284d66b6" // Added named collision groups for wire.
#define AGXSTREAM_ARCHIVE_MODIFICATION_CABLE_PATH_ROUTE "7a4d21b4" // Added PathRoute for cable routing.
#define AGXSTREAM_ARCHIVE_MODIFICATION_PARTICLE_STATE "4c43cd59" // Added particle state to serialization.
#define AGXSTREAM_ARCHIVE_MODIFICATION_MERGE_IGNORE_FILTER "374333dc" // Added MergeIgnoreFilter.
#define AGXSTREAM_ARCHIVE_MODIFICATION_NO_HINGE_IN_HOLO_GEAR "8249c4ba" // Removed m_hinge from HolonomicGear.
#define AGXSTREAM_ARCHIVE_MODIFICATION_MERGE_SPLIT_FILTER "03412991" // Store merge ignore filter from merge split handler.
#define AGXSTREAM_ARCHIVE_MODIFICATION_ADDINGBLOCK_ADDED_MASS_INTERACTION "644e097d" // Added begin/end section for writing block to handle XML storage
#define AGXSTREAM_ARCHIVE_MODIFICATION_EMITTER_COLLISION_GROUPS "0x7813067e" // Added collision groups to emitter
#define AGXSTREAM_ARCHIVE_MODIFICATION_ELECTRIC_MOTOR "ad45139c" // Added electric motor
#define AGXSTREAM_ARCHIVE_MODIFICATION_COLOR_AS_VEC4 "f248a61a" // RenderState is now Vec4f
#define AGXSTREAM_ARCHIVE_MODIFICATION_SOLVER_VALUES "804c887b" // Restoring solver settings.
#define AGXSTREAM_ARCHIVE_MODIFICATION_CONTACT_WARMSTARTING "cd41aac4" // Added contact warm starting enable/disable.
#define AGXSTREAM_ARCHIVE_MODIFICATION_COMBUSTION_ENGINE "4648f508" // Added combustion engine
#define AGXSTREAM_ARCHIVE_MODIFICATION_FRICTION_CONTROLLER "7b49baab" // Added 1D elementary constraint friction controller.
#define AGXSTREAM_ARCHIVE_MODIFICATION_BOOLEAN_SENSOR "c3467288" // Added collider early out flag for sensors.
#define AGXSTREAM_ARCHIVE_MODIFICATION_POWERLINE_COMPONENT_NAME "ea418d98" // Added name to power line components.
#define AGXSTREAM_ARCHIVE_MODIFICATION_COMBUSTION_ENGINE_STARTER "f144687f" // Added combustion engine starter.
#define AGXSTREAM_ARCHIVE_MODIFICATION_DEPRECATION_API_V_2_26_0_0 "0f45b0e5" // Various deprEcation of API
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN "434b6edb" // Added serialization for agxTerrain::Terrain.
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_SUBMERGED_FRACTION "0x1c76ab46" // Added submerged cutting edge fraction
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_SHOVEL_FLAGS "6441b926" // Added shovel flags
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_VELOCITYTABLE "274692fd" // Changed voxel velocity storage to agx::HashTable
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXSDK_ASSEMBLY_STORE_OBSERVERFRAMES "96496ba2" // Assembly can carry ObserverFrames
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXSDK_ASSEMBLY_STORE_EMITTER_PARTICLESYSTEMS "464052d3" // Assembly must store emitters and particle systems
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_EXCAVATION_CONTACT_SETTINGS "834b52e7" // Added struct for excavation settings
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_ADHESION_OVERLAP_FACTOR "0f4935bb" // Added terrain material bulk parameter adhesion overlap factor
#define AGXSTREAM_ARCHIVE_MODIFICATION_LOAD_PARTICLE_JOURNAL_FLAG "e0442ba8" // Added flag if a particle journal has been loaded or not in EventSensor
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_EXCAVATION_STIFFNESS_MULTIPLIER "20452fef" // Added stiffness multiplier for contacts between shovel and aggregates
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_TERRAINMATERIAL_LAST_ERROR "6a49c403" // Added last error string for serialization
#define AGXSTREAM_ARCHIVE_MODIFICATION_ENERGYMANAGER "e144d2ef" // Added EnergyManager
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_ADHESION_OVERLAP_PARTICLE_PROP "1c490abb" // Moved particle adhesionOverlap to ParticleProperties
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_SHOVEL_CONTACT_MATERIALS "324c8db4" // Serialization of shovel-aggregate contact materials
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_PROPERTIES_AVALANCHE_THRESHOLD "cb44c3f9" // Avalanche error threshold
#define AGXSTREAM_ARCHIVE_MODIFICATION_TANGENTIAL_RESTITUTION "1a45d70b" // Tangential restitution, ContactMaterialEntity::restitution changed from Real to Vec3.
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_NO_MERGE "b8467af9" // Added m_noMerge and m_noMergeEdgeMargin fields to terrain class
#define AGXSTREAM_ARCHIVE_MODIFICATION_GRANULAR_ROTATION "2f476a7b" // Granular rotational buffer
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_SHOVEL_CONTACT_THRESHOLD "6e409391" // Added bottom contact threshold to shovel to control how contacts are created with the underside
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_PROPERTIES_DEFORMER_ACTIVATION_SPEED "5341b5df" // Added deformer activation speed when static mass is converted to dynamic
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_CONTACT_DEPTH_ANGLE_THRESHOLD "ae4bc4ac" // Added angle threshold to excavation properties
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_DILATANCY_PROPERTIES "8941510c" // Added dilatancy angle properties in CompactionProperties
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_PROPERTIES_PARTICLE_GROWTH_RATE_FACTOR "0c417158" // Added particle growth rate to TerrainProperties
#define AGXSTREAM_ARCHIVE_MODIFICATION_SOLVER_PPGS_ITERATIONS "f14c0b2e" // Solver PPGS resting iterations
#define AGXSTREAM_ARCHIVE_MODIFICATION_COMBUSTION_ENGINE_STABILITY_FIX "dd416626" // Change engine solver and fix stability problems
#define AGXSTREAM_ARCHIVE_MODIFICATION_NEW_TORQUE_CONVERTER "904156fd" // Replaced old torque converter with new one
#define AGXSTREAM_ARCHIVE_MODIFICATION_PARTICLE_COLLISION_GROUPS "824000b5" // Store per particle collision groupset info
#define AGXSTREAM_ARCHIVE_MODIFICATION_BUOYANCY_SCALING "d646aff6" // Add hydrodynamic parameter for buoyancy
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_GRADUAL_MERGE "86435b72" // Variables concerning gradual merge of soil particles to agxTerrain
#define AGXSTREAM_ARCHIVE_MODIFICATION_DOT2DATA_REFERENCE_BODY "d749007b" // Allow for using second body as reference
#define AGXSTREAM_ARCHIVE_MODIFICATION_FRICTION_CONTROLLER_STATIC_BOUND "9643ba40" // Allow setting the minimum force bound of friction controllers to simulate static friction
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_PROPERTIES_PARTICLE_SCALING_FACTOR "f6455f00" // Added particle scaling factor to TerrainProperties
#define AGXSTREAM_ARCHIVE_MODIFICATION_LINEARIZATION_ON_CONSTRAINT "39492e7d" // Enable linearization flag moved from LockJoint to Constraint.
#define AGXSTREAM_ARCHIVE_MODIFICATION_NON_COMPATIBLE_GRID_CHANGES "7f4ad7f4" // OpenVDB no longer being used by agxTerrain / vdbgrid
#define AGXSTREAM_ARCHIVE_MODIFICATION_SHAFT_ACCUMULATED_ANGLE "024b1e09" // Shafts accumulate the rotation angle
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXSDK_ASSEMBLY_STORE_MERGEDBODIES "9b4bddcd" // Assembly can carry MergedBody
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_DELTA_REPOSE "7e4fd048" // Add delta repose to BulkProperties in Terrain mateiral
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_COLUMN_GRID_RESTORE_FIX "c3451e08" // Fix for restore of columngrid in Terrain
#define AGXSTREAM_ARCHIVE_MODIFICATION_COLLISION_GROUP_MANAGER "894071db" // Added more extensive serialization for CollisionGroupManager to enable journal recording
#define AGXSTREAM_ARCHIVE_MODIFICATION_RIGIDBODYMODEL_BODYTEMPLATE_NAME "ee41d5ca" // changed body template name bug from "particleRadius" to "bodyTemplate" in serialization
#define AGXSTREAM_ARCHIVE_MODIFICATION_COLLISION_GROUP_MANAGER_ORDER "14414766" // changed order of serialization of collision manager
#define AGXSTREAM_ARCHIVE_MODIFICATION_STORE_RENDERMATERIAL_NAME "23466ca5" // Fix so that RenderMaterial name is stored so it can be sent over websocket protocole
#define AGXSTREAM_ARCHIVE_MODIFICATION_TWOBODYTIRE_REMOVE_VALID "754dc3f4" // Removed serialization of m_valid in TwoBodyTire
#define AGXSTREAM_ARCHIVE_MODIFICATION_EMITTER_STATE "074366c1" // Added "state" to Emitter serialization
#define AGXSTREAM_ARCHIVE_MODIFICATION_AGXTERRAIN_EXCAVATION_SETTINGS_FIX "2e47e065" // Fixed serialization for excavation settings
#define AGXSTREAM_ARCHIVE_MODIFICATION_TERRAIN_DEFORMATION_TOGGLE "ac4cf59a" // Added a deformation enable flag to terrain properties
#define AGXSTREAM_ARCHIVE_MODIFICATION_TERRAIN_TOOL_COLLECTION_STATE_FLAGS "3946ed8b" // Add state flags for TerrainToolCollection
#define AGXSTREAM_ARCHIVE_MODIFICATION_EMITTER_FIXED_PLACEMENT_DATA "ca422f0e" // Added fixed rotation and position offset to emitter
#define AGXSTREAM_ARCHIVE_MODIFICATION_RIGIDBODY_EMITTER_MODEL_OBSERVER "d94a9b7a" // Added placement ObserverFrame to RigidBodyEmitter::DistributionModel
#define AGXSTREAM_ARCHIVE_MODIFICATION_NEW_COMBUSTION_ENGINE "e24f37b8" // A new implementation of the CombustionEngine
#define AGXSTREAM_ARCHIVE_MODIFICATION_REMOVE_POISSONS_RATIO "9d47de6e" // Removed BulkMaterial::setPoissonsRatio