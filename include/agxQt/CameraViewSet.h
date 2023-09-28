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

#ifndef AGXQT_CAMERAVIEWSET_H
#define AGXQT_CAMERAVIEWSET_H

#include <agxQt/export.h>
#include <agx/Referenced.h>
#include <agx/List.h>
#include <agx/Vec3.h>
#include <agx/Vector.h>

#define DEFAULT_CAMERADESCRIPTION_FOV 30.0

// Forward declaration
namespace osg
{
  class Camera;
}

namespace osgViewer
{
  class Viewer;
}

namespace agxCFG
{
  class ConfigScript;
}

namespace agxQt
{

  class CameraDescription;
  class CameraDescriptionSet;
  AGX_DECLARE_POINTER_TYPES(CameraDescription);
  AGX_DECLARE_POINTER_TYPES(CameraDescriptionSet);

  /**
  * CameraViewSet is a class that handles and manages a set of cameras that the user should be able to switch between in the active scene
  * Different Cameras can be stored to the active set while looking at the scene for future references. This class serves as a container and
  * manager for the different types of views that the user want to store
  *
  * TODO - nextCam.., prevCam.. should reutrn a descirption instead of index. Should only expose index at getCameraDescription().
  */
  class AGXQT_EXPORT CameraDescriptionSet : public agx::Referenced
  {
    typedef agx::List<CameraDescriptionRef> CameraDescriptionList;

  public:
    /// Constructor
    CameraDescriptionSet();

    /// Add/Remove a camera to the camera set
    void addCamera(CameraDescription * camera);
    void removeCamera(CameraDescription * camera);

    /// Get the active camera index in the set
    agx::UInt getActiveCameraIndex();

    /// Increments the active camera index to the next one. If the active camera index is at max, it cycles back to the first index. Returns the active index after increment
    agx::UInt nextCameraDescription();

    /// Decrements the active camera index to the previous one. If the active camera index is at 0, it cycles forward to the last index. Returns the active index after increment
    agx::UInt previousCameraDescription();

    /// Gets the camera description given an index
    CameraDescription* getCameraDescription(const agx::UInt& index);

    /// Returns the active camera description in the set
    CameraDescription* getActiveCameraDescription();

    /// Returns a camera description given a name
    CameraDescription* getCameraDescriptionFromName(const agx::String& name);

    /// Sets the active camera view index. If the index is out of bounds, nothing happens
    void setActiveCameraViewIndex(agx::UInt index);

    /// Check if the camera is contained in the set
    bool containsCameraDescription(const CameraDescription* camera);

    /// Returns the number of camera descriptions in the active set
    agx::UInt getNumberOfCameraDescriptions();

    /// Set active view in manager from name. Returns false if not found, true if found and deleted.
    bool setActiveViewFromName(const agx::String& name);

    /// Remove a loaded camera view given an string identifier. Returns false if not found, true if found and deleted.
    bool removeCameraViewFromName(const agx::String& name);

    /// Tries to rename a camera description given a name identifier. Returns false if no camera description of that name exists in the set
    bool renameCameraDescription(const agx::String& oldname, const agx::String& newName);

    /// Get the names for all loaded views.
    agx::Vector<agx::String> getCameraViewNames();

    /// Print the set
    agx::String toString();

  protected:
    bool getIndexFromCamera(CameraDescription * camera, agx::UInt& index);

    /// Destructor
    virtual ~CameraDescriptionSet();


    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    CameraDescriptionList   m_cameraDescriptionContainer;
    agx::UInt               m_activeCameraIndex;
  };

  /**
  * Class that describes an abstract camera and it's properties. Can be converted to a specific type of camera that can be set to osg::Viewer
  */
  class AGXQT_EXPORT CameraDescription : public agx::Referenced
  {
  public:
    /// Constructor
    CameraDescription();

    /// Set Camera information
    void setInformation(const agx::Vec3 &eye, const agx::Vec3& center, const agx::Vec3& up, agx::Real fov);

    /// Set name of the camera description
    void setName(const agx::String& name);
    const agx::String& getName() const;

    /// Checks if the camera description is valid
    bool isValid() const;

    /// Get center
    const agx::Vec3& getCenter() const;

    /// Get up vector
    const agx::Vec3& getUp() const;

    /// Get Eye vector
    const agx::Vec3& getEye() const;

    /// Get Field of view
    const agx::Real getFov() const;

    /// Formats the description to a string
    const agx::String toString() const;

  protected:

    /// Destructor
    virtual ~CameraDescription();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Vec3 m_center;
    agx::Vec3 m_eye;
    agx::Vec3 m_up;
    agx::Real m_fov;

    /// Name of the camera description
    agx::String m_name;
  };

  /* Implementation */
  AGX_FORCE_INLINE const agx::Vec3& CameraDescription::getCenter() const { return m_center; }

  AGX_FORCE_INLINE const agx::Vec3& CameraDescription::getUp() const { return m_up; }

  AGX_FORCE_INLINE const agx::Vec3& CameraDescription::getEye() const { return m_eye; }

  AGX_FORCE_INLINE const agx::Real CameraDescription::getFov() const { return m_fov; }

  AGX_FORCE_INLINE void CameraDescription::setName(const agx::String& name) { m_name = name; }

  AGX_FORCE_INLINE const agx::String& CameraDescription::getName() const { return m_name; }

  /**
  * Manager class that acts as a wrapper container for the CameravViewset class for OSG based view systems
  */
  class AGXQT_EXPORT OsgCameraViewManager : public agx::Referenced
  {
  public:
    /// Default constructor
    OsgCameraViewManager();

    /// Stores the current view in osg viewer
    void storeCurrentView(osgViewer::Viewer *viewer);

    /// Sets the active view in the manager an osgViewer
    void setActiveView(osgViewer::Viewer * viewer);

    /// Toggles the next view in stored views and sets it to an osgViewer
    void setNextView(osgViewer::Viewer * viewer);

    /// Toggles a previous view in stored views and sets it to an osgViewer
    void setPreviousView(osgViewer::Viewer * viewer);

    /// Set active view in manager from name. Returns false if not found, true if found and deleted.
    bool setActiveViewFromName(const agx::String& name);

    /// Remove a loaded camera view given an string identifier. Returns false if not found, true if found and deleted.
    bool removeCameraViewFromName(const agx::String& name);

    /// Returns the active view name
    agx::String getActiveCameraViewName() const;

    /// Get the names for all loaded views.
    agx::Vector<agx::String> getCameraViewNames();

    /// Store the active configuration to a file
    bool storeViewsetAsCFGFile(const agx::String& filename);

    /// Restores the a view set from a .cfg file. Will overwrite the current view set in the manager
    bool restoreViewsetFromCFGFile(const agx::String& filename);

    /// Returns the internal structure of the camera view set
    CameraDescriptionSet* getCameraDescriptionSet();

    /// Utility method for converting osgViewer views to CameraViewDescriptions
    static CameraDescription* getCameraDescriptionFromOsgViewer(osgViewer::Viewer * viewer);

  protected:
    void setViewOnOSGViewer(CameraDescription* c, osgViewer::Viewer* viewer);

    /// Tries to parse a camera state from the current position in the config script
    bool parseCameraState(agxCFG::ConfigScript * cfg, CameraDescription *c);

    /// Stores the camera description in the in the current place in the configure script
    bool storeCameraState(agxCFG::ConfigScript * cfg, CameraDescription *c);

    /// Default destructor
    virtual ~OsgCameraViewManager();


    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    CameraDescriptionSetRef m_cameraSet;
  };

  AGX_DECLARE_POINTER_TYPES(OsgCameraViewManager);

  /*Implementation*/
  AGX_FORCE_INLINE CameraDescriptionSet* OsgCameraViewManager::getCameraDescriptionSet(){ return m_cameraSet.get(); }

}

#endif
