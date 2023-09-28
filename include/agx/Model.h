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

#ifndef AGX_MODEL_H
#define AGX_MODEL_H

#include <agx/Component.h>
#include <agx/Namespace.h>
#include <agxIO/Environment.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{
  AGX_DECLARE_POINTER_TYPES(Model);


  /**
  A model is an abstract representation of the class of an agx::Object. The models
  are organized in a hierarchy and supports inheritance.
  For example: Physics.RigidBody.IntegratePositions -> Kernel -> Task -> Component -> Object
  */
  class AGXCORE_EXPORT Model : public Component
  {
  public:
    static Model *ClassModel();

    /**
    \return The root model.
    */
    static agx::Namespace *root();

    using Object::getPath;

    /**
    \return The model path with dot-separator.
    For example: path/to/Components/Physics/Foo/Particle.agxEntity -> Physics.Foo.Particle
    */
    static agx::Path getPath(agx::TiXmlElement *element, agxIO::Environment::Type pathType = agxIO::Environment::RUNTIME_PATH);
    static agx::Path getPath(agx::TiXmlElement *element, agx::Name& implementation, agxIO::Environment::Type pathType = agxIO::Environment::RUNTIME_PATH);
    static agx::Path getPath(const agx::String& filePath, agxIO::Environment::Type pathType = agxIO::Environment::RUNTIME_PATH);

    /**
    \return The component root directory
    For example: path/to/Components/Physics/Foo/Particle.agxEntity -> path/to/Components
    */
    static agx::String getFileSystemRoot(agx::TiXmlElement *element, agxIO::Environment::Type pathType = agxIO::Environment::RUNTIME_PATH);

    /**
    \return The namespace of the model for example path/to/Components/Physics/Foo/Particle.agxEntity -> Physics.Foo
    */
    static agx::Path getNamespace(agx::TiXmlElement *element, agxIO::Environment::Type pathType = agxIO::Environment::RUNTIME_PATH);
    static agx::Path getNamespace(const agx::String& filePath, agxIO::Environment::Type pathType = agxIO::Environment::RUNTIME_PATH);

    /**
    Locate a model definition file.
    \param modelPath - Path to model, for example Physics.Particle
    \param fullModelPath - Full resolved path of model, expanding input path if is a relative path
    \param filePath - Absolute path to the file on disk
    \param relativeRoot - The relative node from where the input path is specified
    \param envType - When to search for files
    \param fileExtension - for example "agxEntity"
    \return false if model is not found
    */
    static bool getDiskPath(const Path& modelPath, Path& fullModelPath, String& filePath, TiXmlElement *relativeRoot, agxIO::Environment::Type envType, const String& fileExtension);

  public:
    Model(const agx::Name& name, const agx::Name& implementation = agx::Name());
    Model(const agx::Name& name, agx::Model *source);

    /**
    \return The implementation name.
    */
    const agx::Name& getImplementationName() const;

    // name:implementation
    agx::String fullName() const;

    // path:implementation
    agx::String fullPath() const;

    /**
    \return The source model.
    */
    agx::Model *getSource();

    /**
    \return The source model.
    */
    const agx::Model *getSource() const;

    /**
    \return True if the specified model is a source (direct and indirect)
    */
    bool hasSource(const agx::Model *source) const;

    /// \deprecated
    bool hasParent(const agx::Model *parent) const;

    /**
    \return True if the specified model is the same or a parent model.
    */
    bool is(const agx::Model *other) const;

    using Referenced::is;

    /**
    \return The namespace of the model
    */
    agx::Namespace *getNamespace();

    /**
    \return The namespace of the model
    */
    const agx::Namespace *getNamespace() const;

    /**
    The active instances of this model.
    */
    // const ObjectPtrSet& getInstances() const;

    /**
    \return the full inheritance path.
    */
    agx::String fullInheritancePath() const;


    /**
    \return The source path on disk which defines the model.
    */
    const agx::String& getSourcePath() const;

    // eg 'foo.bar:hej' --> path=foo.bar, implementation=hej
    static void parsePathString(const agx::String& pathStr, agx::Path& path, agx::Name& implementation);

    // Create and register a model
    static agx::Model *createModel(const agx::Path& path, const agx::Name& implementation, agx::Model *source);

    virtual void buildNavigationTree(agxJson::Value& eNode) const override;
  public:
    class AGXCORE_EXPORT Loader
    {
    public:
      Loader(const char *name, const char *fileExtension);

      virtual ~Loader();
      virtual Model *load(TiXmlElement *eModel, const Path& ns) = 0;

      const char *getName() const;
      const char *getFileExtension() const;
      void addAlias(const char *alias);

    private:
      const char *m_name;
      const char *m_fileExtension;
    };

    // TODO Make private?
    void setSource(Model *source);
    void setSourcePath(const String& path);
    // static Model *createModel(const String& path, Model *source);

    bool isShutdown() const;
  protected:
    virtual ~Model();

  private:
    friend class Object;
    // void addInstance(Object *instance);
    // void removeInstance(Object *instance);

  private:
    Name m_implementation;
    String m_sourcePath;
    ModelRef m_source;
    // ObjectPtrSet m_instances;

  protected:
    friend class ModelCleanup;
    bool m_isShutdown;
  };







  /* Implementation */
  AGX_FORCE_INLINE const Name& Model::getImplementationName() const { return m_implementation; }

  AGX_FORCE_INLINE Namespace *Model::getNamespace()
  {
    agxAssert(!this->getContext() || dynamic_cast<Namespace *>(this->getContext()));
    return static_cast<Namespace *>(this->getContext());
  }

  AGX_FORCE_INLINE const Namespace *Model::getNamespace() const
  {
    return const_cast<Model *>(this)->getNamespace();
  }

  AGX_FORCE_INLINE Model *Model::getSource() { return m_source; }
  AGX_FORCE_INLINE const Model *Model::getSource() const { return m_source; }
  // AGX_FORCE_INLINE const ObjectPtrSet& Model::getInstances() const { return m_instances; }
  AGX_FORCE_INLINE const String& Model::getSourcePath() const { return m_sourcePath; }
  AGX_FORCE_INLINE bool Model::isShutdown() const { return m_isShutdown; }

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif /* AGX_MODEL_H */
