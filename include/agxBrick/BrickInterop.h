#pragma once


#include <agxBrick/agxBrick_export.h>
#include <agx/String.h>
#include <agx/Referenced.h>
#include <agxSDK/Simulation.h>

namespace agxBrick {

  /**
   * Initialize the embedded Brick .NET runtime.
   * \param brickRoot Path to the root Brick directory
   * \param logLevel Logging level for the Brick runtime
   * \param initPythonnet Specify if Brick should initialize pythonnet bridge
   * \return True if successful
   */
  AGXBRICK_EXPORT bool init(const agx::String& brickRoot, const agx::String& logLevel = "info", bool initPythonnet = true);

  /// \return True if Brick runtime is initialized
  AGXBRICK_EXPORT bool isInitialized();

  /**
   * Load a Brick model from file into a target simulation
   * \param target The target simulation for loading the model
   * \param filePath The filepath to the Brick model
   * \param model The name of the model in the file
   */
  AGXBRICK_EXPORT bool loadBrickFile(agxSDK::Simulation *target, const agx::String& filePath, const agx::String& model);

  /**
   * Load a Brick model from file into a target assembly
   * \param target The target simulation for loading the model
   * \param filePath The filepath to the Brick model
   * \param model The name of the model in the file
   */
  AGXBRICK_EXPORT bool loadBrickFile(agxSDK::Assembly*   target,
                                     agxSDK::Simulation* simulation,
                                     const agx::String&  filePath,
                                     const agx::String&  model);

  AGXBRICK_EXPORT bool syncBrickInputs( agxSDK::Simulation* target );
  AGXBRICK_EXPORT bool syncBrickOutputs( agxSDK::Simulation* target );

  /// Parsing of composite brick model path syntax, eg `path/to/modelfile.yml:modelname`
  AGXBRICK_EXPORT bool parseBrickModelLoadingDeclaration(const agx::String& declaration, agx::String& filepath, agx::String& modelname);

  // Interop of SWIG.Python and Brick.Pythonnet
  AGXBRICK_EXPORT bool registerAgxSimulation(agxSDK::Simulation *target, const agx::String& name);
  AGXBRICK_EXPORT bool unregisterAgxSimulation(const agx::String& name);
  AGXBRICK_EXPORT agxSDK::Simulation* getRegisteredAgxSimulation(const agx::String& name);

  // Try and locate the Brick runtime resources
  AGXBRICK_EXPORT agx::String locateBrickRootFromEnvironment();

  AGX_DECLARE_POINTER_TYPES( BrickInterop );
  class AGXBRICK_EXPORT BrickInterop : public agx::Referenced {
  public:
    static BrickInterop* instance();

  public:
    virtual bool init(const agx::String& agxBrickDistDir, const agx::String& logLevel, bool initPythonnet) = 0;
    virtual void setLogLevel(const agx::String& logLevel) = 0;
    virtual bool loadBrickFile(agxSDK::Simulation *target, const agx::String& filePath, const agx::String& model) = 0;
    virtual bool loadBrickFile(agxSDK::Assembly*   target,
                               agxSDK::Simulation* simulation,
                               const agx::String&  filePath,
                               const agx::String&  model) = 0;
    virtual bool syncBrickInputs( agxSDK::Simulation* target ) = 0;
    virtual bool syncBrickOutputs( agxSDK::Simulation* target ) = 0;

    // Allow sharing pythonnet instances
    virtual bool registerAgxSimulation(agxSDK::Simulation *target, const agx::String& name) = 0;
    virtual bool unregisterAgxSimulation(const agx::String& name) = 0;
    virtual agxSDK::Simulation* getRegisteredAgxSimulation(const agx::String& name) = 0;

  protected:
    // virtual ~BrickInterop() override = 0;
    virtual ~BrickInterop() override {};
  };

  // Will be called from C# application when launching from dotnet directly, instead of the embedded dotnet runtime
  AGXBRICK_EXPORT void setInteropInstance(BrickInterop *instance);


  // To be used by non-embedded C# runtime
  class AGXBRICK_EXPORT BrickInteropHost {
  public:
    static agxSDK::Simulation* createSimulation();
  private:
    BrickInteropHost() = default;
  };

  // Special struct to pass .NET implementation back to C++ during bootstraping
  struct BrickInteropInitArgs {
    BrickInterop* instance;
  };
}
