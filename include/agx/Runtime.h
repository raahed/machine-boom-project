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

#include <agx/macros.h>
#include <agx/config/AGX_LAAS_ENABLE.h>

#ifdef STANDALONE_RUNTIME

#ifdef AGX_LAAS_ENABLE
#undef AGX_LAAS_ENABLE
#endif
#define AGX_LAAS_ENABLE() 0

#include <map>
#include <vector>
#include <string>

#else

#include <agx/agx.h>
#include <agx/Singleton.h>
#include <agx/String.h>
#include <agx/HashTable.h>
#include <agxIO/win32Tools.h>

#endif


namespace agx {

  DOXYGEN_START_INTERNAL_BLOCK()

  #ifdef STANDALONE_RUNTIME
  typedef std::string String;
  #endif
  DOXYGEN_END_INTERNAL_BLOCK()


  /**
  Struct holding information about a Unique Id.
  */
  typedef struct UniqueIdStruct_ {
    agx::String id;
    agx::String raw;
    agx::String comment;
    unsigned int type;
  } UniqueIdStruct;


  /**
  \brief The Runtime class will read a license file upon creation
         and check what configuration is allowed to run.

  The license file should be placed where it can be found by
  agxIO::Environment via the RESOURCE_PATH or in a per-user
  directory as described in the user manual.
  */
  #ifdef STANDALONE_RUNTIME
  class Runtime {
  #else
  class AGXPHYSICS_EXPORT Runtime : public agx::Singleton {
  #endif

    public:

      enum HardwareTypes {
        NIC = 1,
        USB = 2,
        HDD = 3
      };


      #ifdef STANDALONE_RUNTIME
      typedef std::vector< UniqueIdStruct > UniqueIdVector;
      #else
      typedef agx::Vector< UniqueIdStruct > UniqueIdVector;
      #endif

      /// Singleton instance method.
      static Runtime* instance();

      /// \return true if this singleton is initialized (instance would then return != 0)
      static bool isInitialized();


      /**
      Utility method using agxStream::MD5Checksum::getMD5
      */
      static agx::String generateMD5( const agx::String& deviceID );


      #if AGX_LAAS_ENABLE()

      /** \name Online License System */
      /// \{


      /**
      Load a License file that was created by activating AGX.

      Some license types needs to be refreshed at a certain time
      interval and this file should be user writable to avoid
      the need for any manual handling or that the license must
      be refreshed every time AGX is initialized.
      */
      bool loadLicenseFile( const agx::String& licPath, bool forceRefresh = false );


      /**
      Load a license but provided as a string instead of read from a file.

      This function is similar to loadLicenseFile, but takes the file contents
      as parameter instead of the filename.

      Some license types are refreshed from online source when
      loading and to check if this has happened, isLicenseRefreshed
      should be used after this function and readEncryptedLicense
      if needed.
      */
      bool loadLicenseString( const agx::String& licenseContents, bool forceRefresh = false );

      /**
      Check if license was refreshed upon loading.
      */
      bool isLicenseRefreshed();

      /**
      Read the encrypted license, normally the contents
      from the license file.

      If no license is loaded, this will return the empty string
      */
      agx::String readEncryptedLicense();


      /**
      Perform online activation of AGX.

      On successful activation, the license will also be loaded.

      \return true on successful activation.
      */
      bool activateAgxLicense( int licenseId, const agx::String& activationPassword );


      /**
      Perform online activation of AGX.

      The received license will be stored to a file locally.
      Any previous parsed license information will be cleared.
      On successful activation, the license will also be loaded.

      \return true on successful activation.
      */
      bool activateAgxLicense( int licenseId, const agx::String& activationPassword, const agx::String& outputLicenseFile );


      /**
      Generates an activation request for a machine that does not have Internet access.
      */
      agx::String generateOfflineActivationRequest(int licenseId, const agx::String& activationPassword );

      /**
      Processes the response from a manual activation request.
      \return True if the activation response could be processed successfully and a valid license received
      */
      bool processOfflineActivationRequest(const agx::String& activationResponse);

      /**
      Deactivate the currently loaded license if the license type allows it.
      On failure, getStatus() will contain information about the failure
      \return True if successfully deactivated the license.
      */
      bool deactivateAgxLicense();



      /**
      Try to open a network session using the specified floating license.

      \note  This usecase is for large scale installations e.g. on a cluster
             or some cloud setup. This is not compatible with a normal
             activated license.

      Internet access is needed for the network session to stay valid via
      periodic background polling. If the polling and retries fail, the
      license status can change to invalid.

      \return True if a network session could be started and a seat allocated.
      */
      bool openNetworkSession(const agx::String& licenseFile = "floating.lfx" );

      /**
      Close the active network session.
      There will not be a valid license loaded after this call.

      \see openNetworkSession
      */
      bool closeNetworkSession();

      /**
      Query licensetype.
      \return True if a floating license is in use.
      */
      bool isFloatingLicense() const;


      /**
      Encrypt license id and password for runtime activation. The encryption will
      fail if \p applicationFilename isn't found in agxIO::Environment::RESOURCE_PATH,
      returning an empty string.
      \param runtimeLicenseId - License id of the runtime license.
      \param runtimeLicensePassword - License password of the runtime license.
      \param applicationFilename - Filename (including relative path) to a file in the application
                                   that doesn't change (e.g., an executable or library).
      \return encrypted runtime activation data
      */
      agx::String encryptRuntimeActivation( int runtimeLicenseId,
                                            const agx::String& runtimeLicensePassword,
                                            const agx::String& applicationFilename ) const;

      /**
      Perform online activation of an encrypted runtime license. If successful, the
      license will be written to \p outputLicenseFile.
      \param encryptedRuntimeData - Encrypted data generated from \p encryptRuntimeActivation.
      \param outputLicenseFile - Resulting license file if the online activation is successful.
      \return true if activation is successful, otherwise false (check \p getStatus()) for errors
      */
      bool activateEncryptedRuntime( const agx::String& encryptedRuntimeData,
                                     const agx::String& outputLicenseFile );
      /// \}
      #endif



      /** \name Legacy License functions */
      /// \{

      /**
      \return Unique id for the platform where AGX is running.
      */
      const UniqueIdVector& getUniqueIds() const;


      #ifndef STANDALONE_RUNTIME


      /**
      Will try to unlock AGX with the license provided in licenseString.
      This string should contain the same information as the AGX license
      files normally used.

      If Runtime already had parsed a license and the method isValid
      returns true, using this method will cause isValid to return
      false if licenseString is not considered valid.

      \return true if the license is valid and AGX is allowed to start.
      */
      bool unlock(const char* licenseString);

      bool unlock( const agx::String& licenseString);


      /**
      Unlock AGX Dynamics using obfuscated license string.

      The input string is un-obfuscated to produce an agx.lic-type license
      string that is passed to unlock. The obfuscated string is produced by
      running
         RuntimeKeyGenerator --obfuscate agx.lic obfuscated.txt
      where RuntimeKeyGenerator is included in the AGX Dynamics installation.
      \see unlock
      \param licenseStr Obfuscated license string.
      \return true if the license is valid and AGX is allowed to start.
      */
      bool verifyAndUnlock( const char* licenseStr );

      /**
      \return true if the key=value pair exists for key.
      \see readValue
      */
      bool hasKey( const char* key ) const;

      /**
      \return the value associated with key. If key is not found, the empty string will be returned.
      */
      const agx::String& readValue( const char* key ) const;

      ///\}



      /** \name Status functions */
      /// \{

      /**
      \return true if a valid license has been read and AGX is allowed to start.
      If false, getStatus() contains error information.
      */
      bool isValid() const;

      /**
      Provides status information if license validation failed
      \return Information about the isValid result.
      */
      const agx::String& getStatus() const;

      /**
      Provides optional extended status information about why license validation failed
      */
      const agx::String& getExtendedStatus() const;


      /**
      Check if a named AGX module is enabled
      \return True if module name is enabled
      */
      bool isModuleEnabled( const char* name ) const;

      /**
      \return the full set of enabled modules
      */
      const agx::StringVector& getEnabledModules() const;

      /// \}


      /**
      Version string can be one of two variants:

      1) Version range: begin_range - end_range, 1.2-1.4, or 1-2, 1.2.3.4-1.3.4.5
      2) Pure version: 1.2, 1, 1.2.3.4,
      Compare \p versionString with the version numbers. If generation.major.minor.patch <= versionString then this method returns true

      generation always have to be supplied
      If you are only using three or fewer version numbers, start with generation then major etc.
      \return true If generation.major.minor.patch <= versionString then this method returns true
      */
      bool checkVersion(const agx::String& versionString, int generation, int major=-1, int minor=-1, int patch=-1);


      SINGLETON_CLASSNAME_METHOD();
      #endif

      #ifndef STANDALONE_RUNTIME
      /// Singleton cleanup method.
      virtual void shutdown() override;
      #else
      void shutdown();
      #endif

      /**
      Clear any previously loaded license.
      */
      void clear();

    protected:
      Runtime();

      virtual ~Runtime();

    private:
      static Runtime *s_instance;

      #ifndef STANDALONE_RUNTIME
      typedef agx::HashTable< agx::String, agx::String > StringTable;

      #if AGX_LAAS_ENABLE()
      bool readLaasLicense();
      #endif


      bool readAgxLicense();
      bool validate();

      void fillTable( void* ptr );




      StringTable       m_table;

      agx::StringVector m_modules;

      agx::String       m_status;         // Contains status information
      agx::String       m_extendedStatus; //

      bool              m_valid;
      #endif

      UniqueIdVector m_uniqueIds;
  };


}

