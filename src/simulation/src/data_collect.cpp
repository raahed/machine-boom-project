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

/*

Demonstrates various ways of reading/write resources from disk and memory.

*/

#include <agx/Logger.h>
#include <agx/Hinge.h>
#include <agx/Prismatic.h>
#include <agx/CylindricalJoint.h>

#include <agxCable/Cable.h>

#include <agxOSG/ExampleApplication.h>
#include <agxCollide/ConvexFactory.h>
#include <agxIO/ImageReaderPNG.h>
#include <agxIO/ReaderWriter.h>
#include <agxOSG/ReaderWriter.h>

#include <agxUtil/HeightFieldGenerator.h>

#include <agxCollide/Convex.h>

#include "tutorialUtils.h"
#include "collector.h"
#include <unistd.h>

enum JType {
  ROTARY=0, PRISMATIC, CYLINDRICAL
};

class Joint {
  public:
    std::string name;
    JType type;
    double position;
    double velocity;
    double effort;
    agx::ConstraintRef handle;
};

class Boom : public agxSDK::Assembly {
  public:

    void setJointVelocity(std::vector<double> &ref) {
	if(!initialized) return;
	for(unsigned i=0; i<ref.size(); i++) {
	  if(joints[i].type==JType::ROTARY) {
            auto hinge = joints[i].handle->as<agx::Hinge>();
            if(hinge!=NULL) { 
	      hinge->getMotor1D()->setEnable(true);
	      hinge->getMotor1D()->setLocked(false);
	      hinge->getMotor1D()->setSpeed(ref[i]);
	    }
	  }
	  if(joints[i].type==JType::PRISMATIC) {
            auto prismatic = joints[i].handle->as<agx::Prismatic>();
            if(prismatic!=NULL) { 
	      prismatic->getMotor1D()->setEnable(true);
	      prismatic->getMotor1D()->setLocked(false);
	      prismatic->getMotor1D()->setSpeed(ref[i]);
	    }
	  }
	  if(joints[i].type==JType::CYLINDRICAL) {
            auto cylindrical = joints[i].handle->as<agx::CylindricalJoint>();
	    
	    //FIXME: this sets the same speed to both cylidrical axes
            if(cylindrical!=NULL) {  
	      cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setEnable(true);
	      cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setLocked(false);
	      cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setSpeed(ref[i]);
	      cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setEnable(true);
	      cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setLocked(false);
	      cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setSpeed(ref[i]);
	    }
	  }
	
	}
    }

    void getJointPosition(std::vector<double> &ref) {
	if(!initialized) return;
	ref.clear();
        for(auto itr=joints.begin(); itr!=joints.end(); itr++) {
	  if(itr->type==JType::ROTARY) {
            auto hinge = itr->handle->as<agx::Hinge>();
            if(hinge!=NULL) { 
	      ref.push_back(hinge->getAngle());
	    }
	  }
	  if(itr->type==JType::PRISMATIC) {
            auto prismatic = itr->handle->as<agx::Prismatic>();
            if(prismatic!=NULL) { 
	      ref.push_back(prismatic->getAngle());
	    }
	  }
	  if(itr->type==JType::PRISMATIC) {
            auto prismatic = itr->handle->as<agx::Prismatic>();
            if(prismatic!=NULL) { 
	      ref.push_back(prismatic->getAngle());
	    }
	  }
          if(itr->type==JType::CYLINDRICAL) {
            auto cylindrical = itr->handle->as<agx::CylindricalJoint>();
            if(cylindrical!=NULL) {
	      ref.push_back(cylindrical->getAngle(agx::Constraint2DOF::FIRST));
	      ref.push_back(cylindrical->getAngle(agx::Constraint2DOF::SECOND));
	    }
	  }
	}
    }

    /** gets all node positions in order x y z per node */
    void getCableNodesSerialized(std::vector<double> &nodes) {
      if(!initialized) return;
      nodes.clear();
      for(auto itr=cable1->begin(); itr!=cable1->end(); itr++) {
	agx::Vec3 pos = itr->getCenterPosition();
	nodes.push_back(pos.x());
	nodes.push_back(pos.y());
	nodes.push_back(pos.z());
      }
    }

    bool initialize(std::vector<std::string> &names, std::vector<JType> &type) {
      if(names.size()!=type.size()) {
        std::cerr<<"joint names and types size mismatch\n"; 
	return false;
      }
      joints.clear();
      auto jtr=type.begin();
      for(auto itr=names.begin(); 
	       itr!=names.end()||jtr!=type.end(); itr++,jtr++) {
	    Joint q;
	    q.name = *itr;
	    q.type = *jtr;
	    agx::Name name (q.name);
	    q.handle = this->getConstraint(name);
	    q.handle->setEnable(true);		
            if(q.type==JType::ROTARY) {
	      auto hinge = q.handle->as<agx::Hinge>();
	      if(hinge!=NULL) {
  	        std::cerr<<q.name<<" is a hinge. Enabled? "<<hinge->isEnabled()<<" Valid? "<<hinge->getValid()<<" Locking it.\n";
	        hinge->getMotor1D()->setEnable(true);
	        hinge->getMotor1D()->setLocked(true);
	      }
            } else if(q.type==JType::PRISMATIC) {
              auto prismatic = q.handle->as<agx::Prismatic>();
              if(prismatic!=NULL) {
  	        std::cerr<<q.name<<" is prismatic. Enabled? "<<prismatic->isEnabled()<<" Valid? "<<prismatic->getValid()<<" Locking it.\n";
	        prismatic->getMotor1D()->setEnable(true);
	        prismatic->getMotor1D()->setLocked(true);
	      }
            } else if(q.type==JType::CYLINDRICAL) {
                auto cylindrical = q.handle->as<agx::CylindricalJoint>();
                if(cylindrical!=NULL) {
  	          std::cerr<<q.name<<" is cylindrical. Enabled? "<<cylindrical->isEnabled()<<" Valid? "<<cylindrical->getValid()<<" Locking it.\n";
		  cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setEnable(true);
		  cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setLocked(true);
		  cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setEnable(true);
		  cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setLocked(true);
		}
	    }
	    joints.push_back(q);
      }

      //get pointers to the two cables
      agxSDK::Assembly *tmp = NULL;
      tmp = this->getAssembly(agx::Name("AGXUnity.Cable"));
      if(tmp!=NULL) cable1=dynamic_cast<agxCable::Cable*>(tmp);
      tmp = this->getAssembly(agx::Name("AGXUnity.Cable (1)"));
      if(tmp!=NULL) cable2=dynamic_cast<agxCable::Cable*>(tmp);

      initialized=(cable1!=NULL && cable2!=NULL);
      return initialized;
    }

  private:
    bool initialized=false;
    std::vector<Joint> joints;
    agxCable::Cable *cable1, *cable2;
};

//Do this at every step
class MyStepEventListener : public agxSDK::StepEventListener
{
  public:
    MyStepEventListener( Boom* boom )
      : m_boom( boom )
    {
      // Mask to tell which step events we're interested in. Available step events are:
      // Pre-collide: Before collision detection
      // Pre: After collision detection (contact list available) but before the solve stage
      // Post: After solve (transforms and velocities have been updated)
      setMask( PRE_COLLIDE | POST_STEP );  

      //if ( m_rb.isValid() )
      //  m_rb->setPosition( m_targetPosition );
      period = {4, 3, 2, 2, 2, 5, 5};
      amplitude = {0.5, 1.0, 0.2, 0.2, 0.2, 0.5, 0.2};
    }
    virtual void preCollide( const agx::TimeStamp& t )
    {
      std::vector<double> velocities;
      for(unsigned i=0; i<period.size(); i++) {
        velocities.push_back(amplitude[i]*sin(2*M_PI*t/period[i]));
      }
      
      m_boom->setJointVelocity(velocities);

    }

    virtual void post( const agx::TimeStamp& t )
    {
      std::vector<double> joint_angles;
      m_boom->getJointPosition(joint_angles);
      
      /* write csv */
      Collector::instance().append(joint_angles);

      //print cable points TODO
      std::vector<double> nodes; 
      m_boom->getCableNodesSerialized(nodes);
      for(auto itr=nodes.begin(); itr!=nodes.end(); itr++){
	std::cout<<*itr<<",";
      }
      std::cout<<"\n";
    }

  private:
//    agx::observer_ptr< agx::RigidBody > m_rb;
    Boom* m_boom;
    agx::Vec3 m_targetPosition;
    std::vector<double> period, amplitude;
};


/**
Read in the hoses scene
*/
osg::Group* loadHosesScene(  agxSDK::Simulation* simulation, agxOSG::ExampleApplication* application  )
{

  //application->setEnableDebugRenderer( true );

  agx::String filename = "BoomAndCables.agx"; // the file
  osg::Group* root = new osg::Group;

  Boom *myBoom = new Boom();
  agxSDK::AssemblyRef assembly = myBoom;
  //agxSDK::AssemblyRef assembly = new agxSDK::Assembly();

  //load scene
  agxOSG::readFile( filename, simulation, root, assembly );

  simulation->add(assembly); 
  std::vector<std::string> joint_names = {"Hinge8", "Hinge12", "Hinge15", "Hinge24", "Hinge29", "Cylindrical6", "Prismatic27"};
  std::vector<JType> joint_type_Rotation = {JType::ROTARY,JType::ROTARY,JType::ROTARY,JType::ROTARY,JType::ROTARY,JType::CYLINDRICAL,JType::PRISMATIC};
  myBoom->initialize(joint_names,joint_type_Rotation);

  simulation->setUniformGravity( agx::Vec3(0, 0, -9.81));

  Collector::instance().setup(".", filename, joint_names);

  std::cerr<<"Assembly loaded: "<<assembly->getName().str()<<std::endl;
  for(auto itr=assembly->getAssemblies().begin(); itr!= assembly->getAssemblies().end(); itr++) {
    std::cerr<<"sub-assembly "<<(*itr)->getName().str()<<std::endl;
  }

  simulation->add( new MyStepEventListener(myBoom) ); 

  return root;

}

int main( int argc, char** argv )
{
  agx::AutoInit agxInit;
  std::cerr <<
	"\t" << agxGetLibraryName() << " " << agxGetVersion() << " " << agxGetCompanyName() << "(C)\n " <<
	"\tData Collection\n" <<
    "\t--------------------------------\n\n" << std::endl;

  try {

    agxOSG::ExampleApplicationRef application = new agxOSG::ExampleApplication;

    application->addScene( loadHosesScene, '1' );

    if ( application->init( argc, argv ) ) {
      return application->run(); 
    }
  }
  catch ( std::exception& e ) {
    std::cerr << "\nCaught exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
