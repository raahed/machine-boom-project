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

#ifndef AGX_PHYSISCS_HIERARCHICAL_GRID_SOLVE_BODY_MANAGER_H
#define AGX_PHYSISCS_HIERARCHICAL_GRID_SOLVE_BODY_MANAGER_H


#include <agx/debug.h>
#include <agx/AtomicValue.h>
#include <agx/Physics/HierarchicalGrid/CellEntity.h>
#include <agx/Component.h>
#include <agxData/Buffer.h>

#ifdef _MSC_VER
# pragma warning(push)
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
# pragma warning( disable : 4251 )
#endif

namespace agx { namespace Physics { namespace HierarchicalGrid {
  
  AGX_DECLARE_POINTER_TYPES(SolveBodyManager);
  class AGXPHYSICS_EXPORT SolveBodyManager : public Component
  {
  public:
    static agx::Model* ClassModel();
    
    static const Index ArraySize = 32;
    
  public:
    SolveBodyManager();
    
    void registerCell(CellPtr cell);
    void unregisterCell(CellPtr cell);
    
    void clear();
    void resetCounters();

  protected:
    virtual ~SolveBodyManager();

  private:
    void reallocationCallback(agxData::Buffer *);
    
  private:
    agxData::BufferRef m_solveBodyCounterBuffer;
    
    typedef HashSet<CellPtr> CellHash;
    CellHash m_activeCells;
    
    agxData::Buffer::Event::CallbackType m_reallocationCallback;
    IndexVector m_inactiveCounterArrays;
    AtomicValue *m_currentBuffer;
  };
  
}}}

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif
