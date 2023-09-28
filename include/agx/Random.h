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

#ifndef AGX_RANDOM_H
#define AGX_RANDOM_H



#include <agx/macros.h>
#include <agx/Real.h>
#include <agx/Integer.h>
#include <random>

namespace agx
{
  template<typename T, template<typename> class DistributionT, typename EngineT = std::mt19937>
  class RandomGenerator
  {
  public:


    template < class Generator, class Dist >
    class variate_generator {
    public:
      // generator interface
      typedef typename Dist::result_type result_type;
      typedef Generator engine_type;
      typedef Generator engine_value_type;
      typedef Dist distribution_type;

      result_type operator() () { return D(G);  }
      result_type min() const { return D.min(); }
      result_type max() const { return D.max(); }

      // type-specific members
      variate_generator(engine_type G0,
        distribution_type D0) : G(G0), D(D0) {}
      template < class ValT >
      AGX_FORCE_INLINE result_type operator() (ValT value) { return D(G, value); }

      engine_value_type & engine()
      {
        return G;
      }

      const engine_value_type & engine() const
      {
        return G;
      }
      distribution_type & distribution()
      {
        return D;
      }
      const distribution_type & distribution() const
      {
        return D;
      }

      // exposition only:
    private:
      Generator G;
      Dist D;
    };



    typedef variate_generator< EngineT, DistributionT<T> > ImplementationT;

  public:
    RandomGenerator();
    RandomGenerator(T minValue, T maxValue);
    RandomGenerator(agx::UInt32 seed);
    RandomGenerator(agx::UInt32 seed, T minValue, T maxValue);

    /// Returns random number between minValue and maxValue (from constructor).
    T rand();
    T operator() ();

    void seed(agx::UInt32 seed);

    ImplementationT& impl();
    const ImplementationT& impl() const;

  private:
    ImplementationT m_generator;
  };



  typedef RandomGenerator<agx::Real, std::uniform_real_distribution> UniformRealGenerator;
  typedef RandomGenerator<agx::Real32, std::uniform_real_distribution> UniformReal32Generator;
  typedef RandomGenerator<agx::Real64, std::uniform_real_distribution> UniformReal64Generator;


  typedef RandomGenerator<agx::Int, std::uniform_int_distribution> UniformIntGenerator;
  typedef RandomGenerator<agx::Int32, std::uniform_int_distribution> UniformInt32Generator;
  typedef RandomGenerator<agx::Int64, std::uniform_int_distribution> UniformInt64Generator;




  /// Implementation

  template<typename T, template<typename> class DistributionT, typename EngineT>
  RandomGenerator<T, DistributionT, EngineT>::RandomGenerator() :
    m_generator(EngineT(), DistributionT<T>())
  {
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  RandomGenerator<T, DistributionT, EngineT>::RandomGenerator(T minValue, T maxValue) :
    m_generator(EngineT(), DistributionT<T>(minValue, maxValue))
  {
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  RandomGenerator<T, DistributionT, EngineT>::RandomGenerator(agx::UInt32 seed) :
    m_generator(EngineT(), DistributionT<T>())
  {
    this->seed(seed);
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  RandomGenerator<T, DistributionT, EngineT>::RandomGenerator(agx::UInt32 seed, T minValue, T maxValue) :
    m_generator(EngineT(), DistributionT<T>(minValue, maxValue))
  {
    this->seed(seed);
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  AGX_FORCE_INLINE T RandomGenerator<T, DistributionT, EngineT>::rand()
  {
    return m_generator();
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  void RandomGenerator<T, DistributionT, EngineT>::seed(agx::UInt32 seed)
  {
    m_generator.engine().seed(seed);
    m_generator.distribution().reset();
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  AGX_FORCE_INLINE T RandomGenerator<T, DistributionT, EngineT>::operator() ()
  {
    return m_generator();
  }


  template<typename T, template<typename> class DistributionT, typename EngineT>
  AGX_FORCE_INLINE typename RandomGenerator<T, DistributionT, EngineT>::ImplementationT& RandomGenerator<T, DistributionT, EngineT>::impl()
  {
    return m_generator;
  }

  template<typename T, template<typename> class DistributionT, typename EngineT>
  AGX_FORCE_INLINE const typename RandomGenerator<T, DistributionT, EngineT>::ImplementationT& RandomGenerator<T, DistributionT, EngineT>::impl() const
  {
    return m_generator;
  }
}

#endif /* AGX_RANDOM_H */
