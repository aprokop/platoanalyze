#include "hyperbolic/HyperbolicScalarFunctionFactory.hpp"
#include "hyperbolic/HyperbolicScalarFunctionFactory_def.hpp"


#ifdef PLATOANALYZE_1D
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<1>>;
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Electromechanics<1>>;
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::MicromorphicMechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<2>>;
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Electromechanics<2>>;
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::MicromorphicMechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<3>>;
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Electromechanics<3>>;
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::MicromorphicMechanics<3>>;
#endif
