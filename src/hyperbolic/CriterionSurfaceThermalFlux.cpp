/*
 * CriterionSurfaceThermalFlux.cpp
 *
 *  Created on: Jul 21, 2021
 */

#include "hyperbolic/CriterionSurfaceThermalFlux.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionSurfaceThermalFlux, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionSurfaceThermalFlux, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionSurfaceThermalFlux, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
