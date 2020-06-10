#ifndef PLATO_HEAT_EQUATION_PROBLEM_HPP
#define PLATO_HEAT_EQUATION_PROBLEM_HPP

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "parabolic/VectorFunction.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "parabolic/ScalarFunctionBase.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "ComputedField.hpp"

#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "parabolic/ScalarFunctionBaseFactory.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato {

namespace Parabolic {

/**********************************************************************************/
template<typename SimplexPhysics>
class Problem : public Plato::AbstractProblem
{
/**********************************************************************************/
private:

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;

    // required
    Plato::Parabolic::VectorFunction<SimplexPhysics> mEqualityConstraint;

    Plato::OrdinalType mNumSteps;
    Plato::Scalar mTimeStep;

    // optional
    std::shared_ptr<const Plato::Parabolic::ScalarFunctionBase> mObjective;
    std::shared_ptr<const Plato::Elliptic::ScalarFunctionBase>  mConstraint;

    Plato::ScalarMultiVector mAdjoints;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mGlobalState;

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobianP;

    Teuchos::RCP<Plato::ComputedFields<SpatialDim>> mComputedFields;

    bool mIsSelfAdjoint;

    Plato::LocalOrdinalVector mBcDofs;
    Plato::ScalarVector mBcValues;

    rcp<Plato::AbstractSolver> mSolver;
public:
    /******************************************************************************/
    Problem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aParamList,
      Plato::Comm::Machine aMachine
    ) :
      mEqualityConstraint(aMesh, aMeshSets, mDataMap, aParamList, aParamList.get < std::string > ("PDE Constraint")),
      mNumSteps(aParamList.sublist("Time Integration").get<int>("Number Time Steps")),
      mTimeStep(aParamList.sublist("Time Integration").get<Plato::Scalar>("Time Step")),
      mConstraint(nullptr),
      mObjective(nullptr),
      mResidual("MyResidual", mEqualityConstraint.size()),
      mGlobalState("States", mNumSteps, mEqualityConstraint.size()),
      mJacobian(Teuchos::null),
      mJacobianP(Teuchos::null),
      mComputedFields(Teuchos::null),
      mIsSelfAdjoint(aParamList.get<bool>("Self-Adjoint", false))
    /******************************************************************************/
    {
        this->initialize(aMesh, aMeshSets, aParamList);

        Plato::SolverFactory tSolverFactory(aParamList.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh, aMachine, SimplexPhysics::mNumDofsPerNode);
    }

    /******************************************************************************//**
     * @brief Return number of degrees of freedom in solution.
     * @return Number of degrees of freedom
    **********************************************************************************/
    Plato::OrdinalType getNumSolutionDofs()
    {
        return SimplexPhysics::mNumDofsPerNode;
    }

    /******************************************************************************/
    void setGlobalState(const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));
        Kokkos::deep_copy(mGlobalState, aStates);
    }

    /******************************************************************************/
    Plato::ScalarMultiVector getGlobalState()
    /******************************************************************************/
    {
        return mGlobalState;
    }

    /******************************************************************************/
    Plato::ScalarMultiVector getAdjoint()
    /******************************************************************************/
    {
        return mAdjoints;
    }

    /******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    /******************************************************************************/
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
        else
        {
            Plato::applyConstraints<mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
    }

    void applyBoundaryLoads(const Plato::ScalarVector & aForce){}

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aGlobalState 2D container of state variables
     * @param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aGlobalState)
    { return; }

    /******************************************************************************/
    Plato::ScalarMultiVector solution(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
          Plato::ScalarVector tState = Kokkos::subview(mGlobalState, tStepIndex, Kokkos::ALL());
          Plato::ScalarVector tPrevState = Kokkos::subview(mGlobalState, tStepIndex-1, Kokkos::ALL());
          Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tState);

          mResidual = mEqualityConstraint.value(tState, tPrevState, aControl, mTimeStep);

          mJacobian = mEqualityConstraint.gradient_u(tState, tPrevState, aControl, mTimeStep);
          this->applyConstraints(mJacobian, mResidual);

          Plato::ScalarVector deltaT("increment", tState.extent(0));
          Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), deltaT);

          mSolver->solve(*mJacobian, deltaT, mResidual);

          Plato::blas1::axpy(-1.0, deltaT, tState);

        }
        return mGlobalState;
    }

    /******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->value(aStates, aControl);
    }

    /******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mGlobalState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->value(tState, aControl);
    }

    /******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        Plato::ScalarMultiVector tStates = solution(aControl);
        return mObjective->value(tStates, aControl);
    }

    /******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mGlobalState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->value(tState, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        // compute dFd\phi: partial of objective wrt control
        auto tTotalObjectiveWRT_Control = mObjective->gradient_z(aStates, aControl, mTimeStep);

        // compute lagrange multiplier at the last time step, n
        

        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

            auto tState     = Kokkos::subview(aStates,   tStepIndex,   Kokkos::ALL());
            auto tPrevState = Kokkos::subview(aStates,   tStepIndex-1, Kokkos::ALL());
            Plato::ScalarVector tAdjoint   = Kokkos::subview(mAdjoints, tStepIndex,   Kokkos::ALL());

            // compute dFdT^k: partial of objective wrt T at step k = tStepIndex
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(aStates, aControl, mTimeStep, tStepIndex);

            if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1
                Plato::ScalarVector tNextState   = Kokkos::subview(aStates,   tStepIndex+1, Kokkos::ALL());
                Plato::ScalarVector tNextAdjoint = Kokkos::subview(mAdjoints, tStepIndex+1, Kokkos::ALL());
                // compute dQ^{k+1}/dT^k: partial of PDE at k+1 wrt current state, k.
                mJacobianP = mEqualityConstraint.gradient_p(tNextState, tState, aControl, mTimeStep);

                // multiply dQ^{k+1}/dT^k by lagrange multiplier from k+1 and add to dFdT^k
                Plato::MatrixTimesVectorPlusVector(mJacobianP, tNextAdjoint, tPartialObjectiveWRT_State);
            }
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dQ^k/dT^k: partial of PDE at k wrt state current state, k.
            mJacobian = mEqualityConstraint.gradient_u(tState, tPrevState, aControl, mTimeStep);

            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            mSolver->solve(*mJacobian, tAdjoint, tPartialObjectiveWRT_State);

            // compute dQ^k/d\phi: partial of PDE wrt control at step k.
            // dQ^k/d\phi is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mEqualityConstraint.gradient_z(tState, tPrevState, aControl, mTimeStep);
    
            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tAdjoint, tTotalObjectiveWRT_Control);

        }

        return tTotalObjectiveWRT_Control;
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        // compute partial derivative wrt x
        auto tPartialObjectiveWRT_Config  = mObjective->gradient_x(aStates, aControl, mTimeStep);

        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

            auto tState     = Kokkos::subview(aStates,   tStepIndex,   Kokkos::ALL());
            auto tPrevState = Kokkos::subview(aStates,   tStepIndex-1, Kokkos::ALL());
            Plato::ScalarVector tAdjoint   = Kokkos::subview(mAdjoints, tStepIndex,   Kokkos::ALL());

            // compute dFdT^k: partial of objective wrt T at step k = tStepIndex
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(aStates, aControl, mTimeStep, tStepIndex);

            if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1
                Plato::ScalarVector tNextState   = Kokkos::subview(aStates,   tStepIndex+1, Kokkos::ALL());
                Plato::ScalarVector tNextAdjoint = Kokkos::subview(mAdjoints, tStepIndex+1, Kokkos::ALL());
                // compute dQ^{k+1}/dT^k: partial of PDE at k+1 wrt current state, k.
                mJacobianP = mEqualityConstraint.gradient_p(tNextState, tState, aControl, mTimeStep);

                // multiply dQ^{k+1}/dT^k by lagrange multiplier from k+1 and add to dFdT^k
                Plato::MatrixTimesVectorPlusVector(mJacobianP, tNextAdjoint, tPartialObjectiveWRT_State);
            }
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dQ^k/dT^k: partial of PDE at k wrt state current state, k.
            mJacobian = mEqualityConstraint.gradient_u(tState, tPrevState, aControl, mTimeStep);

            this->applyConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            mSolver->solve(*mJacobian, tAdjoint, tPartialObjectiveWRT_State);

            // compute dQ^k/dx: partial of PDE wrt config.
            // dQ^k/dx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mEqualityConstraint.gradient_x(tState, tPrevState, aControl, mTimeStep);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, tAdjoint, tPartialObjectiveWRT_Config);
        }
        return tPartialObjectiveWRT_Config;
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mGlobalState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_z(tState, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_z(tState, aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        return mObjective->gradient_z(mGlobalState, aControl, mTimeStep);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        return mObjective->gradient_x(mGlobalState, aControl, mTimeStep);
    }


    /******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(mGlobalState, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_x(tState, aControl, mTimeStep);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aStates)
    /******************************************************************************/
    {
        assert(aStates.extent(0) == mGlobalState.extent(0));
        assert(aStates.extent(1) == mGlobalState.extent(1));

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << ", MESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER."
                    << " USER SHOULD MAKE SURE THAT CONSTRAINT FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
        auto tLastStepIndex = mNumSteps - 1;
        auto tState = Kokkos::subview(aStates, tLastStepIndex, Kokkos::ALL());
        return mConstraint->gradient_x(tState, aControl, mTimeStep);
    }

private:
    /******************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    /******************************************************************************/
    {
        if(aParamList.isSublist("Computed Fields"))
        {
          mComputedFields = Teuchos::rcp(new Plato::ComputedFields<SpatialDim>(aMesh, aParamList.sublist("Computed Fields")));
        }

        if(aParamList.isSublist("Initial State"))
        {
            Plato::ScalarVector tInitialState = Kokkos::subview(mGlobalState, 0, Kokkos::ALL());
            if(mComputedFields == Teuchos::null) {
              throw std::runtime_error("No 'Computed Fields' have been defined");
            }

            auto tDofNames = mEqualityConstraint.getDofNames();

            auto tInitStateParams = aParamList.sublist("Initial State");
            for (auto i = tInitStateParams.begin(); i != tInitStateParams.end(); ++i) {
                const auto &tEntry = tInitStateParams.entry(i);
                const auto &tName  = tInitStateParams.name(i);

                if (tEntry.isList()) 
                {
                    auto& tStateList = tInitStateParams.sublist(tName);
                    auto tFieldName = tStateList.get<std::string>("Computed Field");
                    int tDofIndex = -1;
                    for (int j = 0; j < tDofNames.size(); ++j)
                    {
                        if (tDofNames[j] == tName) {
                           tDofIndex = j;
                        }
                    }
                    mComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), tInitialState);
                }
            }
        }

        Plato::Elliptic::ScalarFunctionBaseFactory<SimplexPhysics> tFunctionBaseFactory;
        Plato::Parabolic::ScalarFunctionBaseFactory<SimplexPhysics> tParabolicFunctionBaseFactory;
        if(aParamList.isType<std::string>("Constraint"))
        {
            std::string tName = aParamList.get<std::string>("Constraint");
            mConstraint = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aParamList, tName);
        }

        if(aParamList.isType<std::string>("Objective"))
        {
            std::string tName = aParamList.get<std::string>("Objective");
            mObjective = tParabolicFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aParamList, tName);

            auto tLength = mEqualityConstraint.size();
            mAdjoints = Plato::ScalarMultiVector("MyAdjoint", mNumSteps, tLength);
        }

        // parse constraints
        //
        Plato::EssentialBCs<SimplexPhysics>
            tEssentialBoundaryConditions(aParamList.sublist("Essential Boundary Conditions",false));
        tEssentialBoundaryConditions.get(aMeshSets, mBcDofs, mBcValues);
    }
};

} // end namespace Parabolic

} // end namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::Parabolic::Problem<::Plato::Thermal<1>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Parabolic::Problem<::Plato::Thermal<2>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Parabolic::Problem<::Plato::Thermal<3>>;
#endif

#endif // PLATO_PROBLEM_HPP