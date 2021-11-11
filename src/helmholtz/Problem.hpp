#pragma once

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "AnalyzeOutput.hpp"
#include "ImplicitFunctors.hpp"
#include "MultipointConstraints.hpp"
#include "SpatialModel.hpp"

#include "ParseTools.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "PlatoUtilities.hpp"

#include "helmholtz/VectorFunction.hpp"
#include "AnalyzeMacros.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsT>
class Problem: public Plato::AbstractProblem
{
private:

    static constexpr Plato::OrdinalType mSpatialDim = PhysicsT::mNumSpatialDims; /*!< spatial dimensions */

    using VectorFunctionType = Plato::Helmholtz::VectorFunction<PhysicsT>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    std::vector<std::string> mFixedDomainNames; /*!< list of names for fixed domains */

    std::shared_ptr<Plato::MultipointConstraints> mMPCs; /*!< multipoint constraint interface */

    rcp<Plato::AbstractSolver> mSolver;

    std::string mPDEType; /*!< partial differential equation type */
    std::string mPhysics; /*!< physics used for the simulation */

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    Problem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aProblemParams,
      Comm::Machine aMachine
    ) :
      mSpatialModel  (aMesh, aMeshSets, aProblemParams),
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mResidual      ("MyResidual", mPDE->size()),
      mStates        ("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian      (Teuchos::null),
      mPDEType       (aProblemParams.get<std::string>("PDE Constraint")),
      mPhysics       (aProblemParams.get<std::string>("Physics")),
      mMPCs          (nullptr)
    {
        this->initialize(aProblemParams);

        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh.nverts(), aMachine, PhysicsT::mNumDofsPerNode);
    }

    ~Problem(){}

    Plato::OrdinalType numNodes() const
    {
        const auto tNumNodes = mPDE->numNodes();
        return (tNumNodes);
    }

    Plato::OrdinalType numCells() const
    {
        const auto tNumCells = mPDE->numCells();
        return (tNumCells);
    }
    
    Plato::OrdinalType numDofsPerCell() const
    {
        const auto tNumDofsPerCell = mPDE->numDofsPerCell();
        return (tNumDofsPerCell);
    }

    Plato::OrdinalType numNodesPerCell() const
    {
        const auto tNumNodesPerCell = mPDE->numNodesPerCell();
        return (tNumNodesPerCell);
    }

    Plato::OrdinalType numDofsPerNode() const
    {
        const auto tNumDofsPerNode = mPDE->numDofsPerNode();
        return (tNumDofsPerNode);
    }

    Plato::OrdinalType numControlsPerNode() const
    {
        const auto tNumControlsPerNode = mPDE->numControlsPerNode();
        return (tNumControlsPerNode);
    }

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override
    {
        auto tDataMap = this->getDataMap();
        auto tSolution = this->getSolution();
        auto tSolutionOutput = mPDE->getSolutionStateOutputData(tSolution);
        Plato::universal_solution_output<mSpatialDim>(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
    {
        THROWERR("UPDATE PROBLEM: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyFixedBlockConstraints(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector & aVector)
    //**********************************************************************************/
    {
        if(mBcDofs.size() > 0)
        {
            Plato::ScalarVector tBcValues("fixed block filtered values", mBcDofs.size());
            Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), tBcValues);

            Plato::applyConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tBcValues);
        }
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return solution database
    **********************************************************************************/
    Plato::Solutions
    solution(const Plato::ScalarVector & aControl)
    {
        Plato::ScalarVector tStatesSubView = Kokkos::subview(mStates, 0, Kokkos::ALL());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mDataMap.clearStates();
        mDataMap.scalarNodeFields["Topology"] = aControl;

        mResidual = mPDE->value(tStatesSubView, aControl);
        Plato::blas1::scale(-1.0, mResidual);

        mJacobian = mPDE->gradient_u(tStatesSubView, aControl);

        this->applyFixedBlockConstraints(mJacobian, mResidual);

        mSolver->solve(*mJacobian, tStatesSubView, mResidual);

        auto tSolution = this->getSolution();
        return tSolution;
    }

    /******************************************************************************//**
     * \brief Solve system of equations related to chain rule of Helmholtz filter 
     * for gradients
     * \param [in] aControl 1D view of criterion partial derivative 
     * wrt filtered control
     * \param [in] aName Name of criterion (is just a dummy for Helmhomtz to 
     * match signature of base class virtual function).
     * \return 1D view - criterion partial derivative wrt unfiltered control
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        Plato::ScalarVector tSolution("derivative of criterion wrt unfiltered control", mPDE->size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tSolution);

        mJacobian = mPDE->gradient_u(tSolution, aControl);

        auto tPartialPDE_WRT_Control = mPDE->gradient_z(tSolution, aControl);

        Plato::blas1::scale(-1.0, aControl);

        Plato::ScalarVector tIntermediateSolution("intermediate solution", mPDE->size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tIntermediateSolution);
        mSolver->solve(*mJacobian, tIntermediateSolution, aControl);

        Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tIntermediateSolution, tSolution);

        return tSolution;
    }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        THROWERR("CRITERION VALUE: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        THROWERR("CRITERION VALUE: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        THROWERR("CRITERION GRADIENT: NO INSTANCE OF THIS FUNCTION WITH SOLUTION INPUT IMPLEMENTED FOR HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Omega_h::Mesh& aMesh,
                    Teuchos::ParameterList& aProblemParams)
    {
        auto tName = aProblemParams.get<std::string>("PDE Constraint");
        mPDE = std::make_shared<Plato::Helmholtz::VectorFunction<PhysicsT>>(mSpatialModel, mDataMap, aProblemParams, tName);

        if(aProblemParams.isSublist("Multipoint Constraints") == true)
        {
            Plato::OrdinalType tNumDofsPerNode = mPDE->numDofsPerNode();
            auto & tMyParams = aProblemParams.sublist("Multipoint Constraints", false);
            mMPCs = std::make_shared<Plato::MultipointConstraints>(mSpatialModel, tNumDofsPerNode, tMyParams);
            mMPCs->setupTransform();
        }

        if(aProblemParams.isSublist("Fixed Domains"))
        {
            this->setFixedDomainEssentialBCs(aMesh,aProblemParams.sublist("Fixed Domains"));
        }
    }

    void setFixedDomainEssentialBCs(Omega_h::Mesh& aMesh,
                                    Teuchos::ParameterList& aFixedDomainParams)
    {
        if (mPDE->numDofsPerNode() > 1)
            THROWERR("In setFixedDomainEssentialBCs: Number of DOFs per node for Helmoltz physics is greater than 1.")

        for (auto tIndex = aFixedDomainParams.begin(); tIndex != aFixedDomainParams.end(); ++tIndex)
        {
            mFixedDomainNames.push_back(aFixedDomainParams.name(tIndex));
        }
        
        const auto tNumNodes = aMesh.nverts();
        const auto tNumNodesPerCell = mPDE->numNodesPerCell();

        Plato::LocalOrdinalVector tFixedBlockNodes("Nodes in fixed blocks", tNumNodes);
        Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tFixedBlockNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tDomainName = tDomain.getDomainName();
            if (this->isFixedDomain(tDomainName))
                this->markBlockNodes(aMesh, tDomain, tNumNodesPerCell, tFixedBlockNodes);
        }

        auto tNumUniqueNodes = this->getNumberOfUniqueNodes(tFixedBlockNodes);
        Kokkos::resize(mBcDofs, tNumUniqueNodes);
        this->storeUniqueNodes(tFixedBlockNodes);
    }

    bool isFixedDomain(const std::string & aDomainName) 
    {
        for (auto iNameOrdinal(0); iNameOrdinal < mFixedDomainNames.size(); iNameOrdinal++)
        {
            if(mFixedDomainNames[iNameOrdinal] == aDomainName)
                return true;
        }
        return false;
    }

    void markBlockNodes(Omega_h::Mesh & aMesh, 
                    const Plato::SpatialDomain & aDomain, 
                    const Plato::OrdinalType aNumNodesPerCell, 
                    Plato::LocalOrdinalVector aMarkedNodes)
    {
        auto tCells2Nodes = aMesh.ask_elem_verts();
        auto tDomainCells = aDomain.cellOrdinals();
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tDomainCells.size()), LAMBDA_EXPRESSION(Plato::OrdinalType iElemOrdinal)
        {
            Plato::OrdinalType tElement = tDomainCells(iElemOrdinal); 
            for(Plato::OrdinalType iVertOrdinal=0; iVertOrdinal < aNumNodesPerCell; ++iVertOrdinal)
            {
                Plato::OrdinalType tVertIndex = tCells2Nodes[tElement*aNumNodesPerCell + iVertOrdinal];
                aMarkedNodes(tVertIndex) = 1;
            }
        }, "nodes in domain element set");
    }

    Plato::OrdinalType getNumberOfUniqueNodes(const Plato::LocalOrdinalVector & aNodeVector)
    {
        Plato::OrdinalType tSum(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,aNodeVector.size()),
        LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
        {
            aUpdate += aNodeVector(aOrdinal);
        }, tSum);
        return tSum;
    }
    void storeUniqueNodes(const Plato::LocalOrdinalVector & aMarkedNodes)
    {
        Plato::OrdinalType tOffset(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,aMarkedNodes.size()),
        KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const Plato::OrdinalType tVal = aMarkedNodes(iOrdinal);
            if( tIsFinal && tVal ) 
                mBcDofs(aUpdate) = iOrdinal; 
            aUpdate += tVal;
        }, tOffset);
    }

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDEType);
        tSolution.set("State", mStates);
        return tSolution;
    }
};
// class Problem

} // namespace Helmholtz

} // namespace Plato

#include "Helmholtz.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<1>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<2>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<3>>;
#endif

