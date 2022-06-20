/*
 * kcGMLMPython.hpp
 * pybind11 module for calling the GMLM code from Python. 
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2022 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#include "kcGMLMPython_glm.hpp"
#include "kcGMLMPython_gmlm.hpp"

#ifdef USE_SINGLE_PRECISION
  typedef float FPTYPE;
#else
  typedef double FPTYPE;
#endif

PYBIND11_MODULE(pyGMLMcuda, m) {
    m.doc() = "API for working with CUDA code for fast GPU computation with GMLMs (made with pybind11)";

    
    #ifdef USE_GPU
        //m.def("gpuEnabled", []() { return true; });
        m.attr("GPU_ENABLED") = py::bool_([](){ return true; });
    #else
        //m.def("gpuEnabled", []() { return false; });
        m.attr("GPU_ENABLED") = py::bool_([](){ return false; });
    #endif
    #ifdef USE_SINGLE_PRECISION
        //m.def("doublePrecision", []() { return true; });
        m.attr("DOUBLE_PRECISION") = py::bool_([](){ return false; });
    #else
        //m.def("doublePrecision", []() { return false; });
        m.attr("DOUBLE_PRECISION") = py::bool_([](){ return true; });
    #endif

    py::enum_<kCUDA::logLikeType>(m, "logLikeType", "The classes of log likelihod that are allowed.")
        .value("ll_poissExp", kCUDA::logLikeType::ll_poissExp, "Standard Poisson likelihood with exponential inverse link function")
        .value("ll_sqErr", kCUDA::logLikeType::ll_sqErr, "Squared error")
        .value("ll_truncatedPoissExp", kCUDA::logLikeType::ll_truncatedPoissExp, "Truncated Poisson: sums mass over all positive values (for data with spike/no spike in bins).")
        .value("ll_poissSoftRec", kCUDA::logLikeType::ll_poissSoftRec, "Poisson likelihood with softplus inverse link function")
        .export_values();

    py::class_<kCUDA::kcGLM_trial<FPTYPE>, std::shared_ptr<kCUDA::kcGLM_trial<FPTYPE>>>(m, "kcGLM_trial")
        .def(py::init<unsigned int, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast>, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> >(),  R"mydelimiter(
          A single trial's worth of data.

          Args:
            trialNum (unsigned int): the unique trial number (trial numbers should be 0,1,...,max_trials-1; not an arbitrary set of integers)
            X (np.ndarray): [dim_N x numCovariates] the trial's design matrix
            Y (np.ndarray): [dim_N] the vector of observations
        )mydelimiter")
        .def("print", &kCUDA::kcGLM_trial<FPTYPE>::print);
    py::class_<kCUDA::kcGLM_trialBlock<FPTYPE>, std::shared_ptr<kCUDA::kcGLM_trialBlock<FPTYPE>>>(m, "kcGLM_trialBlock")
        .def(py::init<int>(),  R"mydelimiter(
          A set of trials for a GLM that are all on the same GPU.

          Args:
            GPU (int): the device number for this block.
        )mydelimiter")
        .def("addTrial", &kCUDA::kcGLM_trialBlock<FPTYPE>::addTrial,  R"mydelimiter(
            Adds a trial of data to the current block.

            Args:
              trial (kcGLM_trial): Object containing the trial's data..

            Returns:
              trialIndex (int): The trial index of the added trial (returns 0, 1, 2, ... as trials are added)
        )mydelimiter");
    py::class_<kCUDA::kcGLM_python<FPTYPE>>(m, "kcGLM")
        .def(py::init<const unsigned int, kCUDA::logLikeType, double>(),  R"mydelimiter(
          A GLM class for single neurons that works like the GMLM object.

          Args:
            numCovariates (int) : number of terms in the model (including any intercept term - this function won't add an intercept for you)
            logLikeType (pyGMLMcuda.logLikeType) : the type of log likelihood to use 
            binSize_sec (float):  the time bin size in seconds
        )mydelimiter")
        .def("addBlock", &kCUDA::kcGLM_python<FPTYPE>::addBlock, R"mydelimiter(
            Adds a block of trials to the current GLM object.
            Note: calls freeGPU() if data has been sent to the GPU already.

            Args:
              block (pyGMLMcuda.kcGMM_trialBlock): A set of trials to be placed on a single GPU.

            Returns:
              block index (int): Index of the block that was just added (will go 0, 1, 2, ... as blocks are added)
              
            Raises:
              value_error: if trials in block do not match expected GLM structure
        )mydelimiter")
        .def("isOnGPU", &kCUDA::kcGLM_python<FPTYPE>::isOnGPU, R"mydelimiter(
            If the variables are loaded to the GPU or not.

            Returns:
              (bool) True or false if on GPU(s).
        )mydelimiter")
        .def("freeGPU", &kCUDA::kcGLM_python<FPTYPE>::freeGPU, R"mydelimiter(
            Frees all values loaded to the GPU(s).
        )mydelimiter")
        .def("toGPU", &kCUDA::kcGLM_python<FPTYPE>::toGPU, R"mydelimiter(
            Loads all the trial blocks that have been added to the GPU.
        )mydelimiter")
        .def("computeLogLikelihood", &kCUDA::kcGLM_python<FPTYPE>::computeLogLikelihood, R"mydelimiter(
            Computes the log likelihood.

            Args:
              K (np.ndarray) : [numCovariates] vector of parameters

            Returns:
              LL (np.ndarray) : [numTrials] the log likelihood for each trial
        )mydelimiter")
        .def("computeLogLikelihood_grad", &kCUDA::kcGLM_python<FPTYPE>::computeLogLikelihood_grad, R"mydelimiter(
            Computes the gradient of log likelihood.

            Args:
              K (np.ndarray) : [numCovariates] vector of parameters

            Returns:
              G (np.ndarray) : [numCovariates]
        )mydelimiter")
        .def("computeLogLikelihood_hess", &kCUDA::kcGLM_python<FPTYPE>::computeLogLikelihood_hess, R"mydelimiter(
            Computes the Hessian of log likelihood.

            Args:
              K (np.ndarray) : [numCovariates] vector of parameters

            Returns:
              H (np.ndarray) : [numCovariates x numCovariates]
        )mydelimiter");

    // structure of GMLM: tells the code what dimensions to expect
    py::class_<kCUDA::GPUGMLM_group_structure_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<FPTYPE>>>(m, "kcGMLM_modelStructure_tensorGroup")
        .def(py::init<size_t, size_t, std::vector<size_t> &, std::vector<unsigned int> & >(),  R"mydelimiter(
            Create an object of the kcGMLM_modelStructure class to define the structure of one tensorized group of parameters for a GMLM.
            The tensor's multilinear structure is organized into modes, which can be combined into "factors".
            A factor is essentially the Khatri-Rao product of some modes.
            A factor can be one mode or several - the choice depends on how the data are organized for efficiency.
            
            The filters defined in a group can be used multiple times in a trial - defined by the number of "events".

            Args:
              dim_A (unsigned int64: size_t): The number of events for this group
              dim_R_max (unsigned int64: size_t): The maximum rank for the group (used for preallocating computation space)
              modeDimensions (list : unsigned ints) : [dim_J] the dimensions of each mode of the tensor
              modeParts (list : unsigned ints):       [dim_J] the factors which each mode belongs to. The list should start at 0 and be ascending by increments of at most 1
                                                      e.g., if modeDimensions = [6,5,8]  and modeParts = [0, 0, 1]
                                                          We have 2 factors, one containing modes 0 and 1 (of dim_F[0]  = 6*5 = 30) and a second factor (of dim_F[1] = 8)

            Raises:
              value_error: If dim_A, dim_R_max, modeDimensions < 1 (all must be positive)
                           If modeParts is incorrectly specified (not ascending correctly)
        )mydelimiter")
        .def("setSharedRegressors", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::setSharedRegressors,  R"mydelimiter(
            Sets a 'shared regressor' for a factor.
            Shared regressors allow more efficient use of regressors that are repeated across a lot of time bins.
            Each trial defines its regressors only as indices into this shared regressor matrix.

            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)
              X_shared (np.ndarray): [dim_X_shared x dim_F[factorIdx]] the shared regressors 

            Raises:
              value_error: if factorIndex is invalid or if the size of X_shared is invalid (wrong width)
        )mydelimiter") // setSharedRegressors(unsigned int partNum, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_shared) 
        .def("getModeParts", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getModeParts,  R"mydelimiter(
            Gets this list of which part each mode belongs to;

            Returns:
              modeParts (list : unsigned ints):       [dim_J] the factors which each mode belongs to. The list should start at 0 and be ascending by increments of at most 1
                                                      e.g., if modeDimensions = [6,5,8]  and modeParts = [0, 0, 1]
                                                          We have 2 factors, one containing modes 0 and 1 (of dim_F[0]  = 6*5 = 30) and a second factor (of dim_F[1] = 8)

        )mydelimiter")
        .def("isSharedRegressor", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::isSharedRegressor,  R"mydelimiter(
            Finds out if a mode has been assigned a shared regressor with the setSharedRegressors() method.

            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)

            Returns:
              isShared (bool): True if a shared regressor has been specified.

            Raises:
              value_error: if factorIndex is invalid or if the size of X_shared is invalid (wrong width)
        )mydelimiter")
        .def("getDimA", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getDimA,  R"mydelimiter(
            Gets the specified number of 'events' this group contributes to.

            Returns:
              dim_A (unsigned int64 : size_t)
        )mydelimiter")
        .def("getFactorDim", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getFactorDim,  R"mydelimiter(
            Gets the computed size for a given factor.

            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)

            Returns:
              dim_F[factorIdx] (unsigned int64 : size_t): is 0 if factorIndex is invalid
        )mydelimiter")
        .def("getModeDim", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getModeDim,  R"mydelimiter(
            Gets the specified size for a given mode.

            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)

            Returns:
              dim_F[factorIdx] (unsigned int64 : size_t): is 0 if factorIndex is invalid
        )mydelimiter")
        .def("getSharedRegressorDim", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getSharedRegressorDim,  R"mydelimiter(
            For a shared regressor, returns the number of rows in the design matrix.

            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)

            Returns:
              dim_X_shared[factorIdx] (unsigned int64 : size_t): is 0 if factorIndex is invalid or if !isSharedRegressor(facotrIndex)

        )mydelimiter")
        .def("getSharedRegressor", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getSharedRegressor,  R"mydelimiter(
            For a shared regressor, returns the coefficient.

            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)

            Returns:
              X_shared (np.ndarray): [dim_X_shared x dim_F[factorIdx]] the shared regressors 
            
            Raises:
              value_error: if factorIndex is invalid or if the size of X_shared is invalid (wrong width)
        
        )mydelimiter")
        .def("getNumFactors", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::getNumFactors,  R"mydelimiter(
            Gets the number of factors specified

            Returns:
              dim_D (unsigned int64 : size_t): The number of factors. dim_D <= getNumModes()
        )mydelimiter")
        .def("getNumModes", &kCUDA::GPUGMLM_group_structure_python<FPTYPE>::dim_S,  R"mydelimiter(
            Gets the number of modes specified

            Returns:
              dim_J (unsigned int64 : size_t): The number of factors. dim_J >= getNumFactors()
        )mydelimiter");

    py::class_<kCUDA::GPUGMLM_structure_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_structure_python<FPTYPE>>>(m, "kcGMLM_modelStructure")
        .def(py::init<size_t , size_t , kCUDA::logLikeType , double, bool >(),  R"mydelimiter(
            Create an object of the kcGMLM_modelStructure class to define the structure of a GMLM.
            Add the structure of each tensorized group of parameters with the addGroup() method.

            Args:
              numNeurons (unsigned int64: size_t): (dim_P) number of neurons to expect
              numLinearCovariates (unsigned int64: size_t): (dim_B) number of fully linear variables to expect
              logLikeType (pyGMLMcuda.logLikeType) : the type of log likelihood to use 
              binSize_sec (float):  the time bin size in seconds
              simultaneousRecording (bool): If true, data is organized as a simultaneous recording.
                                            If false, each trial holds data for one cell at a time.
        )mydelimiter")
        .def("getLogLikeType", &kCUDA::GPUGMLM_structure_python<FPTYPE>::getLogLikeType,  R"mydelimiter( 
            Gets the type of likelihood for this model.

            Returns:
              logLikeType (pyGMLMcuda.logLikeType) : the type of log likelihood being used.

        )mydelimiter")
        .def("getBinSize", &kCUDA::GPUGMLM_structure_python<FPTYPE>::getBinSize,  R"mydelimiter( 
            Gets the bin size for this model.

            Returns:
              binSize_sec (float):  the time bin size in seconds
        )mydelimiter")
        .def("isSimultaneousRecording", &kCUDA::GPUGMLM_structure_python<FPTYPE>::isSimultaneousRecording,  R"mydelimiter( 
            If is simultaneous recording or not.

            Returns:
              simultaneousRecording (bool): If true, data is organized as a simultaneous recording.
                                              If false, each trial holds data for one cell at a time.
        )mydelimiter")
        .def("addGroup", &kCUDA::GPUGMLM_structure_python<FPTYPE>::addGroup,  R"mydelimiter(
            Adds the structure of a tensor group of coefficients.
            Note: the trials will need to be constructed with the groups in the same order.

            Args:
              structureGroup (kcGMLM_modelStructure_tensorGroup): Object containing the group's structure.

            Returns:
              groupIndex (int): The group index of the added group (returns 0, 1, 2, ... as groups are added)
        )mydelimiter")
        .def("getNumGroups", &kCUDA::GPUGMLM_structure_python<FPTYPE>::dim_J,  R"mydelimiter( 
            Gets the number of parameter groups that have been added.

            Returns:
              dim_J (unsigned int64: size_t): number of groups
        )mydelimiter")
        .def("getNumNeurons", &kCUDA::GPUGMLM_structure_python<FPTYPE>::getNumNeurons,  R"mydelimiter( 
            Gets the number of neurons that was specified.

            Returns:
              dim_P (unsigned int64: size_t): number of neurons
        )mydelimiter")
        .def("getNumLinearTerms", &kCUDA::GPUGMLM_structure_python<FPTYPE>::getNumLinearTerms,  R"mydelimiter( 
            Gets the dimension of the linear parameters that was specified.

            Returns:
              dim_B (unsigned int64: size_t): the number of linear parameters (same for each neuron)
        )mydelimiter")
        .def("getGroup", &kCUDA::GPUGMLM_structure_python<FPTYPE>::getGroup,  R"mydelimiter( 
            Gets the structure for a tensor parameter group.

            Args:
              groupIndex (int): values 0 to getNumGroups()-1. The group requested.

            Returns:
              groupParams (pyGMLMcuda.kcGMLM_modelStructure_tensorGroup): the group
              
            Raises:
              value_error: if groupIndex is invalid
        )mydelimiter");

    // trial data: divided into blocks on GPU
    py::class_<kCUDA::GPUGMLM_trialGroup_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_trialGroup_python<FPTYPE>>>(m, "kcGMLM_trial_tensorGroup")
        .def(py::init<>(),  R"mydelimiter(
            Object that contains one set of tensorized/multilinear coefficients for a single trial formatted for a GMLM.
            Each mode is added individually via the addLocalFactor() or addSharedIdxFactor() methods.
        )mydelimiter")
        .def("addLocalFactor", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::addLocalFactor,  R"mydelimiter(
            Adds a factor of coefficients. The linear term for this type of factor is dense.
            Note: all trials will need to be built by adding the factors in the same order - currently they can't be reordered.
                  The order must match how the corresponded kcGMLM_modelStructure_tensorGroup was built.

            Each factor can have a tensor decomposition in the kcGMLM_modelStructure_tensorGroup structure (hence the usage of factor vs. mode here and the results+params structure).
            The size of the factor (dim_F) is the product of the size of the modes that contribute to the factor (the dim_Ts).

            Args:
              X (np.ndarray): [N x dim_F x (dim_A or 1)] coefficients for a multilinear mode.
                              The third dim is the number of ``events'' - if it is 1, then the linear term is the same for all events. (Or only one event is used)

            Returns:
              factorIndex (int): The index of the added factor (returns 0, 1, 2, ... as factors are added)
        )mydelimiter") // addLocalFactor(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_) 
        .def("addSharedIdxFactor", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::addSharedIdxFactor,  R"mydelimiter(
            Adds a factor of coefficients. The linear term for this type of factor is has a sparse structure.
            The coefficients are contained in a kcGMLM_modelStructure_tensorGroup object - set by kcGMLM_modelStructure_tensorGroup.setSharedRegressors.
            This strcture only needs the indices into the rows of that object,
            Note: all trials will need to be built by adding the factors in the same order - currently they can't be reordered.
                  The order must match how the corresponded kcGMLM_modelStructure_tensorGroup was built.

            Each factor can have a tensor decomposition in the kcGMLM_modelStructure_tensorGroup structure (hence the usage of factor vs. mode here and the results+params structure).
            The size of the factor (dim_F) is the product of the size of the modes that contribute to the factor (the dim_Ts).

            Args:
              iX (np.ndarray of ints): [N x (dim_A or 1)] indices into the rows of the corresponding sharedRegressor from the kcGMLM_modelStructure_tensorGroup object.
                                       Entries that are negative or larger than the sharedRegressor array are treated as 0's.
                                       The second dim is the number of ``events'' - if it is 1, then the linear term is the same for all events. (Or only one event is used)

            Returns:
              factorIndex (int): The index of the added factor (returns 0, 1, 2, ... as factors are added)
        )mydelimiter") // addSharedIdxFactor(py::array_t<int, py::array::f_style | py::array::forcecast> iX_)
        .def("getSharedIdxFactor", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::getSharedIdxFactor,  R"mydelimiter(
            Gets shared coefficient indicies for a facotr
            
            Args:
              factorIndex (int): The index of the factor

            Returns:
              iX (np.ndarray of ints): [N x (dim_A or 1)] indices into the rows of the corresponding sharedRegressor from the kcGMLM_modelStructure_tensorGroup object.
                                       Entries that are negative or larger than the sharedRegressor array are treated as 0's.
                                       The second dim is the number of ``events'' - if it is 1, then the linear term is the same for all events. (Or only one event is used)

            Raises:
              ValueError if factorIndex is invalid
        )mydelimiter")
        .def("getLocalFactor", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::getLocalFactor,  R"mydelimiter(
            Gets local coefficients for a factor
            
            Args:
              factorIndex (int): The index of the factor

            Returns:
              X (np.ndarray):[N x dim_F x (dim_A or 1)] coefficients for a multilinear mode.
                              The third dim is the number of ``events'' - if it is 1, then the linear term is the same for all events. (Or only one event is used)

            Raises:
              ValueError if factorIndex is invalid
        )mydelimiter")  
        .def("validDimA", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::validDimA,  R"mydelimiter(
            If a number of "events" is valid for the coefficient structure provided.
            
            Args:
              dim_A (int): number of events

            Returns:
              bool: True if valid, False if inconsistent with coefficient setup.
        )mydelimiter") //
        .def("getFactorDim", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::getFactorDim,  R"mydelimiter(
            Gets the dimensions of a factor.
            
            Args:
              factorIndex (int): the index of the factor (0, ..., getNumFactors()-1)

            Returns:
              dim_F[factorIdx] (unsigned int64 : size_t): is 0 if factorIndex is invalid
        )mydelimiter") //
        .def("getDimN", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::getDimN,  R"mydelimiter(
            Gets the number of timebins in the trial given the coefficients.
            
            Returns:
              dim_N (unsigned int64 : size_t): is 0 if the setup is invalid/inconsistent
        )mydelimiter") // 
        .def("getNumFactors", &kCUDA::GPUGMLM_trialGroup_python<FPTYPE>::getNumFactors,  R"mydelimiter(
            Gets the number of timebins in the trial given the coefficienbts.
            
            Returns:
              dim_D (unsigned int64 : size_t):  the number of factors that have been added so far.
        )mydelimiter");

    py::class_<kCUDA::GPUGMLM_trial_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_trial_python<FPTYPE>>>(m, "kcGMLM_trial")
        .def(py::init<py::array_t<FPTYPE, py::array::f_style | py::array::forcecast>, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast>, int, int>(),  R"mydelimiter(
            Object that contains data for a single trial formatted for a GMLM.
            Constructor for a trial that only contains data for one neuron: an individually recorded neuron model.

            Args:
              Y (numpy.ndarray): [N x 1] the spike observations where N is the trial length
              X_lin (numpy.ndarray): [N x B] the linear coefficients for this trial
              trialNum (int): the unique index (non-negative) for this trial  (trial numbers should be 0,1,...,max_trials-1; not an arbitrary set of integers)
              neuron (int): the (non-negative) neuron index for which neuron this trial is for (neuron indices should be 0,1,...,dim_P-1; not an arbitrary set of integers)
        )mydelimiter")
        .def(py::init<py::array_t<FPTYPE, py::array::f_style | py::array::forcecast>, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast>, int>(),  R"mydelimiter(
            Object that contains data for a single trial formatted for a GMLM.
            Constructor for a trial that contains data for all neurons: a simultaneous population model.

            Args:
              Y (numpy.ndarray): [N x P] the spike observations where N is the trial length and P is the number of neurons.
              X_lin (numpy.ndarray): [N x B x (P or 1)] the linear coefficients for this trial. They can be unique for each neuron (3rd dim is size P) or the same (3rd dim is size 1 or X_lin is only 2D)
              trialNum (int): the unique index (non-negative) for this trial
        )mydelimiter")
        .def("addGroup", &kCUDA::GPUGMLM_trial_python<FPTYPE>::addGroup,  R"mydelimiter(
            Adds a tensor group of coefficients.
            Note: all trials will need to be built by adding the groups in the same order - currently they can't be reordered.
                  The order must match how the corresponded kcGMLM_modelStructure was built.

            Args:
              trialGroup (kcGMLM_trial_tensorGroup): Object containing the group's coefficients. Size of group's coefficients (N) must match the trial length (N)

            Returns:
              groupIndex (int): The group index of the added group (returns 0, 1, 2, ... as groups are added)
        )mydelimiter")// addGroup(std::shared_ptr<GPUGMLM_trialGroup_python<FPTYPE>> group)
        .def("getDimN", &kCUDA::GPUGMLM_trial_python<FPTYPE>::getDimN,  R"mydelimiter(
            Gets the trial length

            Args:
              N (unsigned int 64 : size_t): The trial's length / number of time bins.
        )mydelimiter")
        .def("getNumGroups", &kCUDA::GPUGMLM_trial_python<FPTYPE>::getNumGroups,  R"mydelimiter( 
            Gets the number of tensor parameter groups That have been added to this trial.

            Returns:
              dim_J (unsigned int): The number of groups.
        )mydelimiter")
        .def("getTrialNum", &kCUDA::GPUGMLM_trial_python<FPTYPE>::getTrialNum,  R"mydelimiter(
            Gets the trial index

            Returns:
              (int): The trial's index
        )mydelimiter")
        .def("getNeuronNum", &kCUDA::GPUGMLM_trial_python<FPTYPE>::getNeuronNum,  R"mydelimiter(
            Gets the neuron index (for non-simultaneous data)

            Returns:
              (int): The trial's neuron idx (will be -1, an invalid value, for simultaneous population data)
        )mydelimiter")
        .def("getObservations",  &kCUDA::GPUGMLM_trial_python<FPTYPE>::getObservations,  R"mydelimiter(
            Gets the observations for the trial.

            Returns:
               Y (numpy.ndarray): [N x P] the spike observations where N is the trial length and P is the number of neurons.
        )mydelimiter")
        .def("getLinearCoefficients",  &kCUDA::GPUGMLM_trial_python<FPTYPE>::getLinearCoefficients,  R"mydelimiter(
            Gets the linear coefficients for the trial.
            
            Returns:
               X_lin (numpy.ndarray): [N x B x (P or 1)] the linear coefficients for this trial. They can be unique for each neuron (3rd dim is size P) or the same (3rd dim is size 1 or X_lin is only 2D)
        )mydelimiter")
        .def("getGroup", &kCUDA::GPUGMLM_trial_python<FPTYPE>::getGroup,  R"mydelimiter(
            Gets a tensor group of coefficients.

            Args:
              groupIdx (int): the index of the tensor group

            Returns:
              trialGroup (kcGMLM_trial_tensorGroup): Object containing the group's coefficients. Size of group's coefficients (N) must match the trial length (N)

            Raises:
              value_error: if group number is invalid
        )mydelimiter"); 
        // GPUGMLM_trial_python(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_, int trialNum, int neuron_idx)
        // GPUGMLM_trial_python(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_, int trialNum)
            
    py::class_<kCUDA::GPUGMLM_trialBlock_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_trialBlock_python<FPTYPE>>>(m, "kcGMLM_trialBlock")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<FPTYPE>>, unsigned int>(),  R"mydelimiter(
            Object that contains a set of trials formatted for a GMLM.
            All trials in a block are to be loaded to the same GPU.

            Args:
              groupStructure (pyGMLMcuda.kcGMLM_modelStructure_tensorGoup): Defines the structure of the GMLM tensor group to expect.
              GPU (int): the device number for this block.
        )mydelimiter")
        .def("getTrial", &kCUDA::GPUGMLM_trialBlock_python<FPTYPE>::getTrial,  R"mydelimiter(
            Gets a trial on current block.

            Args
              trialNumber (int): Trial number

            Returns:
              kcGMLM_trial
              
            Raises:
              value_error: if trial number is invalid
        )mydelimiter")
        .def("getNumTrials", &kCUDA::GPUGMLM_trialBlock_python<FPTYPE>::getNumTrials,  R"mydelimiter(
            Gets a trial on current block.

            Returns:
              (int) number of trials
        )mydelimiter")
        .def("addTrial", &kCUDA::GPUGMLM_trialBlock_python<FPTYPE>::addTrial,  R"mydelimiter(
            Adds a trial of data to the current block.
            The trial will need to match the block's GMLM structure.

            Args:
              trial (kcGMLM_trial): Object containing the trial's data

            Returns:
              trialIndex (int): The trial index of the added trial (returns 0, 1, 2, ... as trials are added)

            Raises:
              value_error: if trial does not match expected GMLM structure
        )mydelimiter"); // addTrial(std::shared_ptr<GPUGMLM_trial_python<FPTPE>> trial) 
        // GPUGMLM_trialBlock_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_, unsigned int devNum)  

    // parameters
     py::class_<kCUDA::GPUGMLM_group_params_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_group_params_python<FPTYPE>>>(m, "kcGMLM_parameters_tensorGroup")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<FPTYPE>>, size_t>(),  R"mydelimiter(
            Object that contains the parameters of a GMLM for one set of tensorized parameters.
            All the pieces are stored separately (not vectorized) for easier interpretability. 
                Objects of this class should be contained in a kcGMLM_parameters object.

            Args:
              groupStructure (pyGMLMcuda.kcGMLM_modelStructure_tensorGoup): Defines the structure of the GMLM tensor group to expect.
              dim_P (unsigned int64 : size_t): the number of neurons to expect. (not in the the group structure)
        )mydelimiter")
        .def("getRank", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::getRank,  R"mydelimiter(
            Gets the rank of the tensor.

            Returns:
              dim_R (unsigned int64 : size_t): the rank.
        )mydelimiter") // size_t getRank()
        .def("setRank", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::setRank,  R"mydelimiter(
            Changes the rank of the tensor group.

            Args:
              dim_R (unsigned int64 : size_t): the rank. Should be 0 <= dim_R <= groupStructure.dim_R_max;

            Raises:
              value_error: if dim_R is invalid (e.g., too large for pre-allocated space)
        )mydelimiter") // setRank(size_t dim_R_new)
        .def("getDimT", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::getDimT, R"mydelimiter(
            Gets the dimensions for a tensor mode (excluding the neuron weight mode)

            Args:
              mode (int): values 0, 1, .., getDimS()-1. Which mode.

            Returns:
              dim_T (unsigned int64 : size_t): the dimension of the mode.
        )mydelimiter") // size_t getDimT(int mode)
        .def("getDimS", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::getDimS, R"mydelimiter(
            Gets the number of tensor modes in the group (excluding the neuron weight mode).

            Returns:
              dim_S (unsigned int64 : size_t): the number of modes.
        )mydelimiter") // size_t getDimS())
        .def("setV", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::setV,  R"mydelimiter( 
            Sets the neuron loading weights for this group.

            Args:
              V (numpy.ndarray): The new weights [neurons x getRank()]

            Raises:
              value_error: if parameter size is invalid
        )mydelimiter") // size_t setV(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> V_new)
        .def("setT", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::setT, R"mydelimiter( 
            Sets the parameters for one mode in this group.

            Args:
              mode (int): values 0, 1, .., getDimS()-1. Which mode.
              T[mode] (numpy.ndarray): The new parameters [getDimT(mode) x getRank()]

            Raises:
              value_error: if parameter size or mode is invalid
        )mydelimiter") // size_t setT(int mode, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> T_new)
        .def("getV", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::getV,  R"mydelimiter( 
            Gets the  neuron loading weights for this group.

            Returns:
              V (numpy.ndarray): The weights [neurons x getRank()]
        )mydelimiter") // py::array_t<FPTYPE, py::array::f_style> getV() 
        .def("getT", &kCUDA::GPUGMLM_group_params_python<FPTYPE>::getT, R"mydelimiter( 
            Gets the parameters for one mode in this group.

            Args:
              mode (int): values 0, 1, .., getDimS()-1. Which mode.

            Returns:
              T[mode] (numpy.ndarray): The parameters [getDimT(mode) x getRank()]

            Raises:
              value_error: if mode is invalid
        )mydelimiter");// py::array_t<FPTYPE, py::array::f_style> getT(unsigned int mode) 
        // GPUGMLM_group_params_python(GPUGMLM_group_structure_python<FPTYPE> & structure, size_t dim_P) 

     py::class_<kCUDA::GPUGMLM_params_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_params_python<FPTYPE>>>(m, "kcGMLM_parameters")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<FPTYPE>>>(),  R"mydelimiter(
            Object that contains the parameters of a GMLM.
            All the pieces are stored separately (not vectorized) for easier interpretability. 

            Args:
              structure (pyGMLMcuda.kcGMLM_modelStructure): Defines the structure of the GMLM to expect.
        )mydelimiter")
        .def("getW", &kCUDA::GPUGMLM_params_python<FPTYPE>::getW,  R"mydelimiter( 
            Gets the of the neuron-wise constant/baseline firing rate parameter (W).

            Returns:
              W (numpy.ndarray): The constant terms [neurons x 1]
        )mydelimiter") // py::array_t<FPTYPE, py::array::f_style> getW() 
        .def("getB", &kCUDA::GPUGMLM_params_python<FPTYPE>::getB,  R"mydelimiter( 
            Gets the neuron-wise linear parameter (B).

            Returns:
              B (numpy.ndarray):  [dim_B x neurons] where dim_B is the dimension of the linear term.
        )mydelimiter") // py::array_t<FPTYPE, py::array::f_style> getB() 
        .def("getNumGroups", &kCUDA::GPUGMLM_params_python<FPTYPE>::getNumGroups,  R"mydelimiter( 
            Gets the number of tensor parameter groups in the model.
            Each group can be obtained by the getGroupParams() method.

            Returns:
              dim_J (unsigned int): The number of groups.
        )mydelimiter") // unsigned int getNumGroups()
        .def("setLinearParams", &kCUDA::GPUGMLM_params_python<FPTYPE>::setLinearParams,  R"mydelimiter( 
            Sets both  the of the neuron-wise constant/baseline firing rate parameter (W) and the neuron-wise linear parameter (B).

            Args:
              W (numpy.ndarray): The constant terms [neurons x 1]
              B (numpy.ndarray):  [dim_B x neurons] where dim_B is the dimension of the linear term.
              
            Raises:
              value_error: if the sizes of the arguments do not match expected dimensions
        )mydelimiter") // setLinearParams(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> W_new, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> B_new) 
        .def("getGroupParams", &kCUDA::GPUGMLM_params_python<FPTYPE>::getGroupParams,  R"mydelimiter( 
            Gets the parameters for a tensor parameter group.

            Args:
              groupIndex (int): values 0 to getNumGroups()-1. The group requested.

            Returns:
              groupParams (pyGMLMcuda.kcGMLM_parameters_tensorGroup): the group
              
            Raises:
              value_error: if groupIndex is invalid
        )mydelimiter"); // std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<FPTYPE> getGroupParams(int group)
        // GPUGMLM_params_python(GPUGMLM_structure_python<FPTYPE> & modelStructure)
        
    // results
    py::class_<kCUDA::GPUGMLM_group_results_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_group_results_python<FPTYPE>>>(m, "kcGMLM_results_tensorGroup")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<FPTYPE>>, size_t>(),  R"mydelimiter(
            Object that contains the results of a GMLM log likelihood gradient computations for one set of tensorized parameters.
            All the gradient pieces are stored separately (not vectorized) for easier interpretability. 
                Objects of this class should be contained in a kcGMLM_results object.

            Args:
              groupStructure (pyGMLMcuda.kcGMLM_modelStructure_tensorGoup): Defines the structure of the GMLM tensor group to expect.
              dim_P (unsigned int64 : size_t): the number of neurons to expect. (not in the the group structure)
        )mydelimiter")
        .def("getRank", &kCUDA::GPUGMLM_group_results_python<FPTYPE>::getRank,  R"mydelimiter(
            Gets the rank of the tensor.

            Returns:
              dim_R (unsigned int64 : size_t): the rank.
        )mydelimiter") // size_t getRank()
        .def("setRank", &kCUDA::GPUGMLM_group_results_python<FPTYPE>::setRank,  R"mydelimiter(
            Changes the rank of the tensor group.

            Args:
              dim_R (unsigned int64 : size_t): the rank. Should be 0 <= dim_R <= groupStructure.dim_R_max;

            Raises:
              value_error: if dim_R is invalid (e.g., too large for pre-allocated space)
        )mydelimiter") // setRank(size_t dim_R_new)
        .def("getDimT", &kCUDA::GPUGMLM_group_results_python<FPTYPE>::getDimT, R"mydelimiter(
            Gets the dimensions for a tensor mode (excluding the neuron weight mode)

            Args:
              mode (int): values 0, 1, .., getDimS()-1. Which mode.

            Returns:
              dim_T (unsigned int64 : size_t): the dimension of the mode.
        )mydelimiter") // size_t getDimT(int mode)
        .def("getDimS", &kCUDA::GPUGMLM_group_results_python<FPTYPE>::getDimS, R"mydelimiter(
            Gets the number of tensor modes in the group (excluding the neuron weight mode).

            Returns:
              dim_S (unsigned int64 : size_t): the number of modes.
        )mydelimiter") // size_t getDimS())
        .def("getDV", &kCUDA::GPUGMLM_group_results_python<FPTYPE>::getDV,  R"mydelimiter( 
            Gets the gradient of the neuron loading weights for this group.

            Returns:
              dV (numpy.ndarray): The gradient [neurons x getRank()]
        )mydelimiter") // py::array_t<FPTYPE, py::array::f_style> getDV() 
        .def("getDT", &kCUDA::GPUGMLM_group_results_python<FPTYPE>::getDT, R"mydelimiter( 
            Gets the gradient of the parameters for one mode in this group.

            Args:
              mode (int): values 0, 1, .., getDimS()-1. Which mode.

            Returns:
              dT[mode] (numpy.ndarray): The gradient [getDimT(mode) x getRank()]
        )mydelimiter");// py::array_t<FPTYPE, py::array::f_style> getT(unsigned int mode) 
        // GPUGMLM_group_results_python(GPUGMLM_structure_Group_args<FPTYPE> & structure, size_t dim_P) 

     py::class_<kCUDA::GPUGMLM_results_python<FPTYPE>, std::shared_ptr<kCUDA::GPUGMLM_results_python<FPTYPE>>>(m, "kcGMLM_results")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<FPTYPE>>, size_t>(),  R"mydelimiter(
            Object that contains the results of a GMLM log likelihood computations.
            The results include the log likelihood for each trial (and neuron) and gradients for all the parameters.
            All the gradient pieces are stored separately (not vectorized) for easier interpretability. 

            Args:
              structure (pyGMLMcuda.kcGMLM_modelStructure): Defines the structure of the GMLM to expect.
              maxTrials (unsigned int64 : size_t): the maximum number of trials to expect (This isn't accessible by structure, which doesn't contain trials)
        )mydelimiter")
        .def("getTrialLL", &kCUDA::GPUGMLM_results_python<FPTYPE>::getTrialLL,  R"mydelimiter( 
            Gets the trial-wise log likelihoods for the GMLM object.

            Returns:
              trialLL (numpy.ndarray):  If the model is for a simulataneous populations is [trials x neurons]
                                        If the model is for individually recorded neurons, is [trials x 1]
        )mydelimiter")// py::array_t<FPTYPE, py::array::f_style> getTrialLL() 
        .def("getDW", &kCUDA::GPUGMLM_results_python<FPTYPE>::getDW,  R"mydelimiter( 
            Gets the gradient of the neuron-wise constant/baseline firing rate parameter (W).

            Returns:
              dW (numpy.ndarray): The gradient [neurons x 1]
        )mydelimiter") // py::array_t<FPTYPE, py::array::f_style> getDW() 
        .def("getDB", &kCUDA::GPUGMLM_results_python<FPTYPE>::getDB,  R"mydelimiter( 
            Gets the gradient of the neuron-wise linear parameter (B).

            Returns:
              dB (numpy.ndarray): The gradient [dim_B x neurons] where dim_B is the dimension of the linear term.
        )mydelimiter") // py::array_t<FPTYPE, py::array::f_style> getDB() 
        .def("getNumGroups", &kCUDA::GPUGMLM_results_python<FPTYPE>::getNumGroups,  R"mydelimiter( 
            Gets the number of tensor parameter groups in the model.
            Each group can be obtained by the getGroupResults() method.

            Returns:
              dim_J (unsigned int): The number of groups.
        )mydelimiter") // unsigned int getNumGroups();
        .def("getGroupResults", &kCUDA::GPUGMLM_results_python<FPTYPE>::getGroupResults,  R"mydelimiter( 
            Gets the gradient results for a tensor parameter group.

            Args:
              groupIndex (int): values 0 to getNumGroups()-1. The group requested.

            Returns:
              groupResults (pyGMLMcuda.kcGMLM_results_tensorGroup): the group
        )mydelimiter"); // std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<FPTYPE> getGroupResults(int group)     

        // GPUGMLM_results_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> & modelStructure, const size_t max_trials)

    // GMLM class
    py::class_<kCUDA::kcGMLM_python<FPTYPE>>(m, "kcGMLM")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<FPTYPE>>>(),  R"mydelimiter(
            Object that contains the complete set of GMLM data and computation space on the GPU.
            It can compute the log likelihood and gradients, and return them to the host in a set of numpy arrays.

            Args:
              structure (pyGMLMcuda.kcGMLM_modelStructure): Defines the structure of the GMLM to expect.
        )mydelimiter")
        .def("getBlock", &kCUDA::kcGMLM_python<FPTYPE>::getBlock, R"mydelimiter(
            Gets a blocks of trials to the current GMLM object.

            Args:
              block (int): index of the block

            Returns:
              array of pyGMLMcuda.kcGMLM_trialBlock
                            
            Raises:
              value_error: if block index is invalid
        )mydelimiter") 
        .def("getNumBlocks", &kCUDA::kcGMLM_python<FPTYPE>::getNumBlocks, R"mydelimiter(
            Gets the number of blocks loaded to the GMLM.
            
            Returns:
              (int) number of blocks
        )mydelimiter") 
        .def("addBlock", &kCUDA::kcGMLM_python<FPTYPE>::addBlock, R"mydelimiter(
            Adds a block of trials to the current GMLM object.
            Note: calls freeGPU() if data has been sent to the GPU already.

            Args:
              block (pyGMLMcuda.kcGMLM_trialBlock): A set of trials to be placed on a single GPU.

            Returns:
              block index (int): Index of the block that was just added (will go 0, 1, 2, ... as blocks are added)
              
            Raises:
              value_error: if trials in block do not match expected GMLM structure
        )mydelimiter") // addBlock(std::shared_ptr<GPUGMLM_trialBlock_python<FPTYPE>> block)
        .def("isOnGPU", &kCUDA::kcGMLM_python<FPTYPE>::isOnGPU, R"mydelimiter(
            If the variables are loaded to the GPU or not.

            Returns:
              (bool) True or false if on GPU(s).
        )mydelimiter")// bool isOnGPU()
        .def("freeGPU", &kCUDA::kcGMLM_python<FPTYPE>::freeGPU, R"mydelimiter(
            Frees all values loaded to the GPU(s).
        )mydelimiter")// void freeGPU()
        .def("toGPU", &kCUDA::kcGMLM_python<FPTYPE>::toGPU, R"mydelimiter(
            Loads all the trial blocks that have been added to the GPU.
        )mydelimiter")// void toGPU()
        .def("computeLogLikelihood", &kCUDA::kcGMLM_python<FPTYPE>::computeLogLikelihood, R"mydelimiter(
            Computes the log likelihood (and gradients if setComputeGradient(true) has been called)

            Args:
              params (pyGMLMcuda.kcGMLM_params): Object containing all the model parameters

            Returns:
              results (pyGMLMcuda.kcGMLM_results): Object containing all the results
        )mydelimiter")// std::shared_ptr<kCUDA::GPUGMLM_results_python<FPTYPE>> computeLogLikelihood(params)

        .def("computeLogLikelihood_async", &kCUDA::kcGMLM_python<FPTYPE>::computeLogLikelihood_async,  R"mydelimiter(
            Send a call to the  GPU to compute the log likelihood (and gradients if setComputeGradient(true) has been called).
            Returns before the computation has finished so that it can be collected later by the getResults() method.

            Args:
              params (pyGMLMcuda.kcGMLM_params): Object containing all the model parameters
        )mydelimiter")// void computeLogLikelihood_async(params)
        .def("getResultsStruct", &kCUDA::kcGMLM_python<FPTYPE>::getResultsStruct,  R"mydelimiter(
            Gets the full results structure that has been set up for this GMLM. Values may be invalid.

            Returns:
              results (pyGMLMcuda.kcGMLM_results): Object containing all the results.
        )mydelimiter")// std::shared_ptr<kCUDA::GPUGMLM_results_python<FPTYPE>> getResults()
        .def("getResults", &kCUDA::kcGMLM_python<FPTYPE>::getResults,  R"mydelimiter(
            Collects the last computed log likelihood results. This can be called after computeLogLikelihood_async()

            Returns:
              results (pyGMLMcuda.kcGMLM_results): Object containing all the results
        )mydelimiter")// std::shared_ptr<kCUDA::GPUGMLM_results_python<FPTYPE>> getResults()
        .def("setComputeGradient", &kCUDA::kcGMLM_python<FPTYPE>::setComputeGradient,R"mydelimiter(
            Sets whether or not to compute the gradient.

            Args:
              gradOn (bool): True if gradient should be computed, False otherwise.
        )mydelimiter")// void setComputeGradient(bool)
        .def("getParams", &kCUDA::kcGMLM_python<FPTYPE>::getParams,  R"mydelimiter(
            Gets the parameter object for this GMLM. The parameters are a shared object - getting this object and modifying the values will change parameters for the next log likelihood call.

            Returns:
              params (pyGMLMcuda.kcGMLM_params): SHARED object
        )mydelimiter")// std::shared_ptr<kCUDA::GPUGMLM_results_python<FPTYPE>> getParams()
        .def("getGMLMStructure", &kCUDA::kcGMLM_python<FPTYPE>::getGMLMStructure,  R"mydelimiter(
            Gets the structure object for this GMLM. 

            Returns:
              structure (pyGMLMcuda.kcGMLM_modelStructure): SHARED object
        )mydelimiter"); //std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> getGMLMStructure()
        // kcGMLM_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_)
}