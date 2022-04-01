#include "kcGMLMPython_glm.hpp"
#include "kcGMLMPython_gmlm.hpp"

PYBIND11_MODULE(pyGMLM, m) {
    py::enum_<kCUDA::logLikeType>(m, "logLikeType")
        .value("ll_poissExp", kCUDA::logLikeType::ll_poissExp)
        .value("ll_sqErr", kCUDA::logLikeType::ll_sqErr)
        .value("ll_truncatedPoissExp", kCUDA::logLikeType::ll_truncatedPoissExp)
        .value("ll_poissExpRefractory", kCUDA::logLikeType::ll_poissExpRefractory)
        .export_values();
    py::class_<kCUDA::kcGLM_trial<double>, std::shared_ptr<kCUDA::kcGLM_trial<double>>>(m, "kcGLM_trial")
        .def(py::init<unsigned int, py::array_t<double, py::array::f_style | py::array::forcecast>, py::array_t<double, py::array::f_style | py::array::forcecast> >())
        .def("print", &kCUDA::kcGLM_trial<double>::print);
    py::class_<kCUDA::kcGLM_trialBlock<double>, std::shared_ptr<kCUDA::kcGLM_trialBlock<double>>>(m, "kcGLM_trialBlock")
        .def(py::init<int>())
        .def("addTrial", &kCUDA::kcGLM_trialBlock<double>::addTrial);
    py::class_<kCUDA::kcGLM_python<double>>(m, "kcGLM")
        .def(py::init<const unsigned int, kCUDA::logLikeType, double>())
        .def("addBlock", &kCUDA::kcGLM_python<double>::addBlock)
        .def("isOnGPU", &kCUDA::kcGLM_python<double>::isOnGPU)
        .def("freeGPU", &kCUDA::kcGLM_python<double>::freeGPU)
        .def("toGPU", &kCUDA::kcGLM_python<double>::toGPU)
        .def("computeLogLikelihood", &kCUDA::kcGLM_python<double>::computeLogLikelihood)
        .def("computeLogLikelihood_grad", &kCUDA::kcGLM_python<double>::computeLogLikelihood_grad)
        .def("computeLogLikelihood_hess", &kCUDA::kcGLM_python<double>::computeLogLikelihood_hess);

    // structure of GMLM: tells the code what dimensions to expect
    py::class_<kCUDA::GPUGMLM_group_structure_python<double>, std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<double>>>(m, "kcGMLM_modelStructure_tensorGroup")
        .def(py::init<size_t, size_t, std::vector<size_t> &, std::vector<unsigned int> & >())
        .def("setSharedRegressors", &kCUDA::GPUGMLM_group_structure_python<double>::setSharedRegressors) // setSharedRegressors(unsigned int partNum, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_shared) 
        .def("isSharedRegressor", &kCUDA::GPUGMLM_group_structure_python<double>::isSharedRegressor)
        .def("getDimA", &kCUDA::GPUGMLM_group_structure_python<double>::getDimA)
        .def("getFactorDim", &kCUDA::GPUGMLM_group_structure_python<double>::getFactorDim)
        .def("getSharedRegressorDim", &kCUDA::GPUGMLM_group_structure_python<double>::getSharedRegressorDim)
        .def("getNumFactors", &kCUDA::GPUGMLM_group_structure_python<double>::getNumFactors);

    py::class_<kCUDA::GPUGMLM_structure_python<double>, std::shared_ptr<kCUDA::GPUGMLM_structure_python<double>>>(m, "kcGMLM_modelStructure")
        .def(py::init<size_t , size_t , kCUDA::logLikeType , double, bool >())
        .def("addGroup", &kCUDA::GPUGMLM_structure_python<double>::addGroup);

    // trial data: divided into blocks on GPU
    py::class_<kCUDA::GPUGMLM_trialGroup_python<double>, std::shared_ptr<kCUDA::GPUGMLM_trialGroup_python<double>>>(m, "kcGMLM_trial_tensorGroup")
        .def(py::init<>())
        .def("addLocalMode", &kCUDA::GPUGMLM_trialGroup_python<double>::addLocalMode) // addLocalMode(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_) 
        .def("addSharedIdxMode", &kCUDA::GPUGMLM_trialGroup_python<double>::addSharedIdxMode) // addSharedIdxMode(py::array_t<int, py::array::f_style | py::array::forcecast> iX_)
        .def("validDimA", &kCUDA::GPUGMLM_trialGroup_python<double>::validDimA) //
        .def("getFactorDim", &kCUDA::GPUGMLM_trialGroup_python<double>::getFactorDim)//
        .def("getDimN", &kCUDA::GPUGMLM_trialGroup_python<double>::getDimN) // 
        .def("getNumFactors", &kCUDA::GPUGMLM_trialGroup_python<double>::getNumFactors);

    py::class_<kCUDA::GPUGMLM_trial_python<double>, std::shared_ptr<kCUDA::GPUGMLM_trial_python<double>>>(m, "kcGMLM_trial")
        .def(py::init<py::array_t<double, py::array::f_style | py::array::forcecast>, py::array_t<double, py::array::f_style | py::array::forcecast>, int, int>())
        .def(py::init<py::array_t<double, py::array::f_style | py::array::forcecast>, py::array_t<double, py::array::f_style | py::array::forcecast>, int>())
        .def("addGroup", &kCUDA::GPUGMLM_trial_python<double>::addGroup)// addGroup(std::shared_ptr<GPUGMLM_trialGroup_python<FPTYPE>> group)
        .def("getDimN", &kCUDA::GPUGMLM_trial_python<double>::getDimN)
        .def("getNumGroups", &kCUDA::GPUGMLM_trial_python<double>::getNumGroups); // getNumGroups
        // GPUGMLM_trial_python(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_, int trialNum, int neuron_idx)
        // GPUGMLM_trial_python(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_, int trialNum)
            
    py::class_<kCUDA::GPUGMLM_trialBlock_python<double>, std::shared_ptr<kCUDA::GPUGMLM_trialBlock_python<double>>>(m, "kcGMLM_trialBlock")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<double>>, unsigned int>())
        .def("addTrial", &kCUDA::GPUGMLM_trialBlock_python<double>::addTrial); // addTrial(std::shared_ptr<GPUGMLM_trial_python<FPTPE>> trial) 
        // GPUGMLM_trialBlock_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_, unsigned int devNum)  

    // parameters
     py::class_<kCUDA::GPUGMLM_group_params_python<double>, std::shared_ptr<kCUDA::GPUGMLM_group_params_python<double>>>(m, "kcGMLM_parameters_tensorGroup")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<double>>, size_t>())
        .def("getRank", &kCUDA::GPUGMLM_group_params_python<double>::getRank) // size_t getRank()
        .def("setRank", &kCUDA::GPUGMLM_group_params_python<double>::setRank) // setRank(size_t dim_R_new)
        .def("getDimT", &kCUDA::GPUGMLM_group_params_python<double>::getDimT) // size_t getDimT(int mode)
        .def("getDimS", &kCUDA::GPUGMLM_group_params_python<double>::getDimS) // size_t getDimS())
        .def("setV", &kCUDA::GPUGMLM_group_params_python<double>::setV) // size_t setV(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> V_new)
        .def("setT", &kCUDA::GPUGMLM_group_params_python<double>::setT) // size_t setT(int mode, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> T_new)
        .def("getV", &kCUDA::GPUGMLM_group_params_python<double>::getV) // py::array_t<FPTYPE, py::array::f_style> getV() 
        .def("getT", &kCUDA::GPUGMLM_group_params_python<double>::getT);// py::array_t<FPTYPE, py::array::f_style> getT(unsigned int mode) 
        // GPUGMLM_group_params_python(GPUGMLM_group_structure_python<FPTYPE> & structure, size_t dim_P) 

     py::class_<kCUDA::GPUGMLM_params_python<double>, std::shared_ptr<kCUDA::GPUGMLM_params_python<double>>>(m, "kcGMLM_parameters")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<double>>>())
        .def("getW", &kCUDA::GPUGMLM_params_python<double>::getW) // py::array_t<FPTYPE, py::array::f_style> getW() 
        .def("getB", &kCUDA::GPUGMLM_params_python<double>::getB) // py::array_t<FPTYPE, py::array::f_style> getB() 
        .def("getNumGroups", &kCUDA::GPUGMLM_params_python<double>::getNumGroups) // unsigned int getNumGroups()
        .def("setLinearParams", &kCUDA::GPUGMLM_params_python<double>::setLinearParams) // setLinearParams(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> W_new, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> B_new) 
        .def("getGroupParams", &kCUDA::GPUGMLM_params_python<double>::getGroupParams); // std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<double> getGroupParams(int group)
        // GPUGMLM_params_python(GPUGMLM_structure_python<FPTYPE> & modelStructure)
        
    // results
    py::class_<kCUDA::GPUGMLM_group_results_python<double>, std::shared_ptr<kCUDA::GPUGMLM_group_results_python<double>>>(m, "kcGMLM_results_tensorGroup")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<double>>, size_t>())
        .def("getRank", &kCUDA::GPUGMLM_group_results_python<double>::getRank) // size_t getRank()
        .def("getDimT", &kCUDA::GPUGMLM_group_results_python<double>::getDimT) // size_t getDimT(int mode)
        .def("getDimS", &kCUDA::GPUGMLM_group_results_python<double>::getDimS) // size_t getDimS())
        .def("getDV", &kCUDA::GPUGMLM_group_results_python<double>::getDV) // py::array_t<FPTYPE, py::array::f_style> getDV() 
        .def("getDT", &kCUDA::GPUGMLM_group_results_python<double>::getDT);// py::array_t<FPTYPE, py::array::f_style> getT(unsigned int mode) 
        // GPUGMLM_group_results_python(GPUGMLM_structure_Group_args<FPTYPE> & structure, size_t dim_P) 

     py::class_<kCUDA::GPUGMLM_results_python<double>, std::shared_ptr<kCUDA::GPUGMLM_results_python<double>>>(m, "kcGMLM_results")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<double>>, size_t>())
        .def("getTrialLL", &kCUDA::GPUGMLM_results_python<double>::getTrialLL) // py::array_t<FPTYPE, py::array::f_style> getTrialLL() 
        .def("getDW", &kCUDA::GPUGMLM_results_python<double>::getDW) // py::array_t<FPTYPE, py::array::f_style> getDW() 
        .def("getDB", &kCUDA::GPUGMLM_results_python<double>::getDB) // py::array_t<FPTYPE, py::array::f_style> getDB() 
        .def("getNumGroups", &kCUDA::GPUGMLM_results_python<double>::getNumGroups) // unsigned int getNumGroups();
        .def("getGroupResults", &kCUDA::GPUGMLM_results_python<double>::getGroupResults); // std::shared_ptr<kCUDA::GPUGMLM_group_structure_python<double> getGroupResults(int group)
        // GPUGMLM_results_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> & modelStructure, const size_t max_trials)

    // GMLM class
    py::class_<kCUDA::kcGMLM_python<double>>(m, "kcGMLM")
        .def(py::init<std::shared_ptr<kCUDA::GPUGMLM_structure_python<double>>>())
        .def("addBlock", &kCUDA::kcGMLM_python<double>::addBlock) // addBlock(std::shared_ptr<GPUGMLM_trialBlock_python<FPTYPE>> block)
        .def("isOnGPU", &kCUDA::kcGMLM_python<double>::addBlock)// bool isOnGPU()
        .def("freeGPU", &kCUDA::kcGMLM_python<double>::freeGPU)// void freeGPU()
        .def("toGPU", &kCUDA::kcGMLM_python<double>::toGPU)// void toGPU()
        .def("computeLogLikelihood", &kCUDA::kcGMLM_python<double>::computeLogLikelihood)// std::shared_ptr<kCUDA::GPUGMLM_results_python<double>> computeLogLikelihood(params)
        .def("computeLogLikelihood_async", &kCUDA::kcGMLM_python<double>::computeLogLikelihood_async)// void computeLogLikelihood_async(params)
        .def("getResults", &kCUDA::kcGMLM_python<double>::getResults)// std::shared_ptr<kCUDA::GPUGMLM_results_python<double>> getResults()
        .def("setComputeGradient", &kCUDA::kcGMLM_python<double>::setComputeGradient)// void setComputeGradient(bool)
        .def("getParams", &kCUDA::kcGMLM_python<double>::getParams);// std::shared_ptr<kCUDA::GPUGMLM_results_python<double>> getParams()
        // kcGMLM_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_)
}