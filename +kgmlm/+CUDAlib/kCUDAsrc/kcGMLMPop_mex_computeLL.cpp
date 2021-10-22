/*
 * kcGMLMPop_mex_computeLL.cu
 * Mex function to call the log likelihood function in a GMLM object.
 *  Takes 4-5 arguments: ptr (in long int form) to GMLM object
 *                       boolean (is double) for if the object is double precision (or single)
 *                       params struct (from GMLM class)
 *                       results struct (from GMLM class) - empty results 
 *                                matrices mean do not compute certain parts (like derivatives)
 *                       trial weights (optional) - a vector of weights for each trial
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2021 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#include "kcGMLM_mex_shared.hpp"
#include "kcGMLMPop.hpp"
#include <cmath>
#include <string>

// using namespace std::chrono; 
  
//creates a c++ param object for running the GMLMPop log likelihood from a matlab structure.
//This requires that the matlab matrices match the precision of the GMLMPop -> it will not cast from double to single!
//This performs no data copying: the pointers to the matlab matrices are taken.
//(not safe to pass back to matlab from reuse: if those pointers are cleared by matlab, seg faults could happen!)
template <class FPTYPE>
class matlab_GPUGMLMPop_params : public kCUDA::GPUGMLMPop_params<FPTYPE> {
private:
    
public:
    //constructor
    matlab_GPUGMLMPop_params(const matlab::data::StructArray & GMLMPop_params, const std::shared_ptr<matlab::engine::MATLABEngine> & matlabPtr ) { 
        //gets the W input and its size
        const matlab::data::TypedArray<const FPTYPE> W_mat = GMLMPop_params[0]["W"];
        this->W =  new GLData_matlab<FPTYPE>(W_mat);
        
        //gets the linear weights
        const matlab::data::TypedArray<const FPTYPE> B_mat = GMLMPop_params[0]["B"];
        this->B = new GLData_matlab<FPTYPE>(B_mat);
        if(this->W->getSize(0) != this->B->getSize(1) && !(this->B->empty())) {
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("inconsistent dimensions in param struct: W rows does not match B cols (number of neurons)") }));
        }
        
        //gets the groups struct and its size (the number of groups)
        const matlab::data::StructArray Groups_mat = GMLMPop_params[0]["Groups"];
        size_t dim_J  = Groups_mat.getNumberOfElements();
        this->Groups.resize(dim_J);
        for(int jj = 0; jj < dim_J; jj++) {
            //for each group: creates a group param object
            //this->Groups[jj] = new kCUDA::GPUGMLMPop_group_params<FPTYPE>();
            this->Groups[jj] = new kCUDA::GPUGMLMPop_group_params<FPTYPE>();
            
            //get's the V input (neuron loadings)
            // V must be dim_P x dim_R
            const matlab::data::TypedArray<const FPTYPE> V_mat = Groups_mat[jj]["V"];
            if(this->W->getSize(0) != V_mat.getDimensions()[0]) {
                matlab::data::ArrayFactory factory;
                matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("inconsistent dimensions in param struct: V rows does not match W") }));
            }
            this->Groups[jj]->V = new GLData_matlab<FPTYPE>(V_mat);
            
            //gets the T inputs
            const matlab::data::CellArray T_mats = Groups_mat[jj]["T"];
            size_t dim_S = T_mats.getNumberOfElements(); //number of T's: order of tensor (minus the neuron dimension)
            this->Groups[jj]->T.resize(dim_S);
            for(int ss = 0; ss < dim_S; ss++) {
                //for each dimension, get's the T and check's size (T{ss} is dim_T[ss] x dim_R
                const matlab::data::TypedArray<const FPTYPE> T_mat = T_mats[ss];
                if(T_mat.getDimensions()[1] != this->Groups[jj]->V->getSize(1)) {
                    matlab::data::ArrayFactory factory;
                    matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("inconsistent dimensions in param struct: T rank does not match V rank") }));
                }
                this->Groups[jj]->T[ss]     = new GLData_matlab<FPTYPE>(T_mat);
            }
        }
    }
    //destructor
    ~matlab_GPUGMLMPop_params() {
        //clear the groups
        delete this->W;
        delete this->B;
        for(int jj = 0; jj < this->Groups.size(); jj++) {
            for(int ss = 0; ss < this->Groups[jj]->T.size(); ss++) {
                delete this->Groups[jj]->T[ss];
            }
            delete this->Groups[jj]->V;
            delete this->Groups[jj];
        }
    }
};

//creates an compute options object (i.e., which derivatives to compute) from the a matlab results struct
//If the result's field is empty, the compute option is false for that derivative. Otherwise, it's true.
template <class FPTYPE>
class matlab_GPUGMLMPop_computeOptions : public kCUDA::GPUGMLMPop_computeOptions<FPTYPE> {
public:
    matlab_GPUGMLMPop_computeOptions(const matlab::data::StructArray & GMLMPop_results, const matlab::data::TypedArray<FPTYPE> * trial_weights_mat = NULL) {

        this->compute_trialLL =  !(GMLMPop_results[0]["trialLL"].isEmpty());
        
        //checks for an assigned dW vector
        this->compute_dW =  !(GMLMPop_results[0]["dW"].isEmpty());
        
        //checks for an assigned dB vector
        this->compute_dB =  !(GMLMPop_results[0]["dB"].isEmpty());
        
        //sets up trial weights
        this->trial_weights = new GLData_matlab<FPTYPE>(trial_weights_mat);
        
        //for each group
        const matlab::data::StructArray Groups_mat = GMLMPop_results[0]["Groups"];
        size_t dim_J  = Groups_mat.getNumberOfElements();
        this->Groups.resize(dim_J);
        for(int jj = 0; jj < dim_J; jj++) {
            this->Groups[jj] = new kCUDA::GPUGMLMPop_group_computeOptions;
            
            //checks for a dV matrix
            this->Groups[jj]->compute_dV =  !(Groups_mat[jj]["dV"].isEmpty());
            
            //for the remaining dimensions, looks at all T
            const matlab::data::CellArray dT_mats  = Groups_mat[jj]["dT"];
            
            size_t dim_S = dT_mats.getNumberOfElements();
            this->Groups[jj]->compute_dT.resize(dim_S);
            for(int ss = 0; ss < dim_S; ss++) {
                const matlab::data::Array dT_mat  = dT_mats[ss];
                
                //checks for a dT matrix
                this->Groups[jj]->compute_dT[ss] =  !(dT_mat.isEmpty());
            }
        }
    }
    //destructor
    ~matlab_GPUGMLMPop_computeOptions() {
        //clears all the groups
        delete this->trial_weights;
        for(int jj = 0; jj < this->Groups.size(); jj++) {
            delete this->Groups[jj];
        }
    }
};

//creates a results object to send into the GMLMPop computations. It takes in a matlab struct and uses the results fields given in that struct.
//This requires that the matlab matrices match the precision of the GMLMPop -> it will not cast from double to single!
//This creates no new space: it gets the matlab pointers and does a pass-by-values assignment.
//This bypasses the MATLAB C++ API in mata::data for assignments, because going through there can cause significant overhead.
//(not safe to pass back to matlab from reuse: if those pointers are cleared by matlab, seg faults could happen!)
template <class FPTYPE>
class matlab_GPUGMLMPop_results : public kCUDA::GPUGMLMPop_results<FPTYPE> {
private:
    
public:
    matlab_GPUGMLMPop_results(const matlab::data::StructArray & GMLMPop_results, const kCUDA::GPUGMLMPop_computeOptions<FPTYPE> * opts, std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr) {
        //assigns the pointers to the dW and d2W results 
        if(opts->compute_dW) {
            const matlab::data::TypedArray<FPTYPE> dW_mat = GMLMPop_results[0]["dW"];
            this->dW =  new GLData_matlab<FPTYPE>(dW_mat);
        }
        else{
            this->dW =  new GLData_matlab<FPTYPE>();
        }
        
        if(opts->compute_dB) {
            const matlab::data::TypedArray<FPTYPE> dB_mat = GMLMPop_results[0]["dB"];
            this->dB = new GLData_matlab<FPTYPE>(dB_mat);
        }
        else {
            this->dB = new GLData_matlab<FPTYPE>();
        }
        
        //assigns the pointers to the trial-wise log likelihood results 
        const matlab::data::TypedArray<FPTYPE> trialLL_mat = GMLMPop_results[0]["trialLL"];
        this->trialLL =  new GLData_matlab<FPTYPE>(trialLL_mat);
        
        //gets the tensor groups
        const matlab::data::StructArray Groups_mat = GMLMPop_results[0]["Groups"];
        size_t dim_J  = Groups_mat.getNumberOfElements();
        
        if(dim_J != opts->Groups.size()) {
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("invalid resultsStruct: group sizes do not match across objects") }));
        }
        
        this->Groups.resize(dim_J);
        for(int jj = 0; jj < dim_J; jj++) {
            //creates a groups object
            this->Groups[jj] = new kCUDA::GPUGMLMPop_group_results<FPTYPE>;
            
            //assigns the pointers to the dV results for the group
            if(opts->Groups[jj]->compute_dV) {
                const matlab::data::TypedArray<FPTYPE> dV_mat = Groups_mat[jj]["dV"];
                this->Groups[jj]->dV  = new GLData_matlab<FPTYPE>(dV_mat);
            }
            else {
                this->Groups[jj]->dV  = new GLData_matlab<FPTYPE>();
            }
            
            //looks at the results for the derivatives of the components in the remaining dimensions
            const matlab::data::CellArray dT_mats  = Groups_mat[jj]["dT"];
            size_t dim_S = dT_mats.getNumberOfElements();
            if(opts->Groups[jj]->compute_dT.size() != dim_S) {
                matlab::data::ArrayFactory factory;
                matlabPtr->feval(u"error", 0,std::vector<matlab::data::Array>({ factory.createScalar("invalid resultsStruct: dT and d2T do not match") }));
            }
            this->Groups[jj]->dT.resize(dim_S);
            for(int ss = 0; ss < dim_S; ss++) {
                //for each dimension, gets the pointers to the dT results
                if(opts->Groups[jj]->compute_dT[ss]) {
                    const matlab::data::TypedArray<FPTYPE> dT_mat  = dT_mats[ss];
                    this->Groups[jj]->dT[ss]  = new GLData_matlab<FPTYPE>(dT_mat);
                }
            }
        }
    }
    //destructor
    ~matlab_GPUGMLMPop_results() {
        //clears all the groups
        delete this->dW;
        delete this->dB;
        delete this->trialLL;
        for(int jj = 0; jj < this->Groups.size(); jj++) {
            delete this->Groups[jj]->dV;
            for(int ss = 0; ss < this->Groups[jj]->dT.size(); ss++) {
                delete this->Groups[jj]->dT[ss];
            }
            delete this->Groups[jj];
        }
    }
};

class MexFunction : public matlab::mex::Function {
private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    matlab::data::ArrayFactory factory;
    
    // Create an output stream
    std::ostringstream stream;
    
    GPUGL_msg_mex * msg;
    
private:
    

    template <class FPTYPE>
    void runComputation(const uint64_t GMLMPop_ptr, const matlab::data::StructArray & GMLMPop_params, const matlab::data::StructArray & GMLMPop_results, const matlab::data::TypedArray<FPTYPE> * trial_weights) {
        if(GMLMPop_ptr == 0) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
            return;
        }
        
        kCUDA::GPUGMLMPop<FPTYPE>         * gmlm_obj = reinterpret_cast<kCUDA::GPUGMLMPop<FPTYPE> *>(GMLMPop_ptr); //forcefully recasts the uint64 value
        //gets the model parameters
        kCUDA::GPUGMLMPop_params<FPTYPE>  * params = new matlab_GPUGMLMPop_params<FPTYPE>(GMLMPop_params, matlabPtr); //GMLMPop_params,matlabPtr, &factory
        //gets the compute options based on which fields are available in restults
        kCUDA::GPUGMLMPop_computeOptions<FPTYPE>    * opts    = new matlab_GPUGMLMPop_computeOptions<FPTYPE>(GMLMPop_results, trial_weights);
        //gets the model results
        kCUDA::GPUGMLMPop_results<FPTYPE> * results = new matlab_GPUGMLMPop_results<FPTYPE>(GMLMPop_results, opts, matlabPtr);
        
        //runs the log likelihood computation. After, the results will be in the matlab arrays as results holds the pointers to those arrays.
        gmlm_obj->computeLogLikelihood(params, opts, results);
        
        delete params;
        delete opts;
        delete results;
    }
    
    
//main func
public:
    
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if(inputs.size() != 4  && inputs.size() != 5) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("mex function requires 4-5 arguments!") }));
        }
    
        //checks that arg 0 is a pointer value to the GMLMPop C++ object
        const matlab::data::TypedArray<uint64_t> gpu_ptr_a = inputs[0];
        if(gpu_ptr_a.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("pointer argument must only contain 1 element!") }));
        }
        
        //get datatype field from GMLMPop object
        const matlab::data::TypedArray<bool> isDouble = inputs[1];
        if(isDouble.getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("cannot determine gpu data type!") }));
        }
        
        const matlab::data::StructArray GMLMPop_params  = inputs[2];
        const matlab::data::StructArray GMLMPop_results = inputs[3];
        
        
        if(gpu_ptr_a.getNumberOfElements() < 1 || gpu_ptr_a[0] == 0) {
            //no gmlm object to clear
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("gpu pointer is not initialized!") }));
        }
        else if(isDouble[0]) {
            if(inputs.size() >= 5 && !inputs[4].isEmpty()) {
                const matlab::data::TypedArray<double> trial_weights_mat = inputs[4];
                runComputation<double>(gpu_ptr_a[0], GMLMPop_params, GMLMPop_results, &trial_weights_mat);
            }
            else {
                runComputation<double>(gpu_ptr_a[0], GMLMPop_params, GMLMPop_results, NULL);
            }
        }
        else {
            if(inputs.size() >= 5 && !inputs[4].isEmpty()) {
                const matlab::data::TypedArray<float> trial_weights_mat = inputs[4];
                runComputation<float>(gpu_ptr_a[0], GMLMPop_params, GMLMPop_results, &trial_weights_mat);
            }
            else {
                runComputation<float>(gpu_ptr_a[0], GMLMPop_params, GMLMPop_results, NULL);
            }
        }
    }
};
