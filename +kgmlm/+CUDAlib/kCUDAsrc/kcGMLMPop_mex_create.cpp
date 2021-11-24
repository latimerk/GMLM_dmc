/*
 * kcGMLMPop_mex_create.cu
 * Mex function to put GMLMPop data on GPUs and setup computation space.
 *  Takes 3 arguments:   GMLMPopstructure struct (from GMLMPop class)
 *                       trials struct (from GMLMPop class)
 *                       boolean (is double) for if the object is double precision (or single)
 *
 *  Requires 1 output:   ptr (in long int form) to GMLMPop object
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

class MexFunction : public matlab::mex::Function {
private:
    // Pointer to MATLAB engine to call fprintf
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    matlab::data::ArrayFactory factory;
    
    // Create an output stream
    std::ostringstream stream;
    
    template <class FPTYPE>
    kCUDA::GPUGMLMPop<FPTYPE> * setupGPUGMLMPop(const matlab::data::StructArray GMLMPopstructure_mat, const matlab::data::StructArray trialBlocks_mat) { 
        // converts into the GMLMPop c++ api types
        //   Not a very object-orientated way to do this, but it's clean enough for this purpose
        
        // setup object for GPUGMLMPop to send messages and errors to MATLAB
        std::shared_ptr<kCUDA::GPUGL_msg> msgObj = std::make_shared<GPUGL_msg_mex>(matlabPtr);
        
        // start with the structure
        kCUDA::GPUGMLMPop_structure_args<FPTYPE> * GMLMPopstructure = new kCUDA::GPUGMLMPop_structure_args<FPTYPE>;
        
        const matlab::data::TypedArray<const FPTYPE> binSize = GMLMPopstructure_mat[0]["binSize"];
        GMLMPopstructure->binSize = binSize[0];
        const matlab::data::TypedArray<const uint64_t> dim_B = GMLMPopstructure_mat[0]["dim_B"];
        GMLMPopstructure->dim_B  = dim_B[0];
        const matlab::data::TypedArray<const uint64_t> dim_P = GMLMPopstructure_mat[0]["dim_P"];
        GMLMPopstructure->dim_P  = dim_P[0];
        
        const matlab::data::TypedArray<const int> logLikeSettings = GMLMPopstructure_mat[0]["logLikeSettings"];
        int logLikeSettings_temp = logLikeSettings[0];
        switch(logLikeSettings_temp) {
            case kCUDA::ll_poissExp:
                GMLMPopstructure->logLikeSettings = kCUDA::ll_poissExp;
                break;
            case kCUDA::ll_sqErr:
                GMLMPopstructure->logLikeSettings = kCUDA::ll_sqErr;
                break;
            case kCUDA::ll_truncatedPoissExp:
                GMLMPopstructure->logLikeSettings = kCUDA::ll_truncatedPoissExp;
                break;
            default:
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("Invalid log likelihood type") }));
        }
         
        const matlab::data::TypedArray<const FPTYPE> logLikeParams = GMLMPopstructure_mat[0]["logLikeParams"];
        size_t np = logLikeParams.getNumberOfElements();
        GMLMPopstructure->logLikeParams.resize(np);
        for(int ii = 0; ii < np; ii++) {
            GMLMPopstructure->logLikeParams[ii] = logLikeParams[ii];
        }
        
        
        // sets up the group structure
        const matlab::data::StructArray GMLMPopGroupStructure_mat = GMLMPopstructure_mat[0]["Groups"];
        size_t dim_J = GMLMPopGroupStructure_mat.getNumberOfElements();
        GMLMPopstructure->Groups.resize(dim_J);
        
        std::vector<std::vector<size_t>> dim_Fs;
        dim_Fs.resize(dim_J);
        
        for(int jj = 0; jj < dim_J; jj++) {
            GMLMPopstructure->Groups[jj] = new kCUDA::GPUGMLMPop_structure_Group_args<FPTYPE>;
            
            const matlab::data::TypedArray<const uint64_t> dim_T      = GMLMPopGroupStructure_mat[jj]["dim_T"];
            const matlab::data::TypedArray<const uint64_t> dim_R_max  = GMLMPopGroupStructure_mat[jj]["dim_R_max"];
            const matlab::data::TypedArray<const uint64_t> dim_A      = GMLMPopGroupStructure_mat[jj]["dim_A"];
            const matlab::data::TypedArray<const uint32_t> factor_idx = GMLMPopGroupStructure_mat[jj]["factor_idx"];
            
            size_t dim_S = dim_T.getNumberOfElements();
            
            GMLMPopstructure->Groups[jj]->dim_A     = dim_A[0];
            GMLMPopstructure->Groups[jj]->dim_R_max = dim_R_max[0];
            
            // gets number of factors and checks that every one exists
            if(factor_idx.getNumberOfElements() != dim_S) {
                 matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("factor setup does not match dim_T!") }));
            }
            size_t dim_D = 1;
            for(int ss = 0; ss < dim_S; ss++) {
                dim_D = (factor_idx[ss] >= dim_D) ? (factor_idx[ss] + 1) : dim_D;
            }
            
            //setup Group structure dimensions
            GMLMPopstructure->Groups[jj]->X_shared.resize(  dim_D);
            GMLMPopstructure->Groups[jj]->dim_T.resize(     dim_S);
            GMLMPopstructure->Groups[jj]->factor_idx.resize(dim_S);
            
            std::vector<bool> factor_exists;
            factor_exists.assign(dim_D, false);
            dim_Fs[jj].assign(dim_D, 1);
            for(int ss = 0; ss < dim_S; ss++) {
                //gets factor idx for each dim and multiplies that dim's size to the factor size
                GMLMPopstructure->Groups[jj]->dim_T[ss]      = dim_T[ss];
                GMLMPopstructure->Groups[jj]->factor_idx[ss] = factor_idx[ss];
                
                factor_exists[factor_idx[ss]] = true;
                dim_Fs[jj][factor_idx[ss]] *= dim_T[ss];
                
                if(dim_T[ss] == 0) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("Groups.dim_T[ss] cannot be 0!") }));
                }
            }
            for(unsigned int ff = 0; ff < dim_D; ff++) {
                // all factors need at least one element (factor numbers are 0:(dim_D-1)
                if(!factor_exists[ff]) {
                     matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("factor setup invalid!") }));
                }
            }
            
            //for each factor
            const matlab::data::CellArray X_shared = GMLMPopGroupStructure_mat[jj]["X_shared"];
            if(X_shared.getNumberOfElements() != dim_D) {
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("Groups.X_shared is not the correct size!") }));
            }
            for(unsigned int ff = 0; ff < dim_D; ff++) {
                const matlab::data::TypedArray<const FPTYPE> X_shared_ff = X_shared[ff];
                if(X_shared_ff.getNumberOfElements() > 0 && X_shared_ff.getDimensions()[1] != dim_Fs[jj][ff]) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("Groups.X_shared[ff] does not have the correct number of columns for the factor!") }));
                }
                GMLMPopstructure->Groups[jj]->X_shared[ff] = new GLData_matlab<FPTYPE>(X_shared_ff);
            }
        }
        
        // sets up the trial blocks
        size_t numBlocks = trialBlocks_mat.getNumberOfElements();
        if(numBlocks == 0) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("No trial blocks given: nothing to load to GPU!") }));
        }
        
        std::vector<kCUDA::GPUGMLMPop_GPU_block_args<FPTYPE> *> gpuBlocks;
        gpuBlocks.resize(numBlocks);
        for(int bb = 0; bb < numBlocks; bb++) {
            gpuBlocks[bb] = new kCUDA::GPUGMLMPop_GPU_block_args<FPTYPE>;
            
            //gets device and compute block info
            const matlab::data::TypedArray<const int32_t> dev_num = trialBlocks_mat[bb]["GPU"];
            gpuBlocks[bb]->dev_num = dev_num[0];
            const matlab::data::TypedArray<const int32_t> max_trials_for_sparse_run = trialBlocks_mat[bb]["max_trials_for_sparse_run"];
            gpuBlocks[bb]->max_trials_for_sparse_run = max_trials_for_sparse_run[0];
            
            //sets up trials
            const matlab::data::StructArray trials_mat = trialBlocks_mat[bb]["trials"];
            
            size_t dim_M_c = trials_mat.getNumberOfElements();
            if(dim_M_c == 0) {
                matlabPtr->feval(u"error", 0,
                    std::vector<matlab::data::Array>({ factory.createScalar("No trials given in block!") }));
            }
            
            //for each trial
            gpuBlocks[bb]->trials.resize(dim_M_c);
            for(int mm = 0; mm < dim_M_c; mm++) {
                gpuBlocks[bb]->trials[mm] = new kCUDA::GPUGMLMPop_trial_args<FPTYPE>;
                
                //trial id
                const matlab::data::TypedArray<const uint32_t> trial_idx = trials_mat[mm]["trial_idx"];
                gpuBlocks[bb]->trials[mm]->trial_idx = trial_idx[0];
            
                //gets spike counts 
                const matlab::data::TypedArray<const FPTYPE> Y = trials_mat[mm]["Y"];
                if(Y.getNumberOfElements() == 0 || Y.getDimensions()[1] != GMLMPopstructure->dim_P) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("Invalid trial size found!") }));
                }
                gpuBlocks[bb]->trials[mm]->Y = new GLData_matlab<FPTYPE>(Y);
                
                //linear term
                const matlab::data::TypedArray<const FPTYPE> X_lin = trials_mat[mm]["X_lin"];
                size_t X_lin_depth = 1;
                if(X_lin.getDimensions().size() > 2) {
                    X_lin_depth = X_lin.getDimensions()[2];
                }
                if(GMLMPopstructure->dim_B > 0 && (X_lin.getDimensions()[0] != gpuBlocks[bb]->trials[mm]->dim_N() || X_lin.getDimensions()[1] != GMLMPopstructure->dim_B || (X_lin_depth != 1 && X_lin_depth != GMLMPopstructure->dim_P))) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("Trial's X_lin is not the correct size") }));
                }
                gpuBlocks[bb]->trials[mm]->X_lin = new GLData_matlab<FPTYPE>(X_lin);
                
                //setup Groups
                const matlab::data::StructArray trialGroups_mat = trials_mat[mm]["Groups"];
                if(trialGroups_mat.getNumberOfElements() != dim_J) {
                    matlabPtr->feval(u"error", 0,
                        std::vector<matlab::data::Array>({ factory.createScalar("Trial does not contain correct number of groups") }));
                }
                
                gpuBlocks[bb]->trials[mm]->Groups.resize(dim_J);
                
                for(int jj = 0; jj < dim_J; jj++) {
                    
                    gpuBlocks[bb]->trials[mm]->Groups[jj] = new kCUDA::GPUGMLMPop_trial_Group_args<FPTYPE>;
                    
                    //for each dimension
                    size_t dim_S = GMLMPopstructure->Groups[jj]->dim_T.size();
                    
                    const matlab::data::CellArray X_local = trialGroups_mat[jj]["X_local"];
                    const matlab::data::CellArray iX_shared = trialGroups_mat[jj]["iX_shared"];
                    if(X_local.getNumberOfElements() != dim_Fs[jj].size() || iX_shared.getNumberOfElements() != dim_Fs[jj].size()) {
                        matlabPtr->feval(u"error", 0,
                            std::vector<matlab::data::Array>({ factory.createScalar("Trial group structure does not match: number of factors and number of regressors inconsistent") }));
                    }
                    
                    size_t dim_D = dim_Fs[jj].size();

                    gpuBlocks[bb]->trials[mm]->Groups[jj]->X.resize(dim_D);
                    gpuBlocks[bb]->trials[mm]->Groups[jj]->iX.resize(dim_D);

                    for(int ff = 0; ff < dim_D; ff++) {
                        size_t dim_F = dim_Fs[jj][ff];
                        
                        //if shared regressor exists
                        if(!(GMLMPopstructure->Groups[jj]->X_shared[ff]->empty())) {
                            const matlab::data::TypedArray<const int> iX_shared_ff = iX_shared[ff];
                            if(iX_shared_ff.getDimensions()[0] != gpuBlocks[bb]->trials[mm]->dim_N()) {
                                matlabPtr->feval(u"error", 0,
                                    std::vector<matlab::data::Array>({ factory.createScalar("Trial group iX_shared structure does not match dim_N") }));
                            }
                            if(iX_shared_ff.getDimensions()[1] != GMLMPopstructure->Groups[jj]->dim_A) {
                                matlabPtr->feval(u"error", 0,
                                    std::vector<matlab::data::Array>({ factory.createScalar("Trial group iX_shared structure does not match dim_A") }));
                            }

                            gpuBlocks[bb]->trials[mm]->Groups[jj]->X[ff]  = new GLData_matlab<FPTYPE>();
                            gpuBlocks[bb]->trials[mm]->Groups[jj]->iX[ff] = new GLData_matlab<int>(iX_shared_ff);
                        }
                        //if expecting local regressors
                        else {
                            const matlab::data::TypedArray<const FPTYPE> X_local_ff = X_local[ff];
                            size_t dim_A_c = 1;
                            const matlab::data::ArrayDimensions X_local_ff_size = X_local_ff.getDimensions();
                            if(X_local_ff_size.size() >= 3) {
                                dim_A_c = X_local_ff_size[2];
                            }
                            if(X_local_ff_size[0] != gpuBlocks[bb]->trials[mm]->dim_N() || X_local_ff_size[1] != dim_F || (dim_A_c != GMLMPopstructure->Groups[jj]->dim_A && dim_A_c > 1)) {
                                matlabPtr->feval(u"error", 0,
                                    std::vector<matlab::data::Array>({ factory.createScalar("Trial group X_local structure does not match factor/event size") }));
                            }

                            gpuBlocks[bb]->trials[mm]->Groups[jj]->X[ff]  = new GLData_matlab<FPTYPE>(X_local_ff);
                            gpuBlocks[bb]->trials[mm]->Groups[jj]->iX[ff] = new GLData_matlab<int>();
                        }
                            
                    } // end group dimensions
                } // end groups
            } // end trials
        } // end trial blocks
        
        // calls the gmlm constructor
        kCUDA::GPUGMLMPop<FPTYPE> * gmlm = new kCUDA::GPUGMLMPop<FPTYPE>(GMLMPopstructure, gpuBlocks, msgObj);
        
        // cleans up the local objects - I didn't make destructors here to take care of this more cleanly, should have
        for(int jj = 0; jj < GMLMPopstructure->Groups.size(); jj++) {
            for(int ff = 0; ff < GMLMPopstructure->Groups[jj]->X_shared.size(); ff++) {
                delete GMLMPopstructure->Groups[jj]->X_shared[ff];
            }
            delete GMLMPopstructure->Groups[jj];
        }
        delete GMLMPopstructure;
        
        for(int bb = 0; bb < gpuBlocks.size(); bb++) {
            for(int mm = 0; mm < gpuBlocks[bb]->trials.size(); mm++) {
                for(int jj = 0; jj < gpuBlocks[bb]->trials[mm]->Groups.size(); jj++) {
                    for(int ff = 0; ff < gpuBlocks[bb]->trials[mm]->Groups[jj]->X.size(); ff++) {
                        delete gpuBlocks[bb]->trials[mm]->Groups[jj]->X[ff];
                        delete gpuBlocks[bb]->trials[mm]->Groups[jj]->iX[ff];
                    }
                    delete gpuBlocks[bb]->trials[mm]->Groups[jj];
                }
                delete gpuBlocks[bb]->trials[mm];
            }
            delete gpuBlocks[bb];
        }
        
        return gmlm;
    }
    
    
public:
    
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // check args: arg 0 is GMLMPop object, args 1 is the block start indices, args 2 is the gpu numbers
        // needs 1 output for MATLAB's weird copy on write behavior
        
        if(outputs.size() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("One outputs required to kcGMLMPop_mex_create - otherwise memory leaks could ensue!") }));
        }
        
        if(inputs.size() != 3) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("3 inputs required to kcGMLMPop_mex_create!") }));
        }
        
        if(inputs[0].getType() != matlab::data::ArrayType::STRUCT ) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Argument 1 must be a struct array!") }));
        }
        if(inputs[1].getType() != matlab::data::ArrayType::STRUCT) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Argument 2 must be a struct array!") }));
        }
        if(inputs[2].getNumberOfElements() != 1) {
            matlabPtr->feval(u"error", 0,
                std::vector<matlab::data::Array>({ factory.createScalar("Argument 3 must be a scalar!") }));
        }
        
        const matlab::data::StructArray  GMLMPopstructure = inputs[0];
        const matlab::data::StructArray  trialBlocks   = inputs[1];
        const matlab::data::TypedArray<bool> isDouble  = inputs[2]; //floating point type
        
        //call SETUP func for correct data type with inputs
        uint64_t gmlmPtr = 10;
        if(isDouble[0]) {
            kCUDA::GPUGMLMPop<double> * gmlm = setupGPUGMLMPop<double>(GMLMPopstructure, trialBlocks);
            //store the pointer as an int to return to matlab
            gmlmPtr = reinterpret_cast<uint64_t>(gmlm);
        }
        else {
            kCUDA::GPUGMLMPop<float > * gmlm = setupGPUGMLMPop<float >(GMLMPopstructure, trialBlocks);
            //store the pointer as an int to return to matlab
            gmlmPtr = reinterpret_cast<uint64_t>(gmlm);
        }
        
        //return ptr to object
        outputs[0] = factory.createScalar(gmlmPtr);
    };
};
