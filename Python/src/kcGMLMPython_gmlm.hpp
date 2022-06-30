/*
 * kcGMLMPython_gmlm.hpp
 * Structures for linking my C++/CUDA GMLM code to Python via pybind11.
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
#ifndef GMLM_PYTHON_GMLM_H
#define GMLM_PYTHON_GMLM_H

#include "kcSharedPython.hpp"
#include "kcGMLM.hpp"


namespace kCUDA { 
template <class FPTYPE> class GPUGMLM_group_structure_python;
template <class FPTYPE> class GPUGMLM_structure_python;

// Tensor group coefficients for a single trial
template <class FPTYPE>
class GPUGMLM_trialGroup_python : public GPUGMLM_trial_Group_args<FPTYPE> {
    public:
        GPUGMLM_trialGroup_python() {
            
        }
        ~GPUGMLM_trialGroup_python() {
            for(auto xx : this->X) {
                delete xx;
            }
            for(auto ixx : this->iX) {
                delete ixx;
            }
        }
        int addLocalFactor(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_) {
            X_numpy.push_back(X_);
            this->X.push_back(new GLData_numpy<FPTYPE>(X_));

            iX_numpy.push_back(py::array_t<int, py::array::f_style>({}));
            this->iX.push_back(new GLData_numpy<int>(iX_numpy[iX_numpy.size() - 1]));
            return X_numpy.size() - 1;
        }
        int addSharedIdxFactor(py::array_t<int, py::array::f_style | py::array::forcecast> iX_) {
            iX_numpy.push_back(iX_);
            this->iX.push_back(new GLData_numpy<int>(iX_));

            X_numpy.push_back(py::array_t<FPTYPE, py::array::f_style>({}));
            this->X.push_back(new GLData_numpy<FPTYPE>(X_numpy[X_numpy.size() - 1]));
            return X_numpy.size() - 1;
        }
        py::array_t<int, py::array::f_style | py::array::forcecast> getSharedIdxFactor(unsigned int idx) {
            if(idx < iX_numpy.size()) {
                return iX_numpy[idx];
            }
            else {
                throw py::value_error("Invalid factor index.");
            }
        }
        py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> getLocalFactor(unsigned int idx) {
            if(idx < X_numpy.size()) {
                return X_numpy[idx];
            }
            else {
                throw py::value_error("Invalid factor index.");
            }
        }
        size_t getFactorDim(unsigned int factor) const {
            return this->dim_F(factor);
        }
        size_t getNumFactors() const {
            return this->dim_D();
        }
        size_t getDimN() const {
            return this->dim_N();
        }
    private:
        std::vector<py::array_t<int, py::array::f_style>> iX_numpy;
        std::vector<py::array_t<FPTYPE, py::array::f_style>> X_numpy;
};

// A single trial's data
template <class FPTYPE>
class GPUGMLM_trial_python : public GPUGMLM_trial_args<FPTYPE> {
    public:
        GPUGMLM_trial_python(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_, int trialNum, int neuron_idx) {
            this->neuron = neuron_idx;
            this->trial_idx = trialNum;
            this->Y = new GLData_numpy<FPTYPE>(Y_);
            this->X_lin = new GLData_numpy<FPTYPE>(X_lin_);
            Y_numpy = Y_;
            X_lin_numpy = X_lin_;
            msg = std::make_shared<GPUGL_msg_python>();
            if(this->dim_N() == 0) {
                throw py::value_error("Trial does not have a valid setup.");
            }
        }
        GPUGMLM_trial_python(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_, int trialNum) : GPUGMLM_trial_python(Y_, X_lin_, trialNum, -1) {

        }
        ~GPUGMLM_trial_python() {
            delete this->Y;
            delete this->X_lin;
        }

        py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> getObservations() {
            return Y_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> getLinearCoefficients() {
            return X_lin_numpy;
        }
        int getNeuronNum() {
            return this->neuron;
        }

        
        int addGroup(std::shared_ptr<GPUGMLM_trialGroup_python<FPTYPE>> group) {
            if(this->dim_N() != group->dim_N()) {
                throw py::value_error("Trial group does have the correct number of elements.");
            }
            else {
                Groups_shared.push_back(group);
                this->Groups.push_back(group.get());
            }
            return this->Groups.size() - 1;
        }
        std::shared_ptr<GPUGMLM_trialGroup_python<FPTYPE>> getGroup(unsigned int grpIdx) {
            if(grpIdx < getNumGroups()) {
                return Groups_shared[grpIdx];
            }
            else {
                throw py::value_error("Invalid group index.");
            }
        }

        size_t getNumGroups() const {
            return this->Groups.size();
        }
        size_t getDimN() const {
            return this->dim_N();
        }
        int getTrialNum() {
            return this->trial_idx;
        }
    private:
        py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_lin_numpy;
        py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> Y_numpy;

        std::vector<std::shared_ptr<GPUGMLM_trialGroup_python<FPTYPE>>> Groups_shared;
        std::shared_ptr<GPUGL_msg_python> msg;
};

// A block of trials on one GPU
template <class FPTYPE>
class GPUGMLM_trialBlock_python : public GPUGMLM_GPU_block_args<FPTYPE> {
    public:
        GPUGMLM_trialBlock_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_, unsigned int devNum) {
            this->dev_num = devNum;
            this->max_trials_for_sparse_run = 16;

            structure = structure_;
            msg = std::make_shared<GPUGL_msg_python>();
        };
        int addTrial(std::shared_ptr<GPUGMLM_trial_python<FPTYPE>> trial) {
            int vd = structure->validateTrialStructure(trial.get());
            if(vd == 1) {
                trials_shared.push_back(trial);
                this->trials.push_back(trial.get());
                return trials_shared.size() - 1;
            }
            else {
                std::ostringstream output_stream;
                output_stream << "Invalid trial error number " << vd << "\n";
                msg->printMsgTxt(output_stream);
                throw py::value_error("Trial does not match GMLM structure.");
            }
        }
        size_t getNumTrials() const {
            return this->trials_shared.size();
        }
        std::shared_ptr<GPUGMLM_trial_python<FPTYPE>> getTrial(unsigned int tr) {
            if(tr < getNumTrials()) {
                return trials_shared[tr];
            }
            else {
                throw py::value_error("Trial number invbalid.");
            }
        }
    private:
        std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure;
        std::vector<std::shared_ptr<GPUGMLM_trial_python<FPTYPE>>> trials_shared;
        std::shared_ptr<GPUGL_msg_python> msg;
};

// The structure of one tensor group
template <class FPTYPE>
class GPUGMLM_group_structure_python : public GPUGMLM_structure_Group_args<FPTYPE> {
    public:
        GPUGMLM_group_structure_python(size_t dim_A_, size_t dim_R_max_, std::vector<size_t> & modeDimensions, std::vector<unsigned int> & modeParts) {
            msg = std::make_shared<GPUGL_msg_python>();
            this->dim_A = dim_A_;
            this->dim_T = std::vector<size_t>(modeDimensions);
            this->factor_idx = std::vector<unsigned int>(modeParts);
            this->dim_R_max = dim_R_max_;
            if(this->dim_T.size() != this->factor_idx.size()) {
                throw py::value_error("'modeDimensions' must be same length as 'modeParts'.");
            }
            for(auto tt : modeDimensions) {
                if(tt < 1) {
                    throw py::value_error("'modeDimensions' must be positive.");
                }
            }
            if(this->dim_A < 1) {
                throw py::value_error("'dim_A' must be positive.");
            }
            if(this->dim_R_max < 1) {
                throw py::value_error("'dim_R_max' must be positive.");
            }

            size_t max_factor = 0;
            for(auto ss : modeParts) {
                max_factor = ss > max_factor ? ss : max_factor;
            }
            for(unsigned int ss_0 = 0; ss_0 <= max_factor; ss_0++) {
                bool found = false;
                for(auto ss : modeParts) {
                    if(ss == ss_0) {
                        found = true;
                        break;
                    }
                }
                if(!found) {
                    throw py::value_error("'modeParts' must be dense: values 0-max(modeParts) must all exist in vector.");
                }
            }

            for(unsigned int ss_0 = 0; ss_0 <= max_factor; ss_0++) {
                X_shared_numpy.push_back(py::array_t<FPTYPE, py::array::f_style>(0));
                this->X_shared.push_back(new GLData_numpy<FPTYPE>(X_shared_numpy[ss_0]));
            }
        }

        std::vector<unsigned int> getModeParts() {
            return this->factor_idx;
        }


        void setSharedRegressors(unsigned int partNum, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> X_shared_) {
            if(partNum >= X_shared_numpy.size()) {
                throw py::value_error("'partNum' does not exist.");
            }
            size_t w = 1;
            if(X_shared_.ndim() > 1) {
                w = X_shared_.shape(1);
            }
            

            if(w != this->dim_F(partNum, msg)) {
                throw py::value_error("Shared dimensions does not match part's width.");
            }
            X_shared_numpy[partNum] = X_shared_;
            delete this->X_shared[partNum];
            this->X_shared[partNum] = new GLData_numpy<FPTYPE>(X_shared_);
        }
        bool isSharedRegressor(unsigned int partNum) const {
            if(partNum >= X_shared_numpy.size()) {
                throw py::value_error("'partNum' does not exist.");
            }
            return this->X_shared[partNum] != NULL && !(this->X_shared[partNum]->empty());
        }
        size_t getDimA() const {
            return this->dim_A;
        }
        size_t getFactorDim(unsigned int factor) const {
            return this->dim_F(factor);
        }
        size_t getModeDim(unsigned int mode) const {
            if(mode >= this->dim_T.size()) {
                throw py::value_error("mode index too large.");
            }
            return this->dim_T[mode];
        }
        size_t getSharedRegressorDim(unsigned int factor) const {
            if(isSharedRegressor(factor)) {
                return this->X_shared[factor]->getSize(0);
            }
            else {
                return 0;
            }
        }
        py::array_t<FPTYPE, py::array::f_style> getSharedRegressor(unsigned int factor) const {
            if(isSharedRegressor(factor)) {
                return this->X_shared_numpy[factor];
            }
            else {
                throw py::value_error("Invalid shared regressor index.");
            }
        }
        size_t getNumFactors() const {
            return this->dim_D();
        }
    private:
        std::vector<py::array_t<FPTYPE, py::array::f_style>> X_shared_numpy;
        std::shared_ptr<GPUGL_msg_python> msg;
};

// The structure of the model
template <class FPTYPE>
class GPUGMLM_structure_python : public GPUGMLM_structure_args<FPTYPE> {
    public:
        GPUGMLM_structure_python(size_t numNeurons, size_t numLinearCovariates, logLikeType logLike, FPTYPE binSize_sec, bool simultaneousRecording) {
            if(numNeurons < 1) {
                throw py::value_error("Number of neurons must be positive.");
            }
            this->dim_B = numLinearCovariates;
            this->dim_P = numNeurons;

            this->isSimultaneousPopulation = simultaneousRecording;
            this->binSize = binSize_sec;
            this->logLikeSettings = logLike;
        }

        ~GPUGMLM_structure_python() {
            groups_shared.clear(); // not actually needed - nothing should need deleting from this object
        }

        int addGroup(std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> group) {
            groups_shared.push_back(group);
            this->Groups.push_back(group.get());
            return this->Groups.size()-1;
        }

        std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> getGroup(unsigned int jj) {
            if(jj >= groups_shared.size()) {
                throw py::value_error("Invalid group.");
            }
            return groups_shared[jj];
        }
        size_t getNumNeurons() {
            return this->dim_P;
        }
        size_t getNumLinearTerms() {
            return this->dim_B;
        }
        bool isSimultaneousRecording() {
            return this->isSimultaneousPopulation;
        }
        FPTYPE getBinSize() {
            return this->binSize;
        }
        logLikeType getLogLikeType() {
            return this->logLikeSettings;
        }
    private:
        std::vector<std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>>> groups_shared;
};


// handlers for parameters group
template <class FPTYPE>
class GPUGMLM_group_params_python : public GPUGMLM_group_params<FPTYPE> {
    public:
        GPUGMLM_group_params_python(std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> structure, size_t dim_P) {
            msg = std::make_shared<GPUGL_msg_python>();
            V_numpy = py::array_t<FPTYPE, py::array::f_style>({dim_P, structure->dim_R_max});
            this->V = new GLData_numpy<FPTYPE>(V_numpy);

            for(int tt = 0; tt < structure->dim_S(); tt++) {
                T_numpy.push_back(py::array_t<FPTYPE, py::array::f_style>({structure->dim_T[tt], structure->dim_R_max}));
                this->T.push_back(new GLData_numpy<FPTYPE>(T_numpy[tt]));
            }
            dim_R_max = structure->dim_R_max;
        }
        ~GPUGMLM_group_params_python() {
            delete this->V;
            for(auto tt : this->T) {
                delete tt;
            }
        }

        size_t getRank() {
            size_t rank = this->dim_R(msg);
            if(rank > dim_R_max) {
                throw py::value_error("Rank is too large.");
            }
            return this->dim_R(msg);
        }
        size_t getDimT(int mode) {
            return this->dim_T(mode, msg);
        } 
        size_t getDimP() {
            return this->dim_P();
        }
        size_t getDimS() {
            return this->dim_S();
        }
        void setV(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> V_new) {
            //check for correct dim_P
            if(V_new.ndim() != 2 || V_new.shape(0) != this->dim_P()) {
                throw py::value_error("Size of V does not match number of neurons.");
            }
            if(V_new.shape(1) > dim_R_max) {
                throw py::value_error("Rank of V is greater than space allocated.");
            }
            V_numpy = V_new;
            delete this->V;
            this->V = new GLData_numpy<FPTYPE>(V_new);
        }
        void setT(int mode, py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> T_new) {
            //check for correct dim_T
            if(mode < 0 || mode >= this->dim_S()) {
                throw py::value_error("Invalid mode.");
            }
            if(T_new.ndim() != 2 || T_new.shape(0) != this->dim_T(mode, msg)) {
                throw py::value_error("Size of T does not match mode dimension.");
            }
            if(T_new.shape(1) > dim_R_max) {
                throw py::value_error("Rank of V is greater than space allocated.");
            }
            T_numpy[mode] = T_new;
            delete this->T[mode];
            this->T[mode] = new GLData_numpy<FPTYPE>(T_new);
        }
        size_t setTensorGroupParams(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> V_new, std::vector<py::array_t<FPTYPE, py::array::f_style | py::array::forcecast>> T_new) {
            if(T_new.size() != this->T.size()) {
                throw py::value_error("Number of parameters doesn't match number of tensor modes.");
            }
            for(int tt = 0; tt < T_new.size(); tt++) {
                setT(tt, T_new[tt]);
            }
            setV(V_new);
            //check for matching dims
            return this->dim_R(msg);
        }
        void setRank(size_t dim_R_new) {
            if(dim_R_new > dim_R_max) {
                throw py::value_error("Rank is greater than space allocated.");
            }
            
            V_numpy = py::array_t<FPTYPE, py::array::f_style>({this->dim_P(), dim_R_new});
            delete this->V;
            this->V = new GLData_numpy<FPTYPE>(V_numpy);

            for(int tt = 0; tt < this->dim_S(); tt++) {
                T_numpy[tt] = py::array_t<FPTYPE, py::array::f_style>({this->dim_T(tt, msg), dim_R_new});
                delete this->T[tt];
                this->T[tt] = new GLData_numpy<FPTYPE>(T_numpy[tt]);
            }
        }

        py::array_t<FPTYPE, py::array::f_style> getV() {
            return V_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style> getT(unsigned int mode) {
            if(mode > T_numpy.size()) {
                throw py::value_error("Invalid tensor mode");
            }
            return T_numpy[mode];
        }
        std::vector<py::array_t<FPTYPE, py::array::f_style>> getTs() {
            return T_numpy;
        }

        bool verifyParams(std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> modelStructure, size_t dim_P) {
            // checks ranks: this function will call an error if things are inconsistent
            getRank();

            // check size of V
            if(this->dim_P() != dim_P) {
                return false;
            }

            // check size of Ts
            if(modelStructure->dim_S() != this->dim_S()) {
                return false;
            }
            for(int ss = 0; ss < this->dim_S(); ss++) {
                if(modelStructure->dim_T[ss] != this->dim_T(ss, msg)) {
                    return false;
                }
            }
            return true;
        }
    private:
        py::array_t<FPTYPE, py::array::f_style> V_numpy;
        std::vector<py::array_t<FPTYPE, py::array::f_style>> T_numpy;
        std::shared_ptr<GPUGL_msg_python> msg;
        size_t dim_R_max;
};

// handlers for parameters
template <class FPTYPE>
class GPUGMLM_params_python : public GPUGMLM_params<FPTYPE> {
    public:
        GPUGMLM_params_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> modelStructure) {
            W_numpy = py::array_t<FPTYPE, py::array::f_style>(modelStructure->dim_P);
            this->W = new GLData_numpy<FPTYPE>(W_numpy);
            B_numpy = py::array_t<FPTYPE, py::array::f_style>({modelStructure->dim_B, modelStructure->dim_P});
            this->B = new GLData_numpy<FPTYPE>(B_numpy);

            groups_shared.resize(modelStructure->Groups.size());
            for(int jj = 0; jj < modelStructure->Groups.size(); jj++) {
                groups_shared[jj] = std::make_shared<GPUGMLM_group_params_python<FPTYPE>>(modelStructure->getGroup(jj), modelStructure->dim_P);
                this->Groups.push_back(groups_shared[jj].get());
            }
        }
        ~GPUGMLM_params_python() {
           delete this->W;
           delete this->B;
        }
        std::vector<std::shared_ptr<GPUGMLM_group_params_python<FPTYPE>>> getAllGroupParams() {
            return groups_shared;
        }
        std::shared_ptr<GPUGMLM_group_params_python<FPTYPE>> getGroupParams(unsigned int group) {
            if(group >= getNumGroups()) {
                throw py::value_error("Invalid group.");
            }
            return groups_shared[group];
        }


        py::array_t<FPTYPE, py::array::f_style> getW() { return W_numpy; }
        py::array_t<FPTYPE, py::array::f_style> getB() { return B_numpy; }

        void setLinearParams(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> W_new,
                       py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> B_new) {
            this->setW(W_new);
            this->setB(B_new);
        }
        
        void setW(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> W_new) {
            if(W_new.ndim() > 2 || W_new.shape(0) != this->W->getSize(0) || W_new.strides(0)/sizeof(FPTYPE) != 1) {
                throw py::value_error("Baseline firing rate parameter invalid! Must be a vector of length dim_P.");
            }

            W_numpy = W_new;
            delete this->W;
            this->W = new GLData_numpy<FPTYPE>(W_numpy);
        }
        void setB(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> B_new) {
            if(B_new.ndim() > 2 || B_new.shape(0) != this->B->getSize(0) || B_new.shape(1) != this->B->getSize(1) || B_new.strides(0)/sizeof(FPTYPE) != 1) {
                throw py::value_error("Linear parameter invalid! Must be a matrix of size dim_B x dim_P.");
            }

            B_numpy = B_new;
            delete this->B;
            this->B = new GLData_numpy<FPTYPE>(B_numpy);
        }

        bool verifyParams(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> modelStructure) {
            // check size of linear and constant terms
            if(this->dim_B() != modelStructure->dim_B) {
                return false;
            }
            if(this->dim_P() != modelStructure->dim_P || this->dim_P() == 0) {
                return false;
            }

            // verify Groups
            if(this->dim_J() != modelStructure->dim_J()) {
                return false;
            }

            for(int jj = 0; jj < this->dim_J(); jj++) {
                if(!(this->groups_shared[jj]->verifyParams(modelStructure->getGroup(jj), this->dim_P()))) {
                    return false;
                }
            }

            return true;
        }

        unsigned int getNumGroups() {
            return this->dim_J();
        }
    private:
        py::array_t<FPTYPE, py::array::f_style> W_numpy;
        py::array_t<FPTYPE, py::array::f_style> B_numpy;
        std::vector<std::shared_ptr<GPUGMLM_group_params_python<FPTYPE>>> groups_shared;
};

// handlers for compute options for one tensor group
template <class FPTYPE>
class GPUGMLM_group_computeOptions_python : public GPUGMLM_group_computeOptions {
    public:
        GPUGMLM_group_computeOptions_python(std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> structure, bool gradOn = true) {
            this->compute_dV = gradOn;
            this->compute_dT.resize(structure->dim_S());
            for(int tt = 0; tt < structure->dim_S(); tt++) {
                this->compute_dT[tt] = gradOn;
            }
        }
        void setGrad(bool gradOn) {
            this->compute_dV = gradOn;
            for(int tt = 0; tt < this->compute_dT.size(); tt++) {
                this->compute_dT[tt] = gradOn;
            }
        }
        bool verifyOpts(std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> modelStructure) {
            if(modelStructure->dim_S() != this->compute_dT.size()) {
                return false;
            }
            return true;
        } 
};

// handlers for compute options
template <class FPTYPE>
class GPUGMLM_computeOptions_python : public GPUGMLM_computeOptions<FPTYPE> {
    public:
        GPUGMLM_computeOptions_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> modelStructure, size_t numTrials_,  bool gradOn = true) {
            this->compute_dW = gradOn;
            this->compute_dB = gradOn;
            this->compute_trialLL = true;

            trial_weights_numpy = py::array_t<FPTYPE, py::array::f_style>(0);
            this->trial_weights = new GLData_numpy<FPTYPE>(trial_weights_numpy);

            groups_shared.resize(modelStructure->Groups.size());
            for(int jj = 0; jj < modelStructure->Groups.size(); jj++) {
                groups_shared[jj] = std::make_shared<GPUGMLM_group_computeOptions_python<FPTYPE>>(modelStructure->getGroup(jj), gradOn);
                this->Groups.push_back(groups_shared[jj].get());
            }

            isSimultaneousPopulation = modelStructure->isSimultaneousPopulation;
            numTrials = numTrials_;
            dim_P = modelStructure->dim_P;
        }
        void setGrad(bool gradOn) {
            this->compute_dW = gradOn;
            this->compute_dB = gradOn;
            this->compute_trialLL = true;
            for(int jj = 0; jj < groups_shared.size(); jj++) {
                groups_shared[jj]->setGrad(gradOn);
            }
        }

        ~GPUGMLM_computeOptions_python() {
           delete this->trial_weights;
        }
        std::vector<std::shared_ptr<GPUGMLM_group_params_python<FPTYPE>>> getGroupParams() {
            return groups_shared;
        }

        void setWeights(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> trial_weights_new) {
            GLData_numpy<FPTYPE> * trial_weights_c = new GLData_numpy<FPTYPE>(trial_weights_new);
            
            if(!(trial_weights_c->empty())) {
                if(isSimultaneousPopulation && (trial_weights_c->getSize(0) != numTrials || (trial_weights_c->getSize(1) != 1 && trial_weights_c->getSize(1) != dim_P))) {
                    delete trial_weights_c;
                    throw py::value_error("Invalid shape of trial weights.");
                }
                else if(!isSimultaneousPopulation && (trial_weights_c->getSize(0) != numTrials || trial_weights_c->size() != numTrials)) {
                    delete trial_weights_c;
                    throw py::value_error("Invalid shape of trial weights.");
                }
            }

            trial_weights_numpy = trial_weights_new;
            this->trial_weights = trial_weights_c;
        }

        bool verifyOpts(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> modelStructure) {
            if(!(this->trial_weights->empty())) {
                if(isSimultaneousPopulation && (this->trial_weights->getSize(0) != numTrials || (this->trial_weights->getSize(1) != 1 && this->trial_weights->getSize(1) != dim_P))) {
                    return false;
                }
                else if(!isSimultaneousPopulation && (this->trial_weights->getSize(0) != numTrials || this->trial_weights->size() != numTrials)) {
                    return false;
                }
            }

            // verify Groups
            if(this->dim_J() != modelStructure->dim_J()) {
                return false;
            }

            for(int jj = 0; jj < this->dim_J(); jj++) {
                if(!(this->groups_shared[jj]->verifyParams(modelStructure->getGroup(jj)))) {
                    return false;
                }
            }

            return true;
        }
    private:

        py::array_t<FPTYPE, py::array::f_style> trial_weights_numpy;
        
        std::vector<std::shared_ptr<GPUGMLM_group_computeOptions_python<FPTYPE>>> groups_shared;

        bool isSimultaneousPopulation = true;
        size_t numTrials;
        size_t dim_P;
};


// handlers for results of tensor group
template <class FPTYPE>
class GPUGMLM_group_results_python : public GPUGMLM_group_results<FPTYPE> {
    public:
        GPUGMLM_group_results_python(std::shared_ptr<GPUGMLM_group_structure_python<FPTYPE>> structure, size_t dim_P) {
            msg = std::make_shared<GPUGL_msg_python>();
            dV_numpy = py::array_t<FPTYPE, py::array::f_style>({dim_P, structure->dim_R_max});
            this->dV = new GLData_numpy<FPTYPE>(dV_numpy);

            for(int tt = 0; tt < structure->dim_S(); tt++) {
                dT_numpy.push_back(py::array_t<FPTYPE, py::array::f_style>({structure->dim_T[tt], structure->dim_R_max}));
                this->dT.push_back(new GLData_numpy<FPTYPE>(dT_numpy[tt]));
            }
            dim_R_max = structure->dim_R_max;
        }
        ~GPUGMLM_group_results_python() {
            delete this->dV;
            for(auto tt : this->dT) {
                delete tt;
            }
        }

        size_t getRank() {
            return this->dim_R(msg);
        }
        size_t getDimT(int mode) {
            return this->dim_T(mode, msg);
        }
        size_t getDimS() {
            return this->dim_S();
        }
        void setRank(size_t dim_R_new) {
            if(dim_R_new > dim_R_max) {
                throw py::value_error("Rank is greater than space allocated.");
            }
            
            if(dim_R_new != this->dim_R(msg)) {
                dV_numpy = py::array_t<FPTYPE, py::array::f_style>({this->dim_P(), dim_R_new});
                delete this->dV;
                this->dV = new GLData_numpy<FPTYPE>(dV_numpy);

                for(int tt = 0; tt < this->dim_S(); tt++) {
                    dT_numpy[tt] = py::array_t<FPTYPE, py::array::f_style>({this->dim_T(tt, msg), dim_R_new});
                    delete this->dT[tt];
                    this->dT[tt] = new GLData_numpy<FPTYPE>(dT_numpy[tt]);
                }
            }
        }

        py::array_t<FPTYPE, py::array::f_style> getDV() {
            return dV_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style> getDT(unsigned int mode) {
            if(mode > dT_numpy.size()) {
                throw py::value_error("Invalid tensor mode");
            }
            return dT_numpy[mode];
        }
        std::vector<py::array_t<FPTYPE, py::array::f_style>> getDTs() {
            return dT_numpy;
        }
        void reset() {
            this->dV->assign(0);
            for(int ss = 0; ss < this->dT.size(); ss++) {
                this->dT[ss]->assign(0);
            }
        }

    private:
        py::array_t<FPTYPE, py::array::f_style> dV_numpy;
        std::vector<py::array_t<FPTYPE, py::array::f_style>> dT_numpy;
        std::shared_ptr<GPUGL_msg_python> msg;
        size_t dim_R_max;
};

// main results structure
template <class FPTYPE>
class GPUGMLM_results_python : public GPUGMLM_results<FPTYPE> {
    public:
        GPUGMLM_results_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> modelStructure, size_t max_trials) {
            dW_numpy = py::array_t<FPTYPE, py::array::f_style>(modelStructure->dim_P);
            this->dW = new GLData_numpy<FPTYPE>(dW_numpy);
            dB_numpy = py::array_t<FPTYPE, py::array::f_style>({modelStructure->dim_B, modelStructure->dim_P});
            this->dB = new GLData_numpy<FPTYPE>(dB_numpy);
            if(modelStructure->isSimultaneousPopulation) {
                trialLL_numpy = py::array_t<FPTYPE, py::array::f_style>({max_trials, modelStructure->dim_P});
            }
            else {
                trialLL_numpy = py::array_t<FPTYPE, py::array::f_style>(max_trials);
            }
            this->trialLL = new GLData_numpy<FPTYPE>(trialLL_numpy);

            groups_shared.resize(modelStructure->Groups.size());
            for(int jj = 0; jj < modelStructure->Groups.size(); jj++) {
                groups_shared[jj] = std::make_shared<GPUGMLM_group_results_python<FPTYPE>>(modelStructure->getGroup(jj), modelStructure->dim_P);
                this->Groups.push_back(groups_shared[jj].get());
            }
        }
        ~GPUGMLM_results_python() {
            delete this->dW;
            delete this->dB;
            delete this->trialLL;
        }
        void matchRank(std::shared_ptr<GPUGMLM_params_python<FPTYPE>> params) {
            if(params->getNumGroups() == getNumGroups()) {
                for(int jj = 0; jj < getNumGroups(); jj++){
                    groups_shared[jj]->setRank(params->getGroupParams(jj)->getRank());
                }
            }
            else {
                throw std::runtime_error("Invalid parameters & results setup.");
            }
        }
        std::vector<std::shared_ptr<GPUGMLM_group_results_python<FPTYPE>>> getAllGroupResults() {
            return groups_shared;
        }
        std::shared_ptr<GPUGMLM_group_results_python<FPTYPE>> getGroupResults(unsigned int group) {
            if(group >= getNumGroups()) {
                throw py::value_error("Invalid group.");
            }
            return groups_shared[group];
        }
        py::array_t<FPTYPE, py::array::f_style> getTrialLL() {
            return trialLL_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style> getDW() {
            return dW_numpy;
        }
        py::array_t<FPTYPE, py::array::f_style> getDB() {
            return dB_numpy;
        }
        unsigned int getNumGroups() {
            return groups_shared.size();
        }
        void reset() {
            this->dW->assign(0);
            this->dB->assign(0);
            this->trialLL->assign(0);
            for(int jj = 0; jj < groups_shared.size(); jj++) {
                groups_shared[jj]->reset();
            }
        }


    private:
        py::array_t<FPTYPE, py::array::f_style> dW_numpy;
        py::array_t<FPTYPE, py::array::f_style> dB_numpy;
        py::array_t<FPTYPE, py::array::f_style> trialLL_numpy;
        std::vector<std::shared_ptr<GPUGMLM_group_results_python<FPTYPE>>> groups_shared;
};

// the main interface class
template <class FPTYPE>
class kcGMLM_python {
    public:
        kcGMLM_python(std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure_) {
            structure = structure_;
            resultsWaiting = false;
            kcgmlm = NULL;
            params = std::make_shared<GPUGMLM_params_python<FPTYPE>>(structure);
            msg = std::make_shared<GPUGL_msg_python>();
        }
        inline bool isOnGPU() {
            #ifdef USE_GPU
                return kcgmlm != NULL;
            #else
                return false;
            #endif
        }
        inline void freeGPU() {
            #ifdef USE_GPU
                if(isOnGPU()) {
                    delete kcgmlm;
                    kcgmlm = NULL;
                }
                resultsWaiting = false;
            #endif
        }
        std::shared_ptr<GPUGMLM_trialBlock_python<FPTYPE>>  getBlock(unsigned int block) {
            if(block < blocks_shared.size()) {
                return  blocks_shared[block];
            }
            else {
                throw py::value_error("Invalid block index.");
            }
        }
        int getNumBlocks() {
            return blocks_shared.size();
        }

        int addBlock(std::shared_ptr<GPUGMLM_trialBlock_python<FPTYPE>> block) {
            freeGPU();
            if(!(structure->validateTrialStructure(block.get()))) {
                throw py::value_error("Trial block does not match GMLM structure.");
            }

            blocks_shared.push_back(block);
            blocks.push_back(block.get()); // gross to store both the shared & raw pointer, but my simple API needed the raw pointers and I don't want to change that 
            
            
            // create results
            unsigned int maxTrialIdx = 0;
            for(int bb = 0; bb < blocks_shared.size(); bb++) {
                for(int mm = 0; mm < blocks_shared[bb]->getNumTrials(); mm++) {
                    int trIdx   = blocks_shared[bb]->getTrial(mm)->getTrialNum();
                    maxTrialIdx = trIdx > maxTrialIdx ? trIdx : maxTrialIdx;
                }
            }
            results = std::make_shared<GPUGMLM_results_python<FPTYPE>>(structure, maxTrialIdx + 1);
            
            return blocks.size()-1;
        }
        void toGPU() {
            #ifdef USE_GPU
                freeGPU();

                // send to GPU
                kcgmlm = new GPUGMLM<FPTYPE>(structure.get(), blocks, msg);

                // create options default
                opts = std::make_shared<GPUGMLM_computeOptions_python<FPTYPE>>(structure, kcgmlm->numTrials(), true); 
                
            #else
                throw std::runtime_error("GPU access not available: compiled with CPU only.");
            #endif
        }
        // compute log likelihood: returns trial-wise LL
        std::shared_ptr<GPUGMLM_results_python<FPTYPE>> computeLogLikelihood(std::shared_ptr<GPUGMLM_params_python<FPTYPE>> params_) {
            #ifdef USE_GPU
                if(!(params_->verifyParams(structure))) {
                    throw py::value_error("Parameters object does not match GMLM structure.");
                }
                if(!isOnGPU()) {
                    throw py::value_error("GMLM not on GPU!");
                }
                params = params_;
                results->matchRank(params);
                kcgmlm->computeLogLikelihood(params, opts, results.get());
                resultsWaiting = true;
                return results;
            #else
                throw std::runtime_error("GPU access not available: compiled with CPU only.");
            #endif
        }
        void computeLogLikelihood_async(std::shared_ptr<GPUGMLM_params_python<FPTYPE>> params_) {
            #ifdef USE_GPU
                if(!(params_->verifyParams(structure))) {
                    throw py::value_error("Parameters object does not match GMLM structure.");
                }
                if(!isOnGPU()) {
                    throw py::value_error("GMLM not on GPU!");
                }
                params = params_;
                kcgmlm->computeLogLikelihood_async(params, opts);
                results->matchRank(params);
                resultsWaiting = true;
            #else
                throw std::runtime_error("GPU access not available: compiled with CPU only.");
            #endif
        }
        
        std::shared_ptr<GPUGMLM_results_python<FPTYPE>> getResultsStruct() {
            // returns whatever is in the results structure
            return results;
        }
        std::shared_ptr<GPUGMLM_results_python<FPTYPE>> getResults() {
            #ifdef USE_GPU
                if(!isOnGPU()) {
                    throw py::value_error("GMLM not on GPU!");
                }
                if(!resultsWaiting) {
                    throw py::value_error("No results to collect from GPU! Call 'computeLogLikelihood_async' first.");
                }
                kcgmlm->computeLogLikelihood_gather(results.get(), true);
                return results;
            #else
                throw std::runtime_error("GPU access not available: compiled with CPU only.");
            #endif
        }

        // set options
        void setOptions(std::shared_ptr<GPUGMLM_computeOptions_python<FPTYPE>> opts_) {
            if(!(opts_->verifyOpts(structure))) {
                throw py::value_error("Options object does not match GMLM structure.");
            }
            opts = opts_;
            resultsWaiting = false;
        }
        void setComputeGradient(bool gradOn) {
            if(!isOnGPU()) {
                throw py::value_error("GMLM not on GPU!");
            }
            opts->setGrad(gradOn);
        }
        std::shared_ptr<GPUGMLM_computeOptions_python<FPTYPE>> getOpts() {
            return opts;
        }
        std::shared_ptr<GPUGMLM_params_python<FPTYPE>> getParams() {
            return params;
        }
        std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> getGMLMStructure() {
            return structure;
        }

        ~kcGMLM_python() {
            freeGPU();
        }
    private:
        bool resultsWaiting = false;
        GPUGMLM<FPTYPE> * kcgmlm = NULL;
        std::shared_ptr<GPUGMLM_structure_python<FPTYPE>> structure;
        std::vector<GPUGMLM_GPU_block_args<FPTYPE> *> blocks;
        std::vector<std::shared_ptr<GPUGMLM_trialBlock_python<FPTYPE>>> blocks_shared;
        
        std::shared_ptr<GPUGL_msg_python> msg;

        std::shared_ptr<GPUGMLM_params_python<FPTYPE>> params;
        std::shared_ptr<GPUGMLM_computeOptions_python<FPTYPE>> opts;
        std::shared_ptr<GPUGMLM_results_python<FPTYPE>> results;


        // calls the log likelihood function with parameters given the current opts setup
        //void runComputation(py::array_t<FPTYPE, py::array::f_style | py::array::forcecast> K);

}; //GMLM class

}; //namespace

#endif