#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <map>
#include <assert.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"



using namespace caffe;
using namespace std;

int main(int argc, char** argv)
{
    Caffe::set_mode(Caffe::GPU);

    string param_file = "examples/stochastic_depth/solver54.prototxt";
    SolverParameter param;
    ReadSolverParamsFromTextFileOrDie(param_file, &param);
    Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);

    // cout << "Stochastic Depth Solver" << endl;

    solver->Solve_StochDep();
}


//--------------------------------------- NET -------------------------------------------------------------------------



template <typename Dtype>
void Net<Dtype>::SetLearnableParams_StochDep() {
    // cout << "SetLearnableParams_StochDep" << endl;
    learnable_params_ids_stochdept_.resize(0);
    for (int i = 0; i < layers_chosen.size(); i++) {
        int layer_id = layers_chosen[i];
        typedef typename map<int, vector<int>* >::const_iterator iter;
        iter pair;
        if (layer_num_to_learnable_params_.count(layer_id) > 0) {
            pair = layer_num_to_learnable_params_idxs.find(layer_id);
            vector<int> idx_vec =  *pair->second;
            for ( int j = 0; j < idx_vec.size() ; j++){
                learnable_params_ids_stochdept_.push_back(idx_vec[j]);
            }
        }
    }
    // cout << "SetLearnableParams_StochDep end" << endl;
}

template <typename Dtype>
void Net<Dtype>::standardResLayer(int & elts, int & idx, double ran, double prob) {
    // cout << "standardResLayer" << endl;
    if (ran < prob){ // include res block
        for (int i = 0; i < 10; i++){
            layerHelper_StochDep(elts, idx, 1, 1, 0, true);
        }
    }
    else{  // skip res block
        layerHelper_StochDep(elts, idx, 10, 1, 10, false);
      ////  cout << "skipping block: " << elts << endl;
    }
    // cout << "standardResLayer end" << endl;
}

template <typename Dtype>
void Net<Dtype>::transitionResLayer(int & elts, int& idx, double ran, double prob){
    // cout << "transitionResLayer" << endl;
    if (ran < prob) { //include res block
        for (int i = 0; i < 13; i++) {
            layerHelper_StochDep(elts, idx, 1, 1, 0, true);
        }
    }
    else { // skip res block
        layerHelper_StochDep(elts, idx, 2, 1, 2, false);
        layerHelper_StochDep(elts, idx, 1, 1, 0, true);
        layerHelper_StochDep(elts, idx, 1, 1, 0, true);
        layerHelper_StochDep(elts, idx, 9, 1, 9, false);
		//// cout << "skipping block: " << elts << endl;
    }
    // cout << "transitionResLayer end" << endl;
}

template <typename Dtype>
void Net<Dtype>::layerHelper_StochDep(int & elts, int& idx, int elt_incr, int idx_incr, int bottom_incr, bool use_top) {
    // cout << "layerHelper_StochDep" << endl;
    bottom_vecs_stochdept_[idx] = bottom_vecs_[elts];
    if (use_top) {
        top_vecs_stochdept_[idx] = top_vecs_[elts + bottom_incr];}
    else {
        top_vecs_stochdept_[idx] = bottom_vecs_[elts + bottom_incr];}
    layers_chosen[idx] = elts;
    elts += elt_incr;
    idx += idx_incr;
    // cout << "layerHelper_StochDep end" << endl;
}


template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo_StochDep() {
    // cout << "ForwardFromTo_StochDep" << endl;
    Dtype loss = 0;
    int layer_idx;
    for (int i = 0; i < layers_chosen.size(); i++) {
        layer_idx = layers_chosen[i];
        shared_ptr<Layer<Dtype> > curr_layer = layers_[layer_idx];

        vector<Blob<Dtype>*> curr_bottom = bottom_vecs_stochdept_[i];
        vector<Blob<Dtype>*> curr_top = top_vecs_stochdept_[i];

        Dtype layer_loss = curr_layer->Forward(curr_bottom, curr_top);
        loss += layer_loss;
        if (debug_info_) { ForwardDebugInfo(layer_idx); }
    }
    // cout << "ForwardFromTo_StochDep end" << endl;
    return loss;
}

template<typename Dtype>
void Net<Dtype>::printvecblobs(vector<vector<Blob<Dtype>*> > vec, int &idx) {
    // cout << "printvecblobs" << endl;
    for (int i = 0; i < vec[idx].size(); i++) {
        Blob<Dtype>* blo= vec[idx][i];
        //// cout << blo->shape(0) << " " << blo->shape(1) << " " << blo->shape(2) << " " <<  blo->shape(3)  << endl;
        //// cout << blo << endl;
    }
    // cout << "printvecblobs end" << endl;
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo_StochDep() {
    // cout << "BackwardFromTo_StochDep" << endl;
    int layer_idx;
    for (int i = layers_chosen.size() - 1; i >= 0; i--) {
        layer_idx = layers_chosen[i];
        if (layer_need_backward_[layer_idx]) {
            layers_[layer_idx]->Backward(top_vecs_stochdept_[i], bottom_need_backward_[layer_idx], bottom_vecs_stochdept_[i]);
            if (debug_info_) { BackwardDebugInfo(layer_idx); }
        }
    }
    // cout << "BackwardFromTo_StochDep end" << endl;
}

template<typename Dtype>
Dtype Net<Dtype>::ForwardBackward_StochDep() {
    // cout << "ForwardBackward_StochDep" << endl;
    Dtype loss;
    Forward_StochDep(&loss);
    Backward_StochDep();
    // cout << "ForwardBackward_StochDep end" << endl;
    return loss;
}

template<typename Dtype>
void Net<Dtype>::ChooseLayers_StochDep(){
    // cout << "ChooseLayers_StochDep" << endl;
    bottom_vecs_stochdept_.resize(this->layers().size());
    top_vecs_stochdept_.resize(this->layers().size());
    layers_chosen.resize(this->layers().size());
    int elts = 0;
    int idx = 0;
    for (int i = 0; i < 4; i++){
        layerHelper_StochDep(elts, idx, 1, 1, 0, true);
        }
    srand((unsigned)time(NULL));
    double block_num  = 0;
    double num_blocks = 53;
    double prob;
    double ran;

    for (int j = 0; j < 18; j++) {
        ran = (double) rand()/RAND_MAX;
        prob = 1 - 0.5*(block_num)/num_blocks;
        standardResLayer(elts, idx, ran, prob);
        block_num += 1.0;
    }
    ran = (double) rand()/RAND_MAX;
    prob = 1 - 0.5*(block_num)/num_blocks;
    transitionResLayer(elts, idx, ran, prob);
    block_num += 1.0;

    for (int j = 0; j < 17; j++) {
        ran = (double) rand()/RAND_MAX;
        prob = 1 - 0.5*(block_num)/num_blocks;
        standardResLayer(elts, idx, ran, prob);
        block_num += 1.0;
    }
    ran = (double) rand()/RAND_MAX;
    prob = 1 - 0.5*(block_num)/num_blocks;
    transitionResLayer(elts, idx, ran, prob);
    block_num += 1.0;

    for (int j = 0; j < 17; j++) {
        ran = (double) rand()/RAND_MAX;
        prob = 1 - 0.5*(block_num)/num_blocks;
        standardResLayer(elts, idx, ran, prob);
        block_num += 1.0;
    }

    for (int i = 0; i < 4; i++) {
        layerHelper_StochDep(elts, idx, 1, 1, 0, true);
    }
    bottom_vecs_stochdept_.resize(idx);
    top_vecs_stochdept_.resize(idx);
    layers_chosen.resize(idx);
    // cout << "ChooseLayers_StochDep end" << endl;
}

template <typename Dtype>
void Net<Dtype>::Backward_StochDep() {
    // cout << "Backward_StochDep" << endl;
    BackwardFromTo_StochDep();
    if (debug_info_) {
        Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
        for (int i = 0; i < learnable_params_ids_stochdept_.size(); ++i) {
            int param_id = learnable_params_ids_stochdept_[i];
            asum_data += learnable_params_[param_id]->asum_data();
            asum_diff += learnable_params_[param_id]->asum_diff();
            sumsq_data += learnable_params_[param_id]->sumsq_data();
            sumsq_diff += learnable_params_[param_id]->sumsq_diff();
        }
        const Dtype l2norm_data = std::sqrt(sumsq_data);
        const Dtype l2norm_diff = std::sqrt(sumsq_diff);
        LOG(ERROR) << "    [Backward] All net params (data, diff): "
        << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
        << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
    }
    // cout << "Backward_StochDep end" << endl;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward_StochDep(Dtype* loss) {
    // cout << "Forward_StochDep" << endl;
    if (loss != NULL) {
        *loss = ForwardFromTo_StochDep();
    }
    else {
        ForwardFromTo_StochDep();
    }
    // cout << "Forward_StochDep end" << endl;
    return net_output_blobs_;
}


template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward_StochDep_Test(Dtype* loss) {
    // cout << "Forward_StochDep_Test" << endl;
    if (loss != NULL) {
        *loss = ForwardFromTo_StochDep_Test(0, layers_.size() - 1);
    }
    else {
        ForwardFromTo_StochDep_Test(0, layers_.size() - 1);
    }
    // cout << "Forward_StochDep_Test end" << endl;
    return net_output_blobs_;
}


template <typename Dtype>
const Dtype Net<Dtype>::ForwardFromTo_StochDep_Test(int start, int end) {
    // cout << "ForwardFromTo_StochDep_Test" << endl;
    CHECK_GE(start, 0);
    CHECK_LT(end, layers_.size());
    Dtype loss = 0;
    for (int i = start; i <= end; ++i) {
//         LOG(ERROR) << "Forwarding " << layer_names_[i];
//    // cout << layers_[i]->type() << i << "\t bottom size: " <<  bottom_vecs_[i].size() << endl;
        Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
        loss += layer_loss;
        vector<Blob<Dtype>*> top_vec = top_vecs_[i];
        double prob = test_scaling_stochdept_[i];
        if (prob < 1.0) {
            cout << "prob test: " << prob << endl;
            for (int j = 0; j < top_vec.size(); j++) {
                Blob<Dtype> *top_blob = top_vec[j];
//                // cout <<"i: " << i << "\t j: " << j << '\t' << layers_[i]->type() << "\t prob: " << prob << endl;
                top_blob->scale_data(prob);
            }
        }
        if (debug_info_) { ForwardDebugInfo(i); }
    }
    // cout << "ForwardFromTo_StochDep_Test end" << endl;
    return loss;
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs_StochDep() {
    // cout << "ClearParamDiffs_StochDep" << endl;
    const vector<int>& learnable_params_ids = learnable_params_ids_stochdept();
    for (int i = 0; i < learnable_params_ids.size(); i++) {
        int param_id = learnable_params_ids[i];
        Blob<Dtype>* blob = learnable_params_[param_id];
        switch (Caffe::mode()) {
            case Caffe::CPU:
                caffe_set(blob->count(), static_cast<Dtype>(0),
                          blob->mutable_cpu_diff());
                break;
            case Caffe::GPU:
#ifndef CPU_ONLY
                caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                              blob->mutable_gpu_diff());
#else
                NO_GPU;
#endif
                break;
        }
    }
    // cout << "ClearParamDiffs_StochDep end" << endl;
}


//----------------------------------------- SOLVER ---------------------------------------------------------------------

template <typename Dtype>
void Solver<Dtype>::Step_StochDep(int iters) {
    // cout << "Step_StochDep" << endl;
    const int start_iter = iter_;
    const int stop_iter = iter_ + iters;
    int average_loss = this->param_.average_loss();
    losses_.clear();
    smoothed_loss_ = 0;

    while (iter_ < stop_iter) {
        // zero-init the params
        net_->ChooseLayers_StochDep();
        net_->SetLearnableParams_StochDep();
        net_->ClearParamDiffs_StochDep();
        if (param_.test_interval() && iter_ % param_.test_interval() == 0
            && (iter_ > 0 || param_.test_initialization())
            && Caffe::root_solver()) {
            TestAll_StochDep();
            if (requested_early_exit_) {
                // Break out of the while loop because stop was requested while testing.
                break;
            }
        }

        for (int i = 0; i < callbacks_.size(); ++i) {
            cout << "--------------------callback1 called in step_stochdep-------------------" << endl;
            callbacks_[i]->on_start();
        }
        const bool display = param_.display() && iter_ % param_.display() == 0;
        net_->set_debug_info(display && param_.debug_info());
        // accumulate the loss and gradient
        Dtype loss = 0;

        for (int i = 0; i < param_.iter_size(); ++i) {
            loss += net_->ForwardBackward_StochDep();
        }
        loss /= param_.iter_size();
        // average the loss across iterations for smoothed reporting
        UpdateSmoothedLoss(loss, start_iter, average_loss);
        if (display) {
            LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
            << ", loss = " << smoothed_loss_;
            const vector<Blob<Dtype>*>& result = net_->output_blobs();
            int score_index = 0;
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                const string& output_name =
                        net_->blob_names()[net_->output_blob_indices()[j]];
                const Dtype loss_weight =
                        net_->blob_loss_weights()[net_->output_blob_indices()[j]];
                for (int k = 0; k < result[j]->count(); ++k) {
                    ostringstream loss_msg_stream;
                    if (loss_weight) {
                        loss_msg_stream << " (* " << loss_weight
                        << " = " << loss_weight * result_vec[k] << " loss)";
                    }
                    LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
                    << score_index++ << ": " << output_name << " = "
                    << result_vec[k] << loss_msg_stream.str();
                }
            }
        }
        for (int i = 0; i < callbacks_.size(); ++i) {
            cout << "--------------------callback2 called in step_stochdep-------------------" << endl  ;
            callbacks_[i]->on_gradients_ready();
        }
        ApplyUpdate_StochDep();

        // Increment the internal iter_ counter -- its value should always indicate
        // the number of times the weights have been updated.
        ++iter_;

        SolverAction::Enum request = GetRequestedAction();

        // Save a snapshot if needed.
        if ((param_.snapshot()
             && iter_ % param_.snapshot() == 0
             && Caffe::root_solver()) ||
            (request == SolverAction::SNAPSHOT)) {
            Snapshot();
        }
        if (SolverAction::STOP == request) {
            requested_early_exit_ = true;
            // Break out of training loop.
            break;
        }
    }
    // cout << "Step_StochDep end" << endl;
}


template <typename Dtype>
void Solver<Dtype>::Solve_StochDep(const char* resume_file) {
    // cout << "Solve_StochDep" << endl;
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Solving " << net_->name();
    LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

    // Initialize to false every time we start solving.
    requested_early_exit_ = false;

    if (resume_file) {
        LOG(INFO) << "Restoring previous solver status from " << resume_file;
        Restore(resume_file);
    }

    // For a network that is trained by the solver, no bottom or top vecs
    // should be given, and we will just provide dummy vecs.
    int start_iter = iter_;
    Step_StochDep(param_.max_iter() - iter_);
    // If we haven't already, save a snapshot after optimization, unless
    // overridden by setting snapshot_after_train := false
    if (param_.snapshot_after_train()
        && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
        Snapshot();
    }
    if (requested_early_exit_) {
        LOG(INFO) << "Optimization stopped early.";
        return;
    }
    // After the optimization is done, run an additional train and test pass to
    // display the train and test loss/outputs if appropriate (based on the
    // display and test_interval settings, respectively).  Unlike in the rest of
    // training, for the train net we only run a forward pass as we've already
    // updated the parameters "max_iter" times -- this final pass is only done to
    // display the loss, which is computed in the forward pass.
    if (param_.display() && iter_ % param_.display() == 0) {
        int average_loss = this->param_.average_loss();
        Dtype loss;
        net_->Forward_StochDep_Test(&loss);

        UpdateSmoothedLoss(loss, start_iter, average_loss);

        LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
        TestAll_StochDep();
    }
    LOG(INFO) << "Optimization Done.";
    // cout << "Solve_StochDep end" << endl;
}


template <typename Dtype>
void Solver<Dtype>::TestAll_StochDep() {
    // cout << "TestAll_StochDep" << endl;
    for (int test_net_id = 0; test_net_id < test_nets_.size() && !requested_early_exit_; ++test_net_id) {
        Test_StochDep(test_net_id);
    }
    // cout << "TestAll_StochDep end" << endl;
}


template <typename Dtype>
void Solver<Dtype>::Test_StochDep(const int test_net_id) {
    // cout << "Test_StochDep" << endl;
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Iteration " << iter_
    << ", Testing net (#" << test_net_id << ")";
    CHECK_NOTNULL(test_nets_[test_net_id].get())->
            ShareTrainedLayersWith(net_.get());
    vector<Dtype> test_score;
    vector<int> test_score_output_id;
    const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
    Dtype loss = 0;
    for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
        SolverAction::Enum request = GetRequestedAction();
        // Check to see if stoppage of testing/training has been requested.
        while (request != SolverAction::NONE) {
            if (SolverAction::SNAPSHOT == request) {
                Snapshot();
            } else if (SolverAction::STOP == request) {
                requested_early_exit_ = true;
            }
            request = GetRequestedAction();
        }
        if (requested_early_exit_) {
            // break out of test loop.
            break;
        }

        Dtype iter_loss;
        const vector<Blob<Dtype>*>& result =
                                          test_net->Forward_StochDep_Test(&iter_loss);
        if (param_.test_compute_loss()) {
            loss += iter_loss;
        }
        if (i == 0) {
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                for (int k = 0; k < result[j]->count(); ++k) {
                    test_score.push_back(result_vec[k]);
                    test_score_output_id.push_back(j);
                }
            }
        } else {
            int idx = 0;
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                for (int k = 0; k < result[j]->count(); ++k) {
                    test_score[idx++] += result_vec[k];
                }
            }
        }
    }
    if (requested_early_exit_) {
        LOG(INFO)     << "Test interrupted.";
        return;
    }
    if (param_.test_compute_loss()) {
        loss /= param_.test_iter(test_net_id);
        LOG(INFO) << "Test loss: " << loss;
    }
    for (int i = 0; i < test_score.size(); ++i) {
        const int output_blob_index =
                test_net->output_blob_indices()[test_score_output_id[i]];
        const string& output_name = test_net->blob_names()[output_blob_index];
        const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
        ostringstream loss_msg_stream;
        const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
            << " = " << loss_weight * mean_score << " loss)";
        }
        LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
    }
    // cout << "Test_StochDep end" << endl;
}



//------------------------------------------------ SGD SOLVER ----------------------------------------------------------
