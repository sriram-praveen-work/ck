#ifndef TVM_BACKEND_H_
#define TVM_BACKEND_H_

#include <memory>
#include <vector>
#include <cstring>

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>

#include "loadgen.h"
#include "backend.h"

class TVMBackend : public Backend {
public:
    TVMBackend(
            std::shared_ptr<Model> &model, std::shared_ptr<Device> &device,
            size_t performance_sample_count, size_t batch_size,
            bool use_cuda)
            : Backend(model, device, performance_sample_count, batch_size) {

        // Load the TVM module
        tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(model->model_path);
        std::ifstream code(model->model_path + ".ro", std::ios::binary);
        std::stringstream ss;
        ss << code.rdbuf();
        tvm::runtime::Module exec_mod = tvm::runtime::vm::Executable::Load(ss.str(), lib);
        if (exec_mod.operator->() == nullptr) {
            std::cerr << "Failed to load TVM module" << std::endl;
            return;
        }

        for (size_t i = 0; i < device->NumConcurrency(); i++) {
            auto vm = tvm::runtime::make_object<tvm::runtime::vm::VirtualMachine>();
            vm->LoadExecutable(exec_mod);

            // Initialize the VM for the specified device
            DLDevice ctx;
            if (use_cuda) {
                ctx = {kDLCUDA, 0};
            } else {
                ctx = {kDLCPU, 0};
            }

            // Initialize the VM
            int arity = (ctx.device_type == kDLCPU) ? 3 : 6;
            std::vector<TVMValue> init_vals(arity);
            std::vector<int> codes(arity);
            tvm::runtime::TVMArgsSetter setter(init_vals.data(), codes.data());
            setter(0, (int)ctx.device_type);
            setter(1, (int)ctx.device_id);
            setter(2, (uint64_t)tvm::runtime::vm::AllocatorType::kPooled);

            if (ctx.device_type != kDLCPU) {
                setter(3, (int)kDLCPU);
                setter(4, 0);
                setter(5, (uint64_t)tvm::runtime::vm::AllocatorType::kPooled);
            }

            tvm::runtime::TVMRetValue rv;
            vm->GetFunction("init", nullptr).CallPacked(tvm::runtime::TVMArgs(init_vals.data(), codes.data(), arity), &rv);

            // Store the VM and context
            vm_ptrs.emplace_back(vm);

        }
        ctx_ = ctx;
    }

    void RunInference(
            size_t concurrency_index,
            const std::vector<mlperf::QuerySample> &batch,
            std::vector<void *> &batch_data) override {
        
        size_t memory_index = device->GetMemoryIndex(concurrency_index);
        auto vm = vm_ptrs[concurrency_index];
        DLDevice ctx = ctx_;

        for (size_t i = 0; i < model->num_inputs; i++) {
            size_t size = batch.size() * GetSampleSize(batch.front().index, i);
            const std::vector<size_t> &shape = GetSampleShape(batch.front().index, i);

            // Assuming data type is float32
            DLDataType dtype{kDLFloat, 32, 1};
            std::vector<int64_t> input_shape;
            input_shape.push_back(batch.size());
            for (size_t dim : shape)
                input_shape.push_back(dim);

            tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty(
                tvm::runtime::ShapeTuple(input_shape),
                dtype, ctx);

            // Copy data from batch_data to x
            for (size_t j = 0; j < batch.size(); j++) {
                std::memcpy(static_cast<float*>(x->data) + j * shape.size(),
                            static_cast<float*>(batch_data[i]) + j * shape.size(),
                            shape.size() * sizeof(float));
            }

            // set the input
            tvm::runtime::PackedFunc set_input = vm->GetFunction("set_input", nullptr);
            set_input.CallPacked("main", x);

            // Synchronize device
            TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
        }

        // Run inference
        tvm::runtime::PackedFunc run_func = vm->GetFunction("invoke", nullptr);
        // run_func("main");

        // Assuming output is a Tuple of NDArrays representing multiple outputs
        tvm::runtime::ObjectRef out = run_func("main");
        //tvm::runtime::Array<tvm::runtime::NDArray> outputs = out;

        tvm::runtime::NDArray outputs;
        std::vector<tvm::runtime::NDArray> arrays; // Declare a vector to store NDArrays
        if (out.as<tvm::runtime::ADTObj>()) 
        {
            auto adt = tvm::Downcast<tvm::runtime::ADT>(out);
            size_t num_arrays = adt.size(); // Get the number of elements in the ADT
            for (size_t i = 0; i < num_arrays; ++i) 
            {
                // Downcast each element to an NDArray and store it in the vector
                arrays.push_back(tvm::Downcast<tvm::runtime::NDArray>(adt[i]));
            }
            
        } 
            else
        {
            outputs = tvm::Downcast<tvm::runtime::NDArray>(out);
        }

        // IMPLEMENTATION FOR SINGLE o/p TENSOR:
        int ndim = outputs->ndim;
        int tot_dim = 1;

        for (int i = 0; i < ndim; i++)
    {
        tot_dim *= outputs->shape[i];
    }
        auto ssize = ndarray_utils::GetMemSize(outputs);
    //  std::cout<<"size of classifier output: "<<ssize<<std::endl;
        void* data = (void*)malloc(ssize * (outputs->dtype.bits * outputs->dtype.lanes + 7) / 8);
        outputs.CopyToBytes(data, ssize);
        //std::vector<float> inference_output((float *)data, (float *)data + tot_dim);

        // Process output and send responses
        std::vector<mlperf::QuerySampleResponse> responses(batch.size());
        std::vector<std::vector<uint8_t>> response_buffers(batch.size());

        // Determine the total number of elements in all output arrays
        // size_t total_output_elements = 0;
        // for (size_t j = 0; j < outputs.size(); ++j) {
        //     tvm::runtime::NDArray output_array = outputs[j];
        //     total_output_elements += output_array->shape.Size();
        // }



        for (size_t i = 0; i < batch.size(); i++) {
            // get output data and shapes
            std::vector<void *> output_buffers(outputs.size());
            std::vector<std::vector<size_t>> output_shapes(outputs.size());
            for (size_t j = 0; j < outputs.size(); j++) {
                // assume ith position in output is ith sample in batch
                output_buffers[j] =
                    static_cast<uint8_t *>(outputs[j].GetTensorMutableData<void>())
                    + i * model->output_sizes[j];
                size_t rank = outputs[j].GetTensorTypeAndShapeInfo().GetDimensionsCount();
                std::vector<int64_t> output_shape(rank);
                outputs[j].GetTensorTypeAndShapeInfo().GetDimensions(output_shape.data(), rank);
                output_shapes[j].resize(rank);
                for (size_t k = 0; k < rank; k++)
                    output_shapes[j][k] = output_shape[k];
            }

            model->PostProcess(
                batch[i].index, output_buffers, output_shapes, response_buffers[i]);

            responses[i].id = batch[i].id;
            responses[i].data = reinterpret_cast<uintptr_t>(response_buffers[i].data());
            responses[i].size = response_buffers[i].size();
        }

        // // Iterate over each sample in the batch
        // for (size_t i = 0; i < batch.size(); ++i) {
        //     size_t offset = 0;

        //     // Iterate over each output array
        //     for (size_t j = 0; j < outputs.size(); ++j) {
        //         tvm::runtime::NDArray output_array = outputs[j];
        //         const float* output_data = static_cast<float*>(output_array->data);
                
        //         // Assuming output array is flattened
        //         size_t output_size = output_array->shape.Size();
                
        //         // Copy data from output array to response buffer
        //         response_buffers[i].insert(
        //             response_buffers[i].end(),
        //             reinterpret_cast<const uint8_t*>(output_data + offset),
        //             reinterpret_cast<const uint8_t*>(output_data + offset + output_size * sizeof(float))
        //         );

        //         offset += output_size;
        //     }

        //     responses[i].id = batch[i].id;
        //     responses[i].data = reinterpret_cast<uintptr_t>(response_buffers[i].data());
        //     responses[i].size = response_buffers[i].size();
        // }

        // Send responses
        mlperf::QuerySamplesComplete(responses.data(), responses.size());

        
    };

private:
    std::vector<tvm::runtime::vm::VirtualMachinePtr> vm_ptrs;
    DLDevice ctx_;
};

#endif // TVM_BACKEND_H_
