#ifndef TVM_BACKEND_H_
#define TVM_BACKEND_H_

#include <memory>
#include <vector>
#include <cstring>
#include <filesystem>

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>

#include "loadgen.h"
#include "backend.h"

inline size_t GetMemSize(tvm::runtime::NDArray& narr) {
   size_t size = 1;
   for (tvm_index_t i = 0; i < narr->ndim; ++i) {
      size *= static_cast<size_t>(narr->shape[i]);
   }
   size *= (narr->dtype.bits * narr->dtype.lanes + 7) / 8;
   return size;
}

class TVMBackend : public Backend {
public:
    TVMBackend(
            std::shared_ptr<Model> &model, std::shared_ptr<Device> &device,
            size_t performance_sample_count, size_t batch_size,
            bool use_cuda)
            : Backend(model, device, performance_sample_count, batch_size) {

        // Load the TVM module
        std::filesystem::path p = model->model_path;
        std::string modelPath = p.replace_extension().string();
        std::string vmModelPath = modelPath + ".so";
        std::string vmConstsPath = modelPath + ".const";
        std::string vmExecCodePath = modelPath + ".ro";
        
        tvm::runtime::Module vmLib = tvm::runtime::Module::LoadFromFile(vmModelPath);

        std::ifstream code(vmExecCodePath, std::ios::binary);
        std::cout<<vmExecCodePath<<std::endl;
        std::stringstream ss;
        ss << code.rdbuf();

        tvm::runtime::Module vmExecMod = tvm::runtime::vm::Executable::Load(ss.str(), vmLib);
        if (vmExecMod.get() == nullptr)
        {
            std::cout << "Failed to load module" << std::endl;
            //return -1;
        }
        const tvm::runtime::vm::Executable* tmp = vmExecMod.as<tvm::runtime::vm::Executable>();  
        auto vmExec = tvm::runtime::GetObjectPtr<tvm::runtime::vm::Executable>(const_cast<tvm::runtime::vm::Executable*>(tmp));
        vmExec->LoadLateBoundConstantsFromFile(vmConstsPath);
        // std::cout << "consts  loaded\n";
        // auto vmMod = tvm::runtime::make_object<tvm::runtime::vm::VirtualMachine>();
        // vmMod->LoadExecutable(vmExec);

        

        for (size_t i = 0; i < device->NumConcurrency(); i++) {
            auto vmMod = tvm::runtime::make_object<tvm::runtime::vm::VirtualMachine>();
            vmMod->LoadExecutable(vmExec);

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
            setter(2, (uint64_t)tvm::runtime::AllocatorType::kPooled);

            if (ctx.device_type != kDLCPU) {
                setter(3, (int)kDLCPU);
                setter(4, 0);
                setter(5, (uint64_t)tvm::runtime::AllocatorType::kPooled);
            }

            tvm::runtime::TVMRetValue rv;
            vmMod->GetFunction("init", nullptr).CallPacked(tvm::runtime::TVMArgs(init_vals.data(), codes.data(), arity), &rv);

            // Store the VM and context
            vmMod_list.emplace_back(vmMod);
            ctx_list.emplace_back(ctx);

        }
    }

    void RunInference(
            size_t concurrency_index,
            const std::vector<mlperf::QuerySample> &batch,
            std::vector<void *> &batch_data) override {
        
        std::string ENTRY_FUNCTION = "main";

        // size_t memory_index = device->GetMemoryIndex(concurrency_index);
        auto vmMod = vmMod_list[concurrency_index];
        DLDevice ctx = ctx_list[concurrency_index];

        const size_t num_args = model->num_inputs + 1;
        std::vector<TVMValue> values(num_args);
        std::vector<int> type_codes(num_args);
        tvm::runtime::TVMArgsSetter arg_setter(values.data(), type_codes.data());
        arg_setter(0, ENTRY_FUNCTION);
        std::vector<DLTensor*> input_tensors;
        for (size_t i = 0; i < model->num_inputs; i++) {
            size_t size = batch.size() * GetSampleSize(batch.front().index, i);
            const std::vector<size_t> &shape = GetSampleShape(batch.front().index, i);
            std::vector<int64_t> input_shape;
            input_shape.push_back(batch.size());
            for (size_t dim : shape)
                input_shape.push_back(dim);
            auto get_input_index = vmMod->GetFunction("get_input_index", nullptr);
            int inp_index = get_input_index(model->input_names[i], ENTRY_FUNCTION);
            // std::cout << "Input Index: "<<inp_index << std::endl;
            // auto dtype = tvm::runtime::String2DLDataType("float32");
            // tvm::runtime::NDArray inp_ndarray = tvm::runtime::NDArray::Empty(input_shape, dtype, ctx);
            // inp_ndarray.CopyFromBytes(batch_data[i], size);
            DLTensor *inp_array;
            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int64_t * in_shape = input_shape.data();
            int in_ndim = input_shape.size();
            int nbytes_float32 = 4;
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, (int)ctx.device_type, ctx.device_id, &inp_array);
            TVMArrayCopyFromBytes(inp_array,batch_data[i], size);
            arg_setter(inp_index + 1, inp_array);
            input_tensors.emplace_back(inp_array);
        }

        tvm::runtime::PackedFunc set_input = vmMod->GetFunction("set_input", nullptr);
        tvm::runtime::TVMRetValue rv;
        set_input.CallPacked(tvm::runtime::TVMArgs(values.data(), type_codes.data(), int(num_args)), &rv);

        
        // Synchronize device
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
        

        // Run inference
        tvm::runtime::PackedFunc run_func = vmMod->GetFunction("invoke", nullptr);
        // run_func("main");

        // Assuming output is a Tuple of NDArrays representing multiple outputs
        tvm::runtime::ObjectRef out = run_func(ENTRY_FUNCTION);
        //tvm::runtime::Array<tvm::runtime::NDArray> outputs = out;
        for (size_t i = 0; i < model->num_inputs; i++) {
            TVMArrayFree(input_tensors[i]);
        }

        //tvm::runtime::NDArray outputs;
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
            //outputs = tvm::Downcast<tvm::runtime::NDArray>(out);
            arrays.push_back(tvm::Downcast<tvm::runtime::NDArray>(out));
        }    

        // Process output and send responses
        std::vector<mlperf::QuerySampleResponse> responses(batch.size());
        std::vector<std::vector<uint8_t>> response_buffers(batch.size());
        // std::cout << "Output Tensors Count: "<<arrays.size() << std::endl;

        arrays.pop_back();

        // for (size_t j = 0; j < arrays.size(); j++) {
        //     for (int i = 0; i < arrays[j]->ndim; i++) {
        //         std::cout<<arrays[j]->shape[i]<<", ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl;


        for (size_t i = 0; i < batch.size(); i++) {
            // Iterate over the output arrays
            std::vector<void *> output_buffers(arrays.size());
            std::vector<std::vector<size_t>> output_shapes(arrays.size()); 
            for (size_t j = 0; j < arrays.size(); j++) {
                tvm::runtime::NDArray output_array = arrays[j];
                std::vector<size_t> output_shape(output_array->ndim);
                // Calculate the total number of elements in the output array
                int total_elements = 1;
                for (int k = 0; k < output_array->ndim; k++) {
                    total_elements *= output_array->shape[k];
                    output_shape[k] = output_array->shape[k];
                }
                output_shapes[j] = output_shape;

                // Allocate memory for output buffer and copy data from the output array
                auto output_size = GetMemSize(output_array);
                // size_t output_size = total_elements * (output_array->dtype.bits * output_array->dtype.lanes + 7) / 8;
                // std::cout<<"Total Elem: "<<total_elements<<"\tOutput Size: "<<output_size<<std::endl;
                void* output_buffer = malloc(output_size);
                output_array.CopyToBytes(static_cast<uint8_t *>(output_buffer) + i * model->output_sizes[j], output_size);

                // Store output buffer and shape information
                output_buffers[j] = output_buffer;
                
            }

            // Post-process outputs and prepare response
            model->PostProcess(batch[i].index, output_buffers, output_shapes, response_buffers[i]);

            for (size_t j = 0; j < arrays.size(); j++) {
                free(output_buffers[j]);
                }

            responses[i].id = batch[i].id;
            responses[i].data = reinterpret_cast<uintptr_t>(response_buffers[i].data());
            responses[i].size = response_buffers[i].size();
        }

        // Send responses
        mlperf::QuerySamplesComplete(responses.data(), responses.size());

        
    };

private:
    std::vector<tvm::runtime::ObjectPtr<tvm::runtime::vm::VirtualMachine>> vmMod_list;
    std::vector<DLDevice> ctx_list;
};

#endif // TVM_BACKEND_H_
