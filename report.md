# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | hubconf.py:6 | add "import torch_mlu" |
| 2 | setup.py:10 | add "import torch_mlu" |
| 3 | yolox/exp/yolox_base.py:7 | add "import torch_mlu" |
| 4 | yolox/exp/yolox_base.py:225 | change "tensor = torch.LongTensor(2).cuda()" to "tensor = torch.LongTensor(2).mlu() " |
| 5 | yolox/exp/base_exp.py:10 | add "import torch_mlu" |
| 6 | yolox/evaluators/coco_evaluator.py:18 | add "import torch_mlu" |
| 7 | yolox/evaluators/coco_evaluator.py:135 | change "tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor" to "tensor_type = torch.mlu.HalfTensor if half else torch.mlu.FloatTensor " |
| 8 | yolox/evaluators/coco_evaluator.py:154 | change "x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()" to "x = torch.ones(1, 3, test_size[0], test_size[1]).mlu() " |
| 9 | yolox/evaluators/coco_evaluator.py:189 | change "statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])" to "statistics = torch.mlu.FloatTensor([inference_time, nms_time, n_samples]) " |
| 10 | yolox/evaluators/voc_evaluator.py:14 | add "import torch_mlu" |
| 11 | yolox/evaluators/voc_evaluator.py:60 | change "tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor" to "tensor_type = torch.mlu.HalfTensor if half else torch.mlu.FloatTensor " |
| 12 | yolox/evaluators/voc_evaluator.py:78 | change "x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()" to "x = torch.ones(1, 3, test_size[0], test_size[1]).mlu() " |
| 13 | yolox/evaluators/voc_evaluator.py:108 | change "statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])" to "statistics = torch.mlu.FloatTensor([inference_time, nms_time, n_samples]) " |
| 14 | yolox/models/yolo_pafpn.py:5 | add "import torch_mlu" |
| 15 | yolox/models/build.py:4 | add "import torch_mlu" |
| 16 | yolox/models/build.py:53 | change "device = "cuda:0" if torch.cuda.is_available() else "cpu"" to "device = "mlu:0" if torch.mlu.is_available() else "cpu" " |
| 17 | yolox/models/losses.py:5 | add "import torch_mlu" |
| 18 | yolox/models/yolo_fpn.py:5 | add "import torch_mlu" |
| 19 | yolox/models/network_blocks.py:5 | add "import torch_mlu" |
| 20 | yolox/models/yolo_head.py:8 | add "import torch_mlu" |
| 21 | yolox/models/yolo_head.py:324 | change "if "CUDA out of memory. " not in str(e):" to "if "MLU out of memory. " not in str(e): " |
| 22 | yolox/models/yolo_head.py:325 | change "raise  # RuntimeError might not caused by CUDA OOM" to "raise  # RuntimeError might not caused by MLU OOM " |
| 23 | yolox/models/yolo_head.py:332 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 24 | yolox/models/yolo_head.py:353 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 25 | yolox/models/yolo_head.py:435 | change "mode="gpu"," to "mode="mlu", " |
| 26 | yolox/models/yolo_head.py:474 | change "with torch.cuda.amp.autocast(enabled=False):" to "with torch.mlu.amp.autocast(enabled=False): " |
| 27 | yolox/models/yolo_head.py:500 | change "gt_matched_classes = gt_matched_classes.cuda()" to "gt_matched_classes = gt_matched_classes.mlu() " |
| 28 | yolox/models/yolo_head.py:501 | change "fg_mask = fg_mask.cuda()" to "fg_mask = fg_mask.mlu() " |
| 29 | yolox/models/yolo_head.py:502 | change "pred_ious_this_matching = pred_ious_this_matching.cuda()" to "pred_ious_this_matching = pred_ious_this_matching.mlu() " |
| 30 | yolox/models/yolo_head.py:503 | change "matched_gt_inds = matched_gt_inds.cuda()" to "matched_gt_inds = matched_gt_inds.mlu() " |
| 31 | yolox/models/yolo_head.py:519 | change "the number of candidate anchors so that the GPU memory is saved." to "the number of candidate anchors so that the MLU memory is saved. " |
| 32 | yolox/core/launch.py:12 | add "import torch_mlu" |
| 33 | yolox/core/launch.py:41 | change "num_gpus_per_machine," to "num_mlus_per_machine, " |
| 34 | yolox/core/launch.py:44 | change "backend="nccl"," to "backend="cncl", " |
| 35 | yolox/core/launch.py:59 | change "world_size = num_machines * num_gpus_per_machine" to "world_size = num_machines * num_mlus_per_machine " |
| 36 | yolox/core/launch.py:84 | change "nprocs=num_gpus_per_machine," to "nprocs=num_mlus_per_machine, " |
| 37 | yolox/core/launch.py:88 | change "num_gpus_per_machine," to "num_mlus_per_machine, " |
| 38 | yolox/core/launch.py:105 | change "num_gpus_per_machine," to "num_mlus_per_machine, " |
| 39 | yolox/core/launch.py:113 | change "torch.cuda.is_available()" to "torch.mlu.is_available() " |
| 40 | yolox/core/launch.py:114 | change "), "cuda is not available. Please check your installation."" to "), "mlu is not available. Please check your installation." " |
| 41 | yolox/core/launch.py:115 | change "global_rank = machine_rank * num_gpus_per_machine + local_rank" to "global_rank = machine_rank * num_mlus_per_machine + local_rank " |
| 42 | yolox/core/launch.py:131 | change "num_machines = world_size // num_gpus_per_machine" to "num_machines = world_size // num_mlus_per_machine " |
| 43 | yolox/core/launch.py:134 | change "range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)" to "range(i * num_mlus_per_machine, (i + 1) * num_mlus_per_machine) " |
| 44 | yolox/core/launch.py:144 | change "assert num_gpus_per_machine <= torch.cuda.device_count()" to "assert num_mlus_per_machine <= torch.mlu.device_count() " |
| 45 | yolox/core/launch.py:145 | change "torch.cuda.set_device(local_rank)" to "torch.mlu.set_device(local_rank) " |
| 46 | yolox/core/trainer.py:9 | add "import torch_mlu" |
| 47 | yolox/core/trainer.py:25 | change "gpu_mem_usage," to "mlu_mem_usage, " |
| 48 | yolox/core/trainer.py:46 | change "self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)" to "self.scaler = torch.mlu.amp.GradScaler(enabled=args.fp16) " |
| 49 | yolox/core/trainer.py:50 | change "self.device = "cuda:{}".format(self.local_rank)" to "self.device = "mlu:{}".format(self.local_rank) " |
| 50 | yolox/core/trainer.py:104 | change "with torch.cuda.amp.autocast(enabled=self.amp_training):" to "with torch.mlu.amp.autocast(enabled=self.amp_training): " |
| 51 | yolox/core/trainer.py:134 | change "torch.cuda.set_device(self.local_rank)" to "torch.mlu.set_device(self.local_rank) " |
| 52 | yolox/core/trainer.py:253 | change "mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())" to "mem_str = "mlu mem: {:.0f}Mb, mem: {:.1f}Gb".format(mlu_mem_usage(), mem_usage()) " |
| 53 | yolox/data/dataloading.py:11 | add "import torch_mlu" |
| 54 | yolox/data/samplers.py:8 | add "import torch_mlu" |
| 55 | yolox/data/data_prefetcher.py:5 | add "import torch_mlu" |
| 56 | yolox/data/data_prefetcher.py:18 | change "self.stream = torch.cuda.Stream()" to "self.stream = torch.mlu.Stream() " |
| 57 | yolox/data/data_prefetcher.py:19 | change "self.input_cuda = self._input_cuda_for_image" to "self.input_mlu = self._input_mlu_for_image " |
| 58 | yolox/data/data_prefetcher.py:31 | change "with torch.cuda.stream(self.stream):" to "with torch.mlu.stream(self.stream): " |
| 59 | yolox/data/data_prefetcher.py:32 | change "self.input_cuda()" to "self.input_mlu() " |
| 60 | yolox/data/data_prefetcher.py:33 | change "self.next_target = self.next_target.cuda(non_blocking=True)" to "self.next_target = self.next_target.mlu(non_blocking=True) " |
| 61 | yolox/data/data_prefetcher.py:36 | change "torch.cuda.current_stream().wait_stream(self.stream)" to "torch.mlu.current_stream().wait_stream(self.stream) " |
| 62 | yolox/data/data_prefetcher.py:42 | change "target.record_stream(torch.cuda.current_stream())" to "target.record_stream(torch.mlu.current_stream()) " |
| 63 | yolox/data/data_prefetcher.py:46 | change "def _input_cuda_for_image(self):" to "def _input_mlu_for_image(self): " |
| 64 | yolox/data/data_prefetcher.py:47 | change "self.next_input = self.next_input.cuda(non_blocking=True)" to "self.next_input = self.next_input.mlu(non_blocking=True) " |
| 65 | yolox/data/data_prefetcher.py:51 | change "input.record_stream(torch.cuda.current_stream())" to "input.record_stream(torch.mlu.current_stream()) " |
| 66 | yolox/utils/compat.py:4 | add "import torch_mlu" |
| 67 | yolox/utils/checkpoint.py:8 | add "import torch_mlu" |
| 68 | yolox/utils/logger.py:13 | add "import torch_mlu" |
| 69 | yolox/utils/logger.py:86 | change "distributed_rank(int): device rank when multi-gpu environment" to "distributed_rank(int): device rank when multi-mlu environment " |
| 70 | yolox/utils/setup_env.py:13 | change "__all__ = ["configure_nccl", "configure_module", "configure_omp"]" to "__all__ = ["configure_cncl", "configure_module", "configure_omp"] " |
| 71 | yolox/utils/setup_env.py:16 | change "def configure_nccl():" to "def configure_cncl(): " |
| 72 | yolox/utils/boxes.py:6 | add "import torch_mlu" |
| 73 | yolox/utils/model_utils.py:9 | add "import torch_mlu" |
| 74 | yolox/utils/ema.py:7 | add "import torch_mlu" |
| 75 | yolox/utils/ema.py:30 | change "GPU assignment and distributed training wrappers." to "MLU assignment and distributed training wrappers. " |
| 76 | yolox/utils/metric.py:12 | add "import torch_mlu" |
| 77 | yolox/utils/metric.py:19 | change ""gpu_mem_usage"," to ""mlu_mem_usage", " |
| 78 | yolox/utils/metric.py:24 | change "def get_total_and_free_memory_in_Mb(cuda_device):" to "def get_total_and_free_memory_in_Mb(mlu_device): " |
| 79 | yolox/utils/metric.py:26 | change ""nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"" to ""nvidia-smi --query-mlu=memory.total,memory.used --format=csv,nounits,noheader" " |
| 80 | yolox/utils/metric.py:29 | change "if "CUDA_VISIBLE_DEVICES" in os.environ:" to "if "MLU_VISIBLE_DEVICES" in os.environ: " |
| 81 | yolox/utils/metric.py:30 | change "visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')" to "visible_devices = os.environ["MLU_VISIBLE_DEVICES"].split(',') " |
| 82 | yolox/utils/metric.py:31 | change "cuda_device = int(visible_devices[cuda_device])" to "mlu_device = int(visible_devices[mlu_device]) " |
| 83 | yolox/utils/metric.py:32 | change "total, used = devices_info[int(cuda_device)].split(",")" to "total, used = devices_info[int(mlu_device)].split(",") " |
| 84 | yolox/utils/metric.py:36 | change "def occupy_mem(cuda_device, mem_ratio=0.9):" to "def occupy_mem(mlu_device, mem_ratio=0.9): " |
| 85 | yolox/utils/metric.py:38 | change "pre-allocate gpu memory for training to avoid memory Fragmentation." to "pre-allocate mlu memory for training to avoid memory Fragmentation. " |
| 86 | yolox/utils/metric.py:40 | change "total, used = get_total_and_free_memory_in_Mb(cuda_device)" to "total, used = get_total_and_free_memory_in_Mb(mlu_device) " |
| 87 | yolox/utils/metric.py:43 | change "x = torch.cuda.FloatTensor(256, 1024, block_mem)" to "x = torch.mlu.FloatTensor(256, 1024, block_mem) " |
| 88 | yolox/utils/metric.py:48 | change "def gpu_mem_usage():" to "def mlu_mem_usage(): " |
| 89 | yolox/utils/metric.py:50 | change "Compute the GPU memory usage for the current device (MB)." to "Compute the MLU memory usage for the current device (MB). " |
| 90 | yolox/utils/metric.py:52 | change "mem_usage_bytes = torch.cuda.max_memory_allocated()" to "mem_usage_bytes = torch.mlu.max_memory_allocated() " |
| 91 | yolox/utils/dist.py:8 | change "This file contains primitives for multi-gpu communication." to "This file contains primitives for multi-mlu communication. " |
| 92 | yolox/utils/dist.py:21 | add "import torch_mlu" |
| 93 | yolox/utils/dist.py:42 | change "gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)" to "mlu_list = os.getenv('MLU_VISIBLE_DEVICES', None) " |
| 94 | yolox/utils/dist.py:43 | change "if gpu_list is not None:" to "if mlu_list is not None: " |
| 95 | yolox/utils/dist.py:44 | change "return len(gpu_list.split(','))" to "return len(mlu_list.split(',')) " |
| 96 | yolox/utils/dist.py:142 | change "if dist.get_backend() == "nccl":" to "if dist.get_backend() == "cncl": " |
| 97 | yolox/utils/dist.py:150 | change "assert backend in ["gloo", "nccl"]" to "assert backend in ["gloo", "cncl"] " |
| 98 | yolox/utils/dist.py:151 | change "device = torch.device("cpu" if backend == "gloo" else "cuda")" to "device = torch.device("cpu" if backend == "gloo" else "mlu") " |
| 99 | yolox/utils/dist.py:292 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 100 | yolox/utils/dist.py:293 | change "torch.cuda.synchronize()" to "torch.mlu.synchronize() " |
| 101 | yolox/utils/allreduce_norm.py:8 | add "import torch_mlu" |
| 102 | yolox/utils/allreduce_norm.py:41 | change "def pyobj2tensor(pyobj, device="cuda"):" to "def pyobj2tensor(pyobj, device="mlu"): " |
| 103 | yolox/layers/jit_ops.py:58 | change """"Get optional list of compiler flags to forward to nvcc when building CUDA sources"""" to """"Get optional list of compiler flags to forward to nvcc when building MLU sources""" " |
| 104 | yolox/layers/jit_ops.py:62 | change ""-U__CUDA_NO_HALF_OPERATORS__"," to ""-U__MLU_NO_HALF_OPERATORS__", " |
| 105 | yolox/layers/jit_ops.py:63 | change ""-U__CUDA_NO_HALF_CONVERSIONS__"," to ""-U__MLU_NO_HALF_CONVERSIONS__", " |
| 106 | yolox/layers/jit_ops.py:64 | change ""-U__CUDA_NO_HALF2_OPERATORS__"," to ""-U__MLU_NO_HALF2_OPERATORS__", " |
| 107 | yolox/layers/jit_ops.py:106 | change "extra_cuda_cflags=self.nvcc_args()," to "extra_mlu_cflags=self.nvcc_args(), " |
| 108 | tools/export_onnx.py:9 | add "import torch_mlu" |
| 109 | tools/eval.py:11 | add "import torch_mlu" |
| 110 | tools/eval.py:19 | change "configure_nccl," to "configure_cncl, " |
| 111 | tools/eval.py:34 | change ""--dist-backend", default="nccl", type=str, help="distributed backend"" to ""--dist-backend", default="cncl", type=str, help="distributed backend" " |
| 112 | tools/eval.py:116 | change "def main(exp, args, num_gpu):" to "def main(exp, args, num_mlu): " |
| 113 | tools/eval.py:125 | change "is_distributed = num_gpu > 1" to "is_distributed = num_mlu > 1 " |
| 114 | tools/eval.py:128 | change "configure_nccl()" to "configure_cncl() " |
| 115 | tools/eval.py:156 | change "torch.cuda.set_device(rank)" to "torch.mlu.set_device(rank) " |
| 116 | tools/eval.py:157 | change "model.cuda(rank)" to "model.mlu(rank) " |
| 117 | tools/eval.py:166 | change "loc = "cuda:{}".format(rank)" to "loc = "mlu:{}".format(rank) " |
| 118 | tools/eval.py:208 | change "num_gpu = torch.cuda.device_count() if args.devices is None else args.devices" to "num_mlu = torch.mlu.device_count() if args.devices is None else args.devices " |
| 119 | tools/eval.py:209 | change "assert num_gpu <= torch.cuda.device_count()" to "assert num_mlu <= torch.mlu.device_count() " |
| 120 | tools/eval.py:214 | change "num_gpu," to "num_mlu, " |
| 121 | tools/eval.py:219 | change "args=(exp, args, num_gpu)," to "args=(exp, args, num_mlu), " |
| 122 | tools/trt.py:11 | add "import torch_mlu" |
| 123 | tools/trt.py:59 | change "model.cuda()" to "model.mlu() " |
| 124 | tools/trt.py:61 | change "x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()" to "x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).mlu() " |
| 125 | tools/demo.py:12 | add "import torch_mlu" |
| 126 | tools/demo.py:53 | change "help="device to run our model, can either be cpu or gpu"," to "help="device to run our model, can either be cpu or mlu", " |
| 127 | tools/demo.py:128 | change "x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()" to "x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).mlu() " |
| 128 | tools/demo.py:151 | change "if self.device == "gpu":" to "if self.device == "mlu": " |
| 129 | tools/demo.py:152 | change "img = img.cuda()" to "img = img.mlu() " |
| 130 | tools/demo.py:257 | change "args.device = "gpu"" to "args.device = "mlu" " |
| 131 | tools/demo.py:271 | change "if args.device == "gpu":" to "if args.device == "mlu": " |
| 132 | tools/demo.py:272 | change "model.cuda()" to "model.mlu() " |
| 133 | tools/train.py:10 | add "import torch_mlu" |
| 134 | tools/train.py:15 | change "from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices" to "from yolox.utils import configure_module, configure_cncl, configure_omp, get_num_devices " |
| 135 | tools/train.py:25 | change ""--dist-backend", default="nccl", type=str, help="distributed backend"" to ""--dist-backend", default="cncl", type=str, help="distributed backend" " |
| 136 | tools/train.py:81 | change "help="occupy GPU memory first for training."," to "help="occupy MLU memory first for training.", " |
| 137 | tools/train.py:113 | change "configure_nccl()" to "configure_cncl() " |
| 138 | tools/train.py:131 | change "num_gpu = get_num_devices() if args.devices is None else args.devices" to "num_mlu = get_num_devices() if args.devices is None else args.devices " |
| 139 | tools/train.py:132 | change "assert num_gpu <= get_num_devices()" to "assert num_mlu <= get_num_devices() " |
| 140 | tools/train.py:140 | change "num_gpu," to "num_mlu, " |
| 141 | tools/visualize_assign.py:11 | add "import torch_mlu" |
| 142 | tools/visualize_assign.py:38 | change "with torch.cuda.amp.autocast(enabled=self.amp_training):" to "with torch.mlu.amp.autocast(enabled=self.amp_training): " |
| 143 | tools/export_torchscript.py:9 | add "import torch_mlu" |
| 144 | tests/utils/test_model_utils.py:7 | add "import torch_mlu" |
| 145 | demo/nebullvm/nebullvm_optimization.py:1 | add "import torch_mlu" |
| 146 | demo/nebullvm/nebullvm_optimization.py:9 | change "model.cuda()" to "model.mlu() " |
| 147 | demo/nebullvm/nebullvm_optimization.py:12 | change "device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 148 | demo/MegEngine/python/convert_weights.py:7 | add "import torch_mlu" |
| 149 | demo/OpenVINO/python/openvino_inference.py:60 | change "help='Optional. Specify the target device to infer on; CPU, GPU, \" to "help='Optional. Specify the target device to infer on; CPU, MLU, \ " |
