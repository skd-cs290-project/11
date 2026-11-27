# CS290U 2.2 晴天到雨天街景迁移 · 单卡小测

在 BDD100K 上训练 SD 1.5 雨天 LoRA，比较 **LoRA + Canny ControlNet** 和 **同一份 LoRA + SMS**。固定输入、底座、天气提示和 LoRA 权重，判断几何布局保留与雨天外观是否能同时改善。这里比较两套完整的生成流程，基线推理用 FP16，SMS 的逐图优化按原代码使用 FP32。小测检查湿路反光、阴雨光照、车辆数量和位置、道路走向。

## 小测与正式实验

本目录提供同一雨天 LoRA 的两条独立流程。基线以晴图做 Img2Img，并加载 Canny ControlNet。SMS 同样从晴图开始，直接优化其潜变量。[正式实验方案](docs/Technical_Plan.zh-CN.md)另设从基线输出初始化的 SMS 对照、频谱消融及完整盲测。

## 在集群登录节点准备

查看集群使用手册，用 `sinfo -o '%P %G %l %a'` 查询 GRES 类型和时限，并按账户配额选择支持 **单张 A40** 的分区。

```bash
bash setup_env.sh
work="$PWD/runs/pilot01"
.venv/bin/python prepare.py --hub --work "$work"
.venv/bin/python pilot.py models --work "$work"
```

环境默认 Python 3.10，已有兼容环境可以用 `CS290_PYTHON` 指定其 Python 的绝对路径。创建环境时可用 `CS290_CREATE_PYTHON` 指定集群上实际的 Python 3.10/3.11。默认 PyTorch 2.5.1 CUDA 12.1 wheel 使用节点上的 NVIDIA 驱动。登录节点检查 wheel、库、文件、标签、尺寸。拿到卡后先实际运行 FP16 CUDA 矩阵乘法，再训练。

`--hub` 从 Hugging Face BDD100K 镜像按官方 train/val 分区下载选中的图片，记录镜像提交和样本 ID。默认取 160 张雨天训练图、20 张晴天测试图，统一为 768×432。集群已有 BDD100K 时可传本地路径：

```bash
.venv/bin/python prepare.py --work "$work" \
  --train-images /path/to/images/100k/train --val-images /path/to/images/100k/val \
  --train-labels /path/to/train-labels.json --val-labels /path/to/val-labels.json
```

本地标签接受 BDD100K 帧数组或 Scalabel 的 `frames` 对象，读取 `scene=city street`、`timeofday=daytime`、`weather=rainy/clear`。已有模型缓存可把 SD 1.5 和 Canny 的 Diffusers 目录链接到 `$work/models/sd15` 和 `$work/models/canny` 再做预检。GPU 作业完全读取本地数据和权重，依赖安装和下载在排卡前完成。

## 一次提交

```bash
# PARTITION、ACCOUNT 用当前手册和集群查到的值。默认账户可省略 ACCOUNT。
bash submit.sh --check "$work" "$PARTITION" "$ACCOUNT"
bash submit.sh --submit "$work" "$PARTITION" "$ACCOUNT"
```

`--check` 做文件检查和 Slurm `--test-only`，只预览调度，提交需要第二条命令。正式作业请求 **1×A40、6 小时**。GRES 默认是 `NVIDIAA40`，可通过 `CS290_A40_TYPE` 填写 `sinfo` 显示的确切 A40 类型。提交记录写入 `submission/job_id`，同一运行目录有记录时会拒绝再次提交。日志在 `logs/<job-id>.log`。

一次作业依次完成 GPU 检查、600 步雨天 LoRA、两组生成和结果检查。如提供已验证的同底座雨天 LoRA，在运行前放到 `rain_lora/pytorch_lora_weights.safetensors` 即可复用。SMS 每张图默认运行 500 次优化，重置逐图拟合的适配器和优化器，防止不同测试图之间互相影响。形成一组三联图就写盘。

| 输出 | 路径 |
|---|---|
| GPU 检查 | `cuda.json` |
| 雨天 LoRA 与检查点 | `rain_lora/` |
| 独立的两组输出 | `results/baseline/` 和 `results/sms/` |
| 源图、基线、SMS 三联图 | `results/comparison/` |
| 数据选择和随机种子 | `data/manifest.json` |
| 下载的模型版本 | `models/revisions.json` |

运行结束时验证每个源图都对应两张结果和一张三联图。正式报告将天气真实感与几何保留分开打分，同时报告失败图和实际耗时。小测通过后再扩大规模。

## 来源

- [BDD100K 官方下载及标签文档](https://github.com/bdd100k/bdd100k) 与 [使用的数据镜像](https://huggingface.co/datasets/dronefreak/BDD100K)。代码包只记录数据和模型来源。
- [SMS ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Balanced_Image_Stylization_with_Style_Matching_Score_ICCV_2025_paper.html) 和 [官方仓库](https://github.com/showlab/SMS)。原有优化公式保留。改动包括固定本地同一底座、接入 Diffusers LoRA、矩形潜空间尺寸和逐图复位。
- [Diffusers 0.30.0 SD 训练脚本](https://github.com/huggingface/diffusers/blob/v0.30.0/examples/text_to_image/train_text_to_image_lora.py)。上游许可和文件版本记录在 `vendor/`。
