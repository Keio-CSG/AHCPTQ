# AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model

## 1. Environment Settings

### 1.1 Create Environment

We follow the environment settings of [PTQ4SAM](https://github.com/chengtao-lv/PTQ4SAM), please refer to the ``environment.sh`` in the root directory.
1. Install PyTorch
```
conda create -n ptq4sam python=3.7 -y
pip install torch torchvision
```

2. Install MMCV

```
pip install -U openmim
mim install "mmcv-full<2.0.0"
```

3. Install other requirements

```
pip install -r requirements.txt
```

4. Compile CUDA operators

```
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..
```

5. Install mmdet
```
cd mmdetection/
python3 setup.py build develop
cd ..
```

### 1.2 Prepare Dataset
Download the [COCO](https://drive.google.com/file/d/1j92XnlzQZwPff2sP_nwU3LE9Npemkn7Q/view?usp=sharing) dataset, recollect them as the following form, and revise the corresponding root directory in the code:

```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```
### 1.3 Download Model Weights

Download the model weights of SAM and detector, save them at the ``ckpt/`` folder:

| Model       | Download                                                                                    |
|-------------|---------------------------------------------------------------------------------------------|
| SAM-B       | [Link](https://drive.google.com/file/d/1UlwYWVRsS4SbSPDXlR5_dVmcuqT8CzeI/view?usp=sharing)  |
| SAM-L       | [Link](https://drive.google.com/file/d/14MBHh7OFwY8EpaGkX6ZyjUAw83wywk7U/view?usp=sharing)  |
| SAM-H       | [Link](https://drive.google.com/file/d/1fMJyX938_H17OxfVq6PQZ_ef9TBy5r36/view?usp=sharing)  |
| Faster-RCNN | [Link](https://drive.google.com/file/d/1RKTLk07E4apoRzwoeQbnaY8ZxEX1SlbG/view?usp=sharing)  |
| YOLOX       | [Link](https://drive.google.com/file/d/1FQeKOaDJzwqXq4zz8-VHJbn6iKFT4HLt/view?usp=sharing)  |
| HDETR       | [Link](https://drive.google.com/file/d/1i7iMAicmoif8tUbuHEntVtmEsJrpXTZ4/view?usp=sharing)  |
| DINO        | [Link](https://drive.google.com/file/d/1DDHkZcVI9TwmN9vqEYXFBjRZVsBK4yLO/view?usp=sharing)  |

## 2. Run Experiments

Please use the following command to perform AHCPTQ quantization:

```
python ahcptq/solver/test_quant.py \
--config ./projects/configs/<DETECTOR>/<MODEL.py> \
--q_config ./exp/<QCONFIG>.yaml \
--quant-encoder
```

Here, ``<DETECTOR>`` is the folder name of prompt detector, ``<MODEL.py>`` is configuration file of corresponding SAM model, and ``<QCONFIG>.yaml`` is the specific quantization configuration file.

For example, to perform W4A4 quantization for SAM-B with a YOLO detector, use the following command:

```
python ahcptq/solver/test_quant.py \
--config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
--q_config ./exp/config44.yaml \
--quant-encoder
```

We use A6000 GPU with 48G memory to run these experiments. However, we find that the memory is not sufficient to complete experiments on HDETR and DINO since the number of prompt box is large. Therefore, we offload the memory to CPU DRAM and process quantization one by one.

If you have the same problem, please set ``keep_gpu: False`` in the ``<QCONFIG>.yaml`` file, and comment out line 218 to 239 in ``./ahcptq/solver/recon.py``, and unindent line 240 to 279. We hope this could help address this issue.

## 3. Abstract

<a href="" target="_blank">Paper Link</a>

The Segment Anything Model (SAM) has demonstrated strong versatility across various visual tasks. To enable efficient deployment on edge devices, post-training quantization (PTQ) has been introduced to reduce computational complexity. However, existing PTQ methods still suffer from degraded performance due to two primary challenges: highly skewed distribution of post-GELU activations and significant inter-channel variation in linear projection activations. To tackle these issues, we propose AHCPTQ, a novel PTQ framework that improves quantization performance through hardware co-optimization. AHCPTQ employs Hybrid Log-Uniform Quantization (HLUQ), which applies log2 quantization for dense small values and uniform quantization for sparse large values. Additionally, Channel-Aware Grouping (CAG) progressively clusters activation channels with similar distributions, allowing them to share quantization parameters. The combination of HLUQ and CAG not only enhances quantization performance but also ensures compatibility with efficient hardware execution. For instance, under the W4A4 configuration on the SAM-L model, AHCPTQ achieves 36.6\% mAP on instance segmentation with the DINO detector, while achieving a $\times7.89$ speedup and $\times8.64$ energy efficiency over its floating-point counterpart in FPGA implementation.

## 4. Bug in Original PTQ4SAM Framework

We observe that in PTQ4SAM, the dropout probability does not revert to 1.0 during the evaluation phase. As a result, half of the activation values remain in floating-point (not quantized), leading to a significantly overestimated mAP in PTQ4SAM and QDrop, as reported in their [paper](https://arxiv.org/abs/2405.03144). To address this issue, we introduce the following code into line 360 in ``./ahcptq/solver/test_quant.py`` to ensure dropout is properly disabled during evaluation.

```
for n, m in model.named_modules():
    if hasattr(m, 'drop_prob'):
        m.drop_prob = 1
```

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{AHCPTQ,
  title={AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model},
  author={Zhang, Wenlun, and Z., Y. and Ando, Shimpei and Yoshioka, Kentaro},
  journal={arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Acknowledgments
Our work is built upon [PTQ4SAM](https://github.com/chengtao-lv/PTQ4SAM). We thank for their pioneering work and create a nice baseline for quanization of SAM.
