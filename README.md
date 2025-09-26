
<div align="center">
<h2>Direct Noise Alignment for Text-Guided Rectified Flow Editing</h2>



Chenxi Xie<sup>1,2*</sup>
| Minghan Li<sup>3*</sup> | 
Shuai Li<sup>1</sup> | 
Yuhui Wu<sup>1,2</sup> | 
Qiaosi Yi<sup>1,2</sup> | 
Lei zhang<sup>1,2</sup> 

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute <sup>3</sup>Harvard AI and Robotics Lab, Harvard University

🚩 Accepted by NeurIPS 2025 (Spotlight🌟)

<a href='https://arxiv.org/pdf/2506.01430'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

</div>




## 1️⃣Enviroment

torch=2.3.1 diffusers==0.34.0
## 2️⃣Inference on PIE-Bench
1. Modify the path of PIE-Bench and your Diffusion models checkpoints in scripts/run_script_dnaedit.py
2. run: 
```
python scripts/run_script_dnaedit.py --device_number 0 --exp_yaml configs/DNAEdit_SD3_exp.yaml --save ./output
```



## 🌏 Citation

```bash
@article{xie2025dnaedit,
  title={DNAEdit: Direct Noise Alignment for Text-Guided Rectified Flow Editing},
  author={Xie, Chenxi and Li, Minghan and Li, Shuai and Wu, Yuhui and Yi, Qiaosi and Zhang, Lei},
  journal={arXiv preprint arXiv:2506.01430},
  year={2025}
}
```


### License
This project is released under the [Apache 2.0 license](LICENSE).

### Acknowledgement
The code is largely based on [FlowEdit](https://github.com/fallenshock/FlowEdit) whom we thank for their excellent work.

### Contact
If you have any questions, please contact: chenxi.xie@connect.polyu.hk


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=xiechenxi99.MaSS13K)

</details>
