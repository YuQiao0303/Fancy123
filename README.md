# Fancy123
###  [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Fancy123_One_Image_to_High-Quality_3D_Mesh_Generation_via_Plug-and-Play_CVPR_2025_paper.html) | [![arXiv](https://img.shields.io/badge/arXiv-2411.16185-b31b1b.svg)](https://arxiv.org/abs/2411.16185)

This is the official repository for the CVPR2025 paper: *''Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play Deformation''* by Qiao Yu, Xianzhi Li, Yuan Tang, Xu Han, Long Hu, Yixue Hao, and Min Chen.


Fancy123 generates a high-quality textured 3D mesh given a single RGB image:

https://github.com/user-attachments/assets/470f1244-197a-472a-98d6-599775638ac7

Fancy123's core idea is to utilize 2D image deformation to address the multiview inconsistency issue, and 3D mesh deformation to address the low fidelity issue:

![method](https://github.com/user-attachments/assets/fdebcb8e-b98c-450c-8309-8139e547931c)

https://github.com/user-attachments/assets/6700ac0d-7331-4097-8cf3-3a3e7ede6b70






## 🎉 News
- 2025.07.21: Code Released! 🌟
- 2025.02.27: Fancy123 is accepted by **CVPR2025**!🎊 We currently plan to release code in July 2025.


## 🛠️Install & Model Download
Please refer to [scripts/install.md](scripts/install.md).


## ✅Training
Fancy123 works in inference and is training-free!

## 🎬Inference
First, put your images under `examples', then run InstantMesh to get an initial mesh:

```bash
python run_init.py
```
You'll see generated multiview images in outputs/instant-mesh-large/images and initial meshes in outputs/instant-mesh-large/meshes.

Then, run Fancy123's enhancement steps:
```bash
python main_fancy123_refine.py
```
You'll see results in outputs/instant-mesh-large/fancy123_meshes. 
You can use tools like MeshLab or Blender to visualize the final result named `final_mesh.obj'.




## 🤝Acknowledgement
We have intensively borrowed code from the following repositories. Many thanks to the authors for sharing their code.

- [InstantMesh](https://github.com/TencentARC/InstantMesh)
- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [Unique3D](https://github.com/AiuniAI/Unique3D)
- [APAP](https://github.com/KAIST-Visual-AI-Group/APAP)


## 📝Citation
If you find Fancy123 helpful, please cite our paper:
```
@InProceedings{Yu_2025_CVPR,
    author    = {Yu, Qiao and Li, Xianzhi and Tang, Yuan and Han, Xu and Hu, Long and Hao, Yixue and Chen, Min},
    title     = {Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play Deformation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {595-604}
}
```
