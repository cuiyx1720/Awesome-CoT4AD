# Awesome CoT for Autonomous Driving

![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2505.20223-<COLOR>.svg)](https://arxiv.org/abs/2505.20223)

This is the repository of **Chain-of-Thought for Autonomous Driving: A Comprehensive Survey and Future Prospects**. We warmly welcome contributions from everyone to this repository. Please feel free to submit issues or pull requests for any missing papers, datasets, or methodologies. Our team is committed to maintaining and updating the repository regularly.

# Contents

- [Introduction](#Introduction)
- [Methods](#methods)
  - [Modular Driving CoT](#Modular Driving CoT)
  - [Logical Driving CoT](#Logical Driving CoT)
  - [Reflective Driving CoT](#Reflective Driving CoT)
- [Datasets](#datasets)
- [Citation](#citation)

# Introduction

The rapid evolution of large language models in natural language processing has substantially elevated their semantic understanding and logical reasoning capabilities. Such proficiencies have been leveraged in autonomous driving systems, contributing to significant improvements in system performance. Models such as OpenAI o1 and DeepSeek-R1, leverage Chain-of-Thought (CoT) reasoning, an advanced cognitive method that simulates human thinking processes, demonstrating remarkable reasoning capabilities in complex tasks. By structuring complex driving scenarios within a systematic reasoning framework, this approach has emerged as a prominent research focus in autonomous driving, substantially improving the system's ability to handle challenging cases. This paper investigates how CoT methods improve the reasoning abilities of autonomous driving models. Based on a comprehensive literature review, we present a systematic analysis of the motivations, methodologies, challenges, and future research directions of CoT in autonomous driving. Furthermore, we propose the insight of combining CoT with self-learning to facilitate self-evolution in driving systems. To ensure the relevance and timeliness of this study, we have compiled a dynamic repository of literature and open-source projects, diligently updated to incorporate forefront developments.

![photo1](https://github.com/cuiyx1720/Awesome-CoT4AD/blob/master/CoT4AD.assets/photo1.png)

# Methods

### Modular Driving CoT

> **Modular Driving CoT** decomposes driving tasks into independent submodules like perception, prediction, and planning, each performing specific reasoning tasks, which helps in optimizing and adapting the system to diverse scenarios.



<img src="..\github\CoT4AD.assets\photo2.png" alt="photo2" style="zoom:25%;" />

| Name                                             | Venue        | Task       | Input                | Open Source                                                  | CoT Process                                                  |
| :----------------------------------------------- | :----------- | :--------- | :------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [DriveAgent](https://arxiv.org/pdf/2505.02123)   | arXiv 2025   | Perception | MF-SV                | [code](https://github.com/Paparare/DriveAgent?utm_source=catalyzex.com) | Description → Vehicle reasoning → Scene analysis → Perception |
| [Reason2Drive](https://arxiv.org/pdf/2312.03661) | ECCV 2024    | Prediction | MF-SV, Question      | [code](https://github.com/fudan-zvg/Reason2Drive?utm_source=catalyzex.com) | Perception answer → Prediction answer → Motion visualization |
| [Motion-LLaVA](https://arxiv.org/pdf/2407.04281) | arXiv 2024   | Prediction | 2DKS                 | [code](https://github.com/yhli123/WOMD-Reasoning)            | Aggregated in-context reasoning → Prediction answer          |
| [GPT-Driver](https://arxiv.org/pdf/2310.01415)   | NeurIPS 2023 | Planning   | 2DKS, EH, TI         | [code](https://github.com/PointsCoder/GPT-Driver?utm_source=catalyzex.com) | Key object detection → Interaction prediction → Trajectory   |
| [RDA-Driver](https://arxiv.org/pdf/2408.13890)   | ECCV 2024    | Planning   | SF-MV, EH            | ❌                                                            | Key object detection and prediction → High-level intent → Trajectory |
| [AlphaDrive](https://arxiv.org/pdf/2503.07608)   | arXiv 2025   | Planning   | MF-SV                | [code](https://github.com/hustvl/AlphaDrive?utm_source=catalyzex.com) | Key object detection → High-level intent                     |
| [LLM-Driver](https://arxiv.org/pdf/2310.01957)   | IEEE 2024    | E2E        | 2DKS                 | [code](https://github.com/wayveai/Driving-with-LLMs)         | Scene vectors grounding → Prediction answer, Control         |
| [LMDrive](https://arxiv.org/pdf/2312.07488)      | CVPR 2024    | E2E        | MF-MV, TI, LiDAR     | [code](https://github.com/opendilab/LMDrive?utm_source=catalyzex.com) | Feature detection → Trajectory → PID control                 |
| [WiseAD](https://arxiv.org/pdf/2412.09951)       | arXiv 2024   | E2E        | MF-SV, TI, Question  | ❌                                                            | Answer (Scene description, Risk analysis...) + Trajectory    |
| [EMMA](https://arxiv.org/pdf/2410.23262)         | arXiv 2024   | E2E        | SF-MV, EH, TI        | ❌                                                            | Scene description → Key object description → Meta action     |
| [OpenEMMA](https://arxiv.org/pdf/2412.15208)     | WACV 2025    | E2E        | SF-SV, EH            | [code](https://github.com/taco-group/OpenEMMA?utm_source=catalyzex.com) | Key object, Scene description, High-level intent → Trajectory |
| [LightEMMA](https://arxiv.org/pdf/2505.00284v1)  | arXiv 2025   | E2E        | SF-SV, EH            | [code](https://github.com/michigan-traffic-lab/LightEMMA?utm_source=catalyzex.com) | Scene description → High-level intent → Trajectory           |
| [DriveLM](https://arxiv.org/pdf/2312.14150)      | ECCV 2024    | E2E        | SF-SV, Question      | [code](https://github.com/OpenDriveLab/DriveLM?utm_source=catalyzex.com) | Perception → Prediction → High-level intent → Trajectory     |
| [See2DriveX](https://arxiv.org/pdf/2502.14917)   | arXiv 2025   | E2E        | MF-MV, 2DKS, TI, BEV | ❌                                                            | Scene description → Meta action → Trajectory → Control       |



### Logical Driving CoT

> **Logical Driving CoT** breaks down tasks into logically interconnected sub-problems, with a focus on rigorous logical constraints during the reasoning process, generating driving decisions through structured logical steps.



![photo3](..\github\CoT4AD.assets\photo3.png)



| Name                                               | Venue      | Task       | Input                    | Open Source                                                  | CoT Process                                                  |
| -------------------------------------------------- | ---------- | ---------- | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Dolphins](https://arxiv.org/pdf/2312.00438)       | ECCV 2024  | Perception | MF-SV, Question          | [code](https://vlm-driver.github.io/?utm_source=catalyzex.com) | GCoT-fine-tuning → Perception answer                         |
| [RIV-CoT](https://arxiv.org/pdf/2412.08742)        | arXiv 2025 | Perception | SF-SV, Question          | ❌                                                            | Bounding box → Image crop → Perception answer                |
| [AgentThink](https://arxiv.org/pdf/2505.15298)     | arXiv 2025 | Perception | SF-MV, Question          | ❌                                                            | (Tool use → Reasoning) → Perception answer                   |
| [LC-LLM](https://arxiv.org/pdf/2403.18344)         | arXiv 2025 | Prediction | 2DKS, EH                 | ❌                                                            | Feature detection → Intention → Motion prediction, Explanation |
| [CoT-Drive](https://arxiv.org/pdf/2503.07234)      | arXiv 2025 | Prediction | 2DKS                     | ❌                                                            | Background → Interaction analysis → Risk assessment → Motion prediction |
| [SenseRAG](https://arxiv.org/pdf/2501.03535)       | WACV 2025  | Prediction | SF-MV, LiDAR, Structured | ❌                                                            | Data injection → RAG → Motion Prediction                     |
| [DriveVLM](https://arxiv.org/pdf/2402.12289)       | CoRL 2024  | Planning   | MF-MV                    | [code](https://tsinghua-mars-lab.github.io/DriveVLM/?utm_source=catalyzex.com) | Scene description → Scene analysis → Hierarchical planning → Trajectory |
| [CALMM-Drive](https://arxiv.org/pdf/2412.04209)    | arXiv 2025 | Planning   | BEV, EH, TI              | ❌                                                            | Top-K action, Confidence → Trajectory → Hierarchical refinement |
| [LanguageMPC](https://arxiv.org/pdf/2310.03026)    | arXiv 2023 | Decision   | 2DKS, EH, TI             | ❌                                                            | Vehicle detection → Situational awareness → Meta action      |
| [DriveMLM](https://arxiv.org/pdf/2312.09245)       | arXiv 2023 | Decision   | MF-MV, TI, LiDAR         | [code](https://github.com/OpenGVLab/DriveMLM?utm_source=catalyzex.com) | Linguistic description → Speed-path decision                 |
| [CoT-VLM4Tar](https://arxiv.org/pdf/2503.01632)    | arXiv 2025 | Decision   | SF-SV                    | ❌                                                            | Situation classification → Scene analysis → High-level intent → Meta action |
| [DriveCoT](https://arxiv.org/pdf/2403.16996)       | arXiv 2024 | E2E        | MF-MV                    | [code](https://drivecot.github.io/?utm_source=catalyzex.com) | Multidimensional prediction → Logical decision → Meta action |
| [Senna](https://arxiv.org/pdf/2410.22313)          | arXiv 2024 | E2E        | SF-MV, TI                | [code](https://github.com/hustvl/Senna?utm_source=catalyzex.com) | Meta-action → Perception → Motion prediction → Trajectory    |
| [PRIMEDrive-CoT](https://arxiv.org/pdf/2504.05908) | CVPRW 2025 | E2E        | SF-MV, LiDAR             | ❌                                                            | Uncertainty/Risk → Interaction → Logical decision → Meta action |
| [LangCoop](https://arxiv.org/pdf/2504.13406)       | arXiv 2025 | E2E        | SF-SV                    | [code](https://xiangbogaobarry.github.io/LangCoop/?utm_source=catalyzex.com) | Scene description → High-level intent → LangPack integration → Control |
| [X-Driver](https://arxiv.org/pdf/2505.05098)       | arXiv 2025 | E2E        | SF-SV, TI                | ❌                                                            | Object detection, Traffic sign, Lane info → Trajectory       |



### Reflective Driving CoT

> **Reflective Driving CoT** enhances the previous two methods by incorporating a reflective feedback process, allowing the system to evaluate discrepancies between expected and actual outcomes, enabling self-correction and continuous learning.



![photo4](..\github\CoT4AD.assets\photo4.png)

| Name                                                     | Venue        | Task       | Input              | Open Source                                                  | CoT Process                                                  |
| -------------------------------------------------------- | ------------ | ---------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Agent-Driver](https://arxiv.org/pdf/2311.10813)         | COLM 2024    | Perception | SF-MV              | [code](https://usc-gvl.github.io/Agent-Driver/?utm_source=catalyzex.com) | Key object detection → High-level intent → Trajectory ⟲ Refine, Mem |
| [PlanAgent](https://arxiv.org/pdf/2406.01587)            | arXiv 2024   | Planning   | MF-MV, 2DKS        | ❌                                                            | Global, local info → Scene description → Planning code ⟲ Refine |
| [DiLu](https://arxiv.org/pdf/2309.16292)                 | ICLR 2024    | Decision   | 2DKS               | [code](https://pjlab-adg.github.io/DiLu/)                    | Scene description → Meta action ⟲ Refine, Mem                |
| [Receive-Reason-React](https://arxiv.org/pdf/2310.08034) | IEEE 2024    | Decision   | 2DKS, TI, In-cabin | ❌                                                            | Scene description → Explanation → Meta action ⟲ Mem          |
| [SafeDrive](https://arxiv.org/pdf/2412.13238)            | arXiv 2024   | Decision   | 2DKS, TI           | [code](https://mezzi33.github.io/SafeDrive/?utm_source=catalyzex.com) | Risk evaluation → Key object detection → Meta action ⟲ Refine, Mem |
| [KOMA](https://arxiv.org/pdf/2407.14239)                 | IEEE 2024    | Decision   | 2DKS               | [code](https://jkmhhh.github.io/KoMA/?utm_source=catalyzex.com) | Scene description → Goal → Planning → Meta action ⟲ Refine, Mem |
| [LeapAD](https://arxiv.org/pdf/2405.15324)               | NeurIPS 2024 | Decision   | SF-MV              | [code](https://github.com/PJLab-ADG/LeapAD?utm_source=catalyzex.com) | Scene description → Dual-process ⟲ Refine, Mem               |
| [LeapVAD](https://arxiv.org/pdf/2501.08168)              | arXiv 2025   | Decision   | MF-MV              | [code](https://pjlab-adg.github.io/LeapVAD/?utm_source=catalyzex.com) | Scene description, ACC-ACT similarity → Dual-process ⟲ Refine, Mem |
| [CoDrivingLLM](https://arxiv.org/pdf/2409.12812)         | IEEE 2025    | Decision   | 2DKS               | [code](https://github.com/FanGShiYuu/CoDrivingLLM)           | State perception → Intent sharing → Negotiation → Meta action ⟲ Mem |
| [Actor-Reasoner](https://arxiv.org/pdf/2503.00502)       | arXiv 2025   | Decision   | 2DKS, TI, Mem DB   | [code](https://github.com/FanGShiYuu/Actor-Reasoner)         | Intent prediction → Driving style → Meta action ⟲ Mem        |
| [PKRD-CoT](https://arxiv.org/pdf/2412.02025)             | arXiv 2024   | E2E        | SF-MV              | ❌                                                            | Scene description → Object detection → High-level intent ⟲ Mem |
| [ORION](https://arxiv.org/pdf/2503.19755)                | arXiv 2025   | E2E        | SF-MV, TI          | [code](https://xiaomi-mlab.github.io/Orion/?utm_source=catalyzex.com) | Feature extraction → Scene analysis → Meta action → Trajectory ⟲ Mem |



# Datasets

With technological advancements, datasets now encompass not only perceptual and behavioral data but also progressively integrate semantic cognitive information, driving the development of CoT based models.

| Name                                                         | Year | Source                | Cognitive Data Type      | Size                                 | Tasks                  |
| ------------------------------------------------------------ | ---- | --------------------- | ------------------------ | ------------------------------------ | ---------------------- |
| [Talk2Car](https://macchina-ai.cs.kuleuven.be/)              | 2019 | nuScenes              | VQA                      | 850 videos, 11,959 commands          | Perception             |
| [nuScenes-QA](https://github.com/qiantianwen/NuScenes-QA)    | 2024 | nuScenes              | VQA                      | 28K frames, 459,941 QAs              | Perception             |
| [Talk2BEV](https://llmbev.github.io/talk2bev/)               | 2024 | nuScenes              | VQA                      | 1K BEV scenarios, 20K QAs            | Perception             |
| [NuScenes-MQA](https://github.com/turingmotors/NuScenes-MQA) | 2024 | nuScenes              | Markup-QA                | 34,149 scenarios, 1,459,933 QA pairs | Perception             |
| [Reason2Drive](https://github.com/fudan-zvg/reason2drive)    | 2024 | nuScenes, Waymo, ONCE | QA                       | 420K frames, 420K QAs                | Perception, Prediction |
| [DriveMLLM](https://github.com/XiandaGuo/Drive-MLLM)         | 2024 | nuScenes              | QA                       | 880 frames, 4,666 QAs                | Perception             |
| [nuPrompt](https://github.com/wudongming97/Prompt4Driving)   | 2025 | nuScenes              | Language prompts         | 850 videos, 35,367 prompts           | Perception             |
| [DriveLMM-o1](https://github.com/ayesha-ishaq/DriveLMM-o1)   | 2025 | nuScenes              | Reasoning process        | 1,962 frames, 18K QAs                | Perception, Planning   |
| [BDD-OIA](https://arxiv.org/pdf/2003.09405)                  | 2020 | BDD100K               | Explanation              | 22,924 clips, 35,366 explanations    | Perception             |
| [BDD-X](https://github.com/JinkyuKimUCB/explainable-deep-driving) | 2018 | BDD100K               | Description, Explanation | 6,984 clips, 50,298 description      | Planning               |
| [DriveGPT4](https://tonyxuqaq.github.io/projects/DriveGPT4/) | 2024 | BDD-X                 | QA                       | 16K QAs, 40K conversations           | Planning               |
| [Refer-KITTI](https://github.com/wudongming97/RMOT)          | 2023 | KITTI                 | Language prompts         | 18 videos, 818 expressions           | Perception             |
| [WOMD-Reasoning](https://waymo.com/open/download/.)          | 2024 | WOMD                  | QA, Prediction           | 63K scenes, 3M QAs                   | Prediction             |
| [CityFlow-NL](https://arxiv.org/pdf/2101.04741)              | 2021 | CityFlow              | Language prompts         | 3,028 tracks, 5,289 prompts          | Perception             |
| [DRIVINGVQA](https://vita-epfl.github.io/DrivingVQA/)        | 2025 | Code de la Route      | VQA                      | 3,142 frames, 3,142 QAs              | Perception             |
| [Highway-Text](https://arxiv.org/pdf/2503.07234)             | 2025 | NGSIM, HighD          | Language prompts         | 6,606 scenarios                      | Prediction             |
| [Urban-Text](https://arxiv.org/pdf/2503.07234)               | 2025 | MoCAD, ApolloScape    | Language prompts         | 5,431 samples                        | Prediction             |
| [MAPLM](https://github.com/LLVM-AD/MAPLM)                    | 2024 | THMA                  | Language prompts         | 2M scenarios, 2M prompts             | Perception             |
| [MAPLM-QA](https://github.com/LLVM-AD/MAPLM)                 | 2024 | THMA                  | QA                       | 14K scenarios, 61K QAs               | Perception             |
| [Rank2Tell](https://usa.honda-ri.com/rank2tell)              | 2024 | Self-collected        | Chain VQA                | 116 20-second clips                  | Perception             |
| [DRAMA](https://usa.honda-ri.com/drama)                      | 2023 | Self-collected        | Chain QA                 | 17,785 scenarios, 103K QAs           | Planning               |
| [SUP-AD](https://tsinghua-mars-lab.github.io/DriveVLM/)      | 2024 | Self-collected        | Language prompts         | 1,000 clips, 40+ categories          | Planning               |
| [LingoQA](https://github.com/wayveai/LingoQA)                | 2024 | Self-collected        | VQA                      | 28K videos, 419K annotations         | Perception             |
| [DriveMLM](https://github.com/OpenGVLab/DriveMLM)            | 2023 | CARLA                 | Decision, Explanation    | 50K routes, 30 scenarios             | End-to-end             |
| [LMDrive](https://github.com/opendilab/LMDrive)              | 2024 | CARLA                 | Navigation instructions  | 64K clips, 464K instructions         | End-to-end             |
| [DriveLM](https://github.com/OpenDriveLab/DriveLM)           | 2024 | CARLA, nuScenes       | Graph VQA                | 4,063 frames, 377K QAs               | End-to-end             |
| [DriveBench](https://huggingface.co/datasets/drive-bench/arena) | 2025 | DriveLM               | Graph VQA                | 19,200 frames, 20,498 QAs            | End-to-end             |
| [DriveCoT](https://drivecot.github.io/)                      | 2024 | CARLA                 | Reasoning process        | 1,058 scenarios, 36K samples         | End-to-end             |



# Citation

If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@misc{cui2025chainofthoughtautonomousdrivingcomprehensive,
      title={Chain-of-Thought for Autonomous Driving: A Comprehensive Survey and Future Prospects}, 
      author={Yixin Cui and Haotian Lin and Shuo Yang and Yixiao Wang and Yanjun Huang and Hong Chen},
      year={2025},
      eprint={2505.20223},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.20223}, 
}
```



