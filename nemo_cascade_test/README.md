---
license: apache-2.0
language:
- en
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- question-answering
tags:
- reasoning
- chain-of-thought
- instruction-tuning
- sft
configs:
  - config_name: General-Distillation
    data_files:
      - split: train
        path: "kimi-k2.5-main.jsonl"
  - config_name: PHD-Science
    data_files:
      - split: train
        path: "KimiK-2.5-PHD-Science.jsonl"
  - config_name: General-Math
    data_files:
      - split: train
        path: "kimiMath200k.jsonl"
  - config_name: MultilingualSTEM
    data_files:
      - split: train
        path: "MultilingualSTEM.jsonl"
---


 <p align="center">
 <img alt="arc" src="https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png" width="600">
 </p>


------------------------------------


<h1><span style="color:#1473df">KIMI-K2.5-1000000x</span></h1>

  - 1,000,000 reasoning traces distilled from ```KIMI-K2.5``` on ```high``` reasoning, (Each subset has different questions)


------------------------------------


-   <h1><span style="color:#1473df">Distribution:</span></h1>

     ```
     Coding: 50% (Includes: Webdev, Python, C++, Java, JS, C, Ruby, Lua, Rust, and C#)
     Science: 20% (Physics, Chemistry, Biology) - 100k more completions in the PHD-Science subset
     Math: 15% (Algebra, Calculus, Probability) - 200k more completions in kimiMath200k.jsonl
     Computer Science: 5%
     Logical Questions: 5%
     Creative Writing: 5%
     MultilingualSTEM: 100k completions inside of MultilingualSTEM.jsonl
     ```
<h2><span style="color:#1473df">Token Count</span>: <code>5B</code></h2>

<p style="margin-top:12px;font-size:11px;opacity:0.7">
  You can use this dataset for any purpose and you dont need to credit me, preferably dont claim it as your own. - 4/6/2026 with about 20GB of data and 5 billion tokens I will probably stop updating this
  
<br>


------------------------------------

 > [!IMPORTANT]
> ![image](https://cdn-uploads.huggingface.co/production/uploads/6909e3efe7497f3dbbe640ba/AT1xJ2uYHT7LjFsxewu-U.png)
  >  


------------------------------------


####      Data Collection

  - Collected using a modified Datagen by [TeichAI](https://huggingface.co/TeichAI) <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6837935ac3b7ffe0d2559ce9/-AxyvV4wfUY8uo87kNKkK.png" width="20" height="20" style="display: inline-block; vertical-align: middle; margin: 0 3px;">, over the course of about (80) hours


------------------------------------

####      hi - ianncity <img src="https://preview.redd.it/steam-happy-but-high-quality-v0-22ku6htw4u0c1.png?width=640&crop=smart&auto=webp&s=221735cb09dc3d4c1c7349e3187e752f6fe775e4" width="20" height="20" style="display: inline-block; vertical-align: middle; margin: 0 3px;">