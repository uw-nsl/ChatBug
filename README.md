<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://scholar.google.com/citations?user=kTXY8P0AAAAJ&hl=en" target="_blank" style="text-decoration: none;">Fengqing Jiang<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="https://zhangchenxu.com/" target="_blank" style="text-decoration: none;">Zhangchen Xu<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="https://luyaoniu.github.io/" target="_blank" style="text-decoration: none;">Luyao Niu<sup>1,*</sup></a>&nbsp;,&nbsp;<br>
    <a href="https://yuchenlin.xyz/" target="_blank" style="text-decoration: none;">Bill Yuchen Lin<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://labs.ece.uw.edu/nsl/faculty/radha/" target="_blank" style="text-decoration: none;">Radha Poovendran<sup>1</sup></a>&nbsp;&nbsp;
    <br/> <br>
<sup>1</sup>University of Washington&nbsp;&nbsp;&nbsp;<sup>2</sup>Allen Institute for AI&nbsp;&nbsp;&nbsp;<br><sup>*</sup>Equal Contribution<br/>
</p>

<p align='center' style='color: red;';>
<b>
<em>Warning: This project contains model outputs that may be considered offensive</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2406.12935v1" target="_blank" style="text-decoration: none;">[arXiv]</a>
</b>
</p>

## Overview
![](asset/chatbug.jpg)

## Usage
### Setup Environment
```
bash build_env.sh chatbug
```


### Run with Chatbug

```
python chatbug.py
```
You can set up the ```attack.yaml``` or run with cmd args to config the experiments.


## Citation
If you find our project useful in your research, please consider citing:

```
@misc{jiang2024chatbug,
      title={ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates}, 
      author={Fengqing Jiang and Zhangchen Xu and Luyao Niu and Bill Yuchen Lin and Radha Poovendran},
      year={2024},
      eprint={2406.12935},
      archivePrefix={arXiv}
}
```