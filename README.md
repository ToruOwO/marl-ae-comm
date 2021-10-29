# Learning to Ground Multi-Agent Communication with Autoencoders

This repo contains the PyTorch implementation for all models and environments in the paper [Learning to Ground Multi-Agent Communication with Autoencoders](https://toruowo.github.io/marl-ae-comm/)

by [Toru Lin](https://toruowo.github.io/), [Minyoung Huh](http://minyounghuh.com/), [Chris Stauffer](https://scholar.google.com/citations?user=QdFrJOMAAAAJ&hl=en), [Sernam Lim](https://scholar.google.com/citations?user=HX0BfLYAAAAJ&hl=en), and [Phillip Isola](http://web.mit.edu/phillipi/).

### Code layout

Please see each sub-directory for more details.


| Directory          | Detail |
| :-------------: |:-------------:|
| cifar-game | environment and models for training "CIFAR Game" |
| :-------------: |:-------------:|
| marl-grid/env | environments for training "FindGoal" and "RedBlueDoors" | 
| marl-grid/find-goal | models for training "FindGoal" |
| marl-grid/red-blue-doors | models for training "RedBlueDoors" | 

### Paper citation

If you used this code or found our work helpful, please consider citing:

<pre>
@misc{lin2021learning,
      title={Learning to Ground Multi-Agent Communication with Autoencoders}, 
      author={Toru Lin and Minyoung Huh and Chris Stauffer and Ser-Nam Lim and Phillip Isola},
      year={2021},
      eprint={2110.15349},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>
