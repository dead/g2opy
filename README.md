# g2opy

Windows: [![Build status](https://ci.appveyor.com/api/projects/status/sbndgs89mvxecmsh/branch/master?svg=true)](https://ci.appveyor.com/project/dead/g2opy/branch/master)

This is a python binding of graph optimization C++ framework [g2o](https://github.com/RainerKuemmerle/g2o).

> g2o is an open-source C++ framework for optimizing graph-based nonlinear error functions. g2o has been designed to be easily extensible to a wide range of problems and a new problem typically can be specified in a few lines of code. The current implementation provides solutions to several variants of SLAM and BA.  
A wide range of problems in robotics as well as in computer-vision involve the minimization of a non-linear error function that can be represented as a graph. Typical instances are simultaneous localization and mapping (SLAM) or bundle adjustment (BA). The overall goal in these problems is to find the configuration of parameters or state variables that maximally explain a set of measurements affected by Gaussian noise. g2o is an open-source C++ framework for such nonlinear least squares problems. g2o has been designed to be easily extensible to a wide range of problems and a new problem typically can be specified in a few lines of code. The current implementation provides solutions to several variants of SLAM and BA.

Currently, this project doesn't support writing user-defined types in python, but the predefined types are enough to implement the most common algorithms, say **PnP, ICP, Bundle Adjustment and Pose Graph Optimization** in 2d or 3d scenarios. g2o's visualization part is not wrapped, if you want to visualize point clouds or graph, you can give [pangolin](https://github.com/uoip/pangolin) a try, it's a python binding of C++ library [Pangolin](http://github.com/stevenlovegrove/Pangolin).

For convenience, some frequently used Eigen types (Quaternion, Rotation2d, Isometry3d, Isometry2d, AngleAxis) are packed into this library.  
In the contrib folder, I collected some useful 3rd-party C++ code related to g2o, like robust pose graph optimization library [vertigo](http://www.openslam.org/vertigo), stereo sba and smooth estimate propagator from [sptam](https://github.com/lrse/sptam).


## Requirements
* [C++ requirements](#g2oRequirements).   
([pybind11](https://github.com/pybind/pybind11) is also required, but it's built in this repository, you don't need to install) 


## Installation
```
git clone https://github.com/dead/g2opy.git
cd g2opy
python setup.py install
```
Tested under Windows 10, Python 3.6+.


## Get Started
The code snippets below show the core parts of BA and Pose Graph Optimization in a SLAM system.
#### Bundle Adjustment
```python
import numpy
import g2opy as g2o

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()
```

#### Pose Graph Optimization
```python
import numpy
import g2opy as g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()
```
For more details, checkout [python examples](examples/). 
Thanks to [pybind11](https://github.com/pybind/pybind11), g2opy works seamlessly between numpy and underlying Eigen.  


## Motivation
This project is my first step towards implementing complete SLAM system in python, and interacting with Deep Learning models.  
Deep Learning is the hottest field in AI nowadays, it has greatly benefited many Robotics/Computer Vision tasks, like
* Reinforcement Learning
* Self-Supervision
* Control
* Object Tracking
* Object Detection
* Semantic Segmentation
* Instance Segmentation
* Place Recognition
* Face Recognition
* 3D Object Detection
* Point Cloud Segmentation
* Human Pose Estimation
* Stereo Matching
* Depth Estimation
* Optical Flow Estimation
* Interest Point Detection
* Correspondence Estimation
* Image Enhancement 
* Style Transfer
* ...

SLAM, as a subfield of Robotics and Computer Vision, is one of the core modules of robots, MAV, autonomous driving, and augmented reality. The combination of SLAM and Deep Learning (and Deep Learning driving computer vision techniques) is very promising, actually, there are increasing work in this direction, e.g. [CNN-SLAM](https://arxiv.org/abs/1704.03489), [SfM-Net](https://arxiv.org/abs/1704.07804), [DeepVO](https://arxiv.org/abs/1709.08429), [DPC-Net](https://arxiv.org/abs/1709.03128), [MapNet](https://arxiv.org/abs/1712.03342), [SuperPoint](https://arxiv.org/abs/1712.07629).   
Deep Learning community has developed many easy-to-use python libraries, like [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch), [Chainer](https://github.com/chainer/chainer), [MXNet](https://mxnet.incubator.apache.org/). These libraries make writing/training DL models easier, and in turn boost the development of the field itself. But in SLAM/Robotics fields, python is still underrated, most of the software stacks are writen for C/C++ users. Lacking of tools makes it inconvenient to interact with the booming Deep Learning comunity and python scientific computing ecosystem.     
Hope this project can slightly relieve the situation.


## TODO
* Installation via pip;
* Solve the found segfault bugs (be easy, they do not appear in the python examples);
* Introduce **Automatic Differentiation**, make writing user-defined types in python possible.


## License
* For g2o's original C++ code, see [License](#g2oLicense).
* The binding code and python example code of this project is licensed under BSD License.  


## Contact
If you have problems related to **binding code/python interface/python examples** of this project, feel free to open an issue.
