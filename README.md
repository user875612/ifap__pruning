# Flow-Guided Neural Pruning: Continuum Dynamics Framework for Architecture-Agnostic Model Compression

**This codebase provides the implementation supporting our paper, "Flow-Guided Neural Pruning: Continuum Dynamics Framework for Architecture-Agnostic Model Compression" submitted to Machine Learning and Knowledge Extraction.**

This repository implements a pipeline for compressing neural network models using the IFAP (Iterative Flow-Aware Pruning) algorithm. The pipeline is designed to automate the process of pruning models across various model and dataset configurations.

## Flow Evaluation Scheme

![Model Architecture](media/architecture.png)


## Project Structure
```
main.py                       # Entry point for the pipeline
configs/
    model_configs.yaml        # Model weights paths configuration
    datasets.yaml             # Datasets configuration
ifap_pruning/
    process_pruning.py        # Main pruning logic
    ...                       # Other pruning-related modules
models/
    ...                       # Model definitions and utilities
utils/
    ...                       # Utility functions
visualization/
    ...                       # Visualization scripts
```

## Getting Started

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU
- Install required packages:

```bash
pip install -r requirements.txt
```

### Configuration
- Place your model weights paths in `configs/model_configs.yaml` under the `paths` key.
- List your datasets in `configs/datasets.yaml` under the `datasets` key.

Example `model_configs.yaml`:
```yaml
paths:
  - path/to/model1.pth
  - path/to/model2.pth
```

Example `datasets.yaml`:
```yaml
datasets:
  - dataset1
  - dataset2
```

### Running the Pipeline

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required datasets and place them in the appropriate directories as specified in `configs/datasets.yaml`.
4. Place your pretrained model weights in the directories specified in `configs/model_configs.yaml`.
5. Run the main pipeline script:
   ```bash
   python main.py
   ```
   Logs will be saved to `logs/ifap_pruning.log`.
6. After pruning and training, you can visualize the results using the scripts in the `visualization/` directory.

## Results

Below are the main results of our experiments.

Our experiments provide a comprehensive evaluation of the IFAP (Iterative Flow-Aware Pruning) algorithm across a wide range of neural network architectures and datasets.


<table border="2" style="width: 100%; border-collapse: collapse;">
  <caption style="font-weight: bold; margin-bottom: 10px;">Table I. Comparison of Our Pruning Method with SOTA Approaches on the CIFAR-10</caption>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Pruning Method</th>
      <th colspan="2">Accuracy</th>
    </tr>
    <tr>
      <th>Top-1</th>
      <th>Top-5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><strong>ResNet-50</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">98.29</td>
      <td style="background-color: #f7f7f7;">99.82</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>93.75</td>
      <td>99.04</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">93.23</td>
      <td style="background-color: #f7f7f7;">98.84</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>97.57</strong></td>
      <td><strong>99.72</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>EfficientNet-B4</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">96.41</td>
      <td style="background-color: #f7f7f7;">99.34</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>94.30</td>
      <td>99.12</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">93.77</td>
      <td style="background-color: #f7f7f7;">98.99</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>96.81</strong></td>
      <td><strong>99.49</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>ViT-Base/16</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">98.65</td>
      <td style="background-color: #f7f7f7;">98.83</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>91.72</td>
      <td>98.42</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">90.91</td>
      <td style="background-color: #f7f7f7;">98.18</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>98.16</strong></td>
      <td><strong>98.69</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>MobileNetV3-L</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">89.82</td>
      <td style="background-color: #f7f7f7;">92.47</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>91.99</td>
      <td>98.33</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">91.25</td>
      <td style="background-color: #f7f7f7;">98.15</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>89.26</strong></td>
      <td><strong>92.34</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>DenseNet-121</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">94.17</td>
      <td style="background-color: #f7f7f7;">99.07</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>93.42</td>
      <td>98.51</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">92.80</td>
      <td style="background-color: #f7f7f7;">98.59</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>94.13</strong></td>
      <td><strong>98.47</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>ConvNeXt-Small</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">94.18</td>
      <td style="background-color: #f7f7f7;">99.55</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>93.51</td>
      <td>99.13</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">94.29</td>
      <td style="background-color: #f7f7f7;">99.39</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>93.72</strong></td>
      <td><strong>99.24</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>VGG19 (BN)</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">93.54</td>
      <td style="background-color: #f7f7f7;">99.75</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>92.99</td>
      <td>99.23</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">91.06</td>
      <td style="background-color: #f7f7f7;">98.94</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>92.72</strong></td>
      <td><strong>99.22</strong></td>
    </tr>
    <tr>
      <td rowspan="4"><strong>ShuffleNet V2 x2.0</strong></td>
      <td style="background-color: #f7f7f7;">Baseline</td>
      <td style="background-color: #f7f7f7;">95.42</td>
      <td style="background-color: #f7f7f7;">98.82</td>
    </tr>
    <tr>
      <td>SWD</td>
      <td>90.23</td>
      <td>97.93</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">SFP</td>
      <td style="background-color: #f7f7f7;">90.38</td>
      <td style="background-color: #f7f7f7;">97.79</td>
    </tr>
    <tr>
      <td><strong>IFAP (Ours)</strong></td>
      <td><strong>94.87</strong></td>
      <td><strong>97.85</strong></td>
    </tr>
  </tbody>
</table>


<div style="overflow-x: auto; margin: 20px 0;">
<table border="2" style="width: 100%;">
  <caption style="font-weight: bold; font-size: 1.1em; margin-bottom: 10px; text-align: center;">
    Table II. Comparative pruning results across different architectures and datasets
  </caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th rowspan="2" style="text-align: left;">Dataset</th>
      <th rowspan="2" style="text-align: left;">Architecture</th>
      <th colspan="3">GFlops</th>
      <th colspan="3">Top-1 Acc</th>
    </tr>
    <tr style="background-color: #f2f2f2;">
      <th>Base</th>
      <th>Pruned</th>
      <th>↓%</th>
      <th>Base</th>
      <th>Pruned</th>
      <th>↓%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10" style="vertical-align: top; font-weight: bold;">CIFAR-10</td>
      <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">0.9</td><td style="background-color: #f7f7f7;">-68</td><td style="background-color: #f7f7f7;">94.20</td><td style="background-color: #f7f7f7;">94.03</td><td style="background-color: #f7f7f7;">-0.18</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>5.9</td><td>-70</td><td>93.46</td><td>92.91</td><td>-0.58</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.2</td><td style="background-color: #f7f7f7;">-71</td><td style="background-color: #f7f7f7;">98.19</td><td style="background-color: #f7f7f7;">97.63</td><td style="background-color: #f7f7f7;">-0.57</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>1.7</td><td>-70</td><td>95.33</td><td>94.87</td><td>-0.48</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-43</td><td style="background-color: #f7f7f7;">89.80</td><td style="background-color: #f7f7f7;">89.34</td><td style="background-color: #f7f7f7;">-0.52</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.1</td><td>-81</td><td>90.45</td><td>89.95</td><td>-0.55</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">2.3</td><td style="background-color: #f7f7f7;">-73</td><td style="background-color: #f7f7f7;">94.21</td><td style="background-color: #f7f7f7;">93.78</td><td style="background-color: #f7f7f7;">-0.45</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.5</td><td>-64</td><td>96.90</td><td>96.51</td><td>-0.41</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">2.9</td><td style="background-color: #f7f7f7;">-67</td><td style="background-color: #f7f7f7;">97.12</td><td style="background-color: #f7f7f7;">96.79</td><td style="background-color: #f7f7f7;">-0.33</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>5.1</td><td>-71</td><td>98.61</td><td>98.18</td><td>-0.44</td>
    </tr>
    <!-- CIFAR-100 -->
    <tr style="border-top: 2px solid #333;">
      <td rowspan="10" style="vertical-align: top; font-weight: bold;">CIFAR-100</td>
      <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">1.0</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">72.07</td><td style="background-color: #f7f7f7;">71.29</td><td style="background-color: #f7f7f7;">-1.07</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>6.8</td><td>-65</td><td>73.89</td><td>72.57</td><td>-1.79</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.3</td><td style="background-color: #f7f7f7;">-68</td><td style="background-color: #f7f7f7;">86.61</td><td style="background-color: #f7f7f7;">85.09</td><td style="background-color: #f7f7f7;">-1.76</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>1.9</td><td>-67</td><td>82.15</td><td>80.62</td><td>-1.86</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-54</td><td style="background-color: #f7f7f7;">77.70</td><td style="background-color: #f7f7f7;">76.93</td><td style="background-color: #f7f7f7;">-1.00</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-61</td><td>75.33</td><td>74.16</td><td>-1.55</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">2.5</td><td style="background-color: #f7f7f7;">-71</td><td style="background-color: #f7f7f7;">85.59</td><td style="background-color: #f7f7f7;">84.09</td><td style="background-color: #f7f7f7;">-1.75</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.7</td><td>-60</td><td>90.12</td><td>88.84</td><td>-1.42</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.2</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">88.44</td><td style="background-color: #f7f7f7;">87.13</td><td style="background-color: #f7f7f7;">-1.48</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>6.8</td><td>-61</td><td>94.24</td><td>93.16</td><td>-1.15</td>
    </tr>
    <!-- Fashion MNIST -->
    <tr style="border-top: 2px solid #333;">
      <td rowspan="10" style="vertical-align: top; font-weight: bold;">Fashion MNIST</td>
      <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">1.0</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">93.00</td><td style="background-color: #f7f7f7;">92.29</td><td style="background-color: #f7f7f7;">-0.77</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td>
      <td>19.6</td>
      <td>5.4</td>
      <td>-72</td>
      <td>91.77</td>
      <td>91.13</td>
      <td>-0.69</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td>
      <td style="background-color: #f7f7f7;">4.1</td>
      <td style="background-color: #f7f7f7;">1.1</td>
      <td style="background-color: #f7f7f7;">-73</td>
      <td style="background-color: #f7f7f7;">93.18</td>
      <td style="background-color: #f7f7f7;">92.73</td>
      <td style="background-color: #f7f7f7;">-0.45</td>
    </tr>
    <tr>
      <td>Inception-v3</td>
      <td>5.7</td>
      <td>1.6</td>
      <td>-72</td>
      <td>92.78</td>
      <td>92.30</td>
      <td>-0.52</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td>
      <td style="background-color: #f7f7f7;">0.2</td>
      <td style="background-color: #f7f7f7;">0.1</td>
      <td style="background-color: #f7f7f7;">-55</td>
      <td style="background-color: #f7f7f7;">89.80</td>
      <td style="background-color: #f7f7f7;">89.34</td>
      <td style="background-color: #f7f7f7;">-0.46</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>-78</td>
      <td>90.89</td>
      <td>90.31</td>
      <td>-0.58</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td>
      <td style="background-color: #f7f7f7;">8.6</td>
      <td style="background-color: #f7f7f7;">3.5</td>
      <td style="background-color: #f7f7f7;">-59</td>
      <td style="background-color: #f7f7f7;">89.66</td>
      <td style="background-color: #f7f7f7;">89.27</td>
      <td style="background-color: #f7f7f7;">-0.90</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.6</td><td>-62</td><td>94.97</td><td>94.54</td><td>-0.46</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td>
      <td style="background-color: #f7f7f7;">8.8</td>
      <td style="background-color: #f7f7f7;">2.8</td>
      <td style="background-color: #f7f7f7;">-68</td>
      <td style="background-color: #f7f7f7;">95.35</td>
      <td style="background-color: #f7f7f7;">94.88</td>
      <td style="background-color: #f7f7f7;">-0.49</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>6.4</td><td>-63</td><td>94.80</td><td>94.30</td><td>-0.47</td>
    </tr>
    <!-- ImageNet -->
    <tr style="border-top: 2px solid #333;">
      <td rowspan="10" style="vertical-align: top; font-weight: bold;">ImageNet</td>
      <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">1.0</td><td style="background-color: #f7f7f7;">-65</td><td style="background-color: #f7f7f7;">74.65</td><td style="background-color: #f7f7f7;">73.96</td><td style="background-color: #f7f7f7;">-0.92</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>7.2</td><td>-63</td><td>73.12</td><td>71.62</td><td>-1.50</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.6</td><td style="background-color: #f7f7f7;">-61</td><td style="background-color: #f7f7f7;">76.14</td><td style="background-color: #f7f7f7;">75.19</td><td style="background-color: #f7f7f7;">-1.25</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>2.1</td><td>-63</td><td>77.16</td><td>75.86</td><td>-1.69</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td>
      <td style="background-color: #f7f7f7;">-53</td><td style="background-color: #f7f7f7;">74.03</td><td style="background-color: #f7f7f7;">72.83</td><td style="background-color: #f7f7f7;">-1.63</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-60</td><td>76.22</td><td>74.85</td><td>-1.37</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">2.6</td><td style="background-color: #f7f7f7;">-70</td><td style="background-color: #f7f7f7;">82.06</td><td style="background-color: #f7f7f7;">80.99</td><td style="background-color: #f7f7f7;">-1.82</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.6</td><td>-62</td><td>83.37</td><td>81.71</td><td>-1.66</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.4</td><td style="background-color: #f7f7f7;">-61</td><td style="background-color: #f7f7f7;">84.21</td><td style="background-color: #f7f7f7;">82.82</td><td style="background-color: #f7f7f7;">-1.66</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>6.9</td><td>-61</td><td>80.17</td><td>79.85</td><td>-1.51</td>
    </tr>
    <tr style="border-top: 2px solid #333;">
      <td rowspan="10" style="vertical-align: top; font-weight: bold;">Stanford Cars</td>
      <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">1.0</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">83.13</td><td style="background-color: #f7f7f7;">81.95</td><td style="background-color: #f7f7f7;">-1.18</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>7.1</td><td>-64</td><td>86.99</td><td>85.70</td><td>-1.29</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.4</td><td style="background-color: #f7f7f7;">-66</td><td style="background-color: #f7f7f7;">92.52</td><td style="background-color: #f7f7f7;">90.99</td><td style="background-color: #f7f7f7;">-1.53</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>2.0</td><td>-65</td><td>83.86</td><td>82.27</td><td>-1.67</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-45</td><td style="background-color: #f7f7f7;">68.32</td><td style="background-color: #f7f7f7;">67.43</td><td style="background-color: #f7f7f7;">-1.29</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-57</td><td>82.56</td><td>81.39</td><td>-1.42</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">2.9</td><td style="background-color: #f7f7f7;">-66</td><td style="background-color: #f7f7f7;">86.22</td><td style="background-color: #f7f7f7;">81.21</td><td style="background-color: #f7f7f7;">-1.18</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.6</td><td>-60</td><td>91.34</td><td>90.06</td><td>-1.28</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.3</td><td style="background-color: #f7f7f7;">-63</td><td style="background-color: #f7f7f7;">90.24</td><td style="background-color: #f7f7f7;">89.61</td><td style="background-color: #f7f7f7;">-0.63</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>6.9</td><td>-61</td><td>93.73</td><td>92.21</td><td>-1.62</td>
    </tr>    
    <tr style="border-top: 2px solid #333;">
      <td rowspan="10" style="vertical-align: top; font-weight: bold;">iNaturalist</td>
      <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">1.0</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">69.75</td><td style="background-color: #f7f7f7;">68.73</td><td style="background-color: #f7f7f7;">-1.46</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>6.9</td><td>-65</td><td>67.20</td><td>65.43</td><td>-1.77</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.6</td><td style="background-color: #f7f7f7;">-61</td><td style="background-color: #f7f7f7;">76.15</td><td style="background-color: #f7f7f7;">74.68</td><td style="background-color: #f7f7f7;">-1.93</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>2.1</td><td>-63</td><td>72.35</td><td>71.13</td><td>-1.22</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-45</td><td style="background-color: #f7f7f7;">68.31</td><td style="background-color: #f7f7f7;">67.43</td><td style="background-color: #f7f7f7;">-1.29</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-58</td><td>66.77</td><td>65.58</td><td>-1.78</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">3.4</td><td style="background-color: #f7f7f7;">-60</td><td style="background-color: #f7f7f7;">68.90</td><td style="background-color: #f7f7f7;">67.43</td><td style="background-color: #f7f7f7;">-1.47</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.6</td><td>-62</td><td>70.58</td><td>68.64</td><td>-1.28</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.4</td><td style="background-color: #f7f7f7;">-63</td><td style="background-color: #f7f7f7;">74.30</td><td style="background-color: #f7f7f7;">73.10</td><td style="background-color: #f7f7f7;">-1.20</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>6.9</td><td>-61</td><td>68.66</td><td>67.83</td><td>-1.21</td>
    </tr>
  <tr style="border-top: 2px solid #333;">
    <td rowspan="10" style="vertical-align: top; font-weight: bold;">Food101</td>
    <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">0.9</td><td style="background-color: #f7f7f7;">-68</td><td style="background-color: #f7f7f7;">87.35</td><td style="background-color: #f7f7f7;">85.70</td><td style="background-color: #f7f7f7;">-1.89</td>
  </tr>
  <tr>
    <td>VGG19 (BN)</td><td>19.6</td><td>6.3</td><td>-68</td><td>86.32</td><td>84.32</td><td>-1.57</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.5</td><td style="background-color: #f7f7f7;">-63</td><td style="background-color: #f7f7f7;">90.46</td><td style="background-color: #f7f7f7;">89.30</td><td style="background-color: #f7f7f7;">-1.16</td>
  </tr>
  <tr>
    <td>Inception-v3</td><td>5.7</td><td>2.0</td><td>-65</td><td>88.12</td><td>86.78</td><td>-1.52</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-53</td><td style="background-color: #f7f7f7;">86.05</td><td style="background-color: #f7f7f7;">83.75</td><td style="background-color: #f7f7f7;">-2.30</td>
  </tr>
  <tr>
    <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-58</td><td>84.23</td><td>82.84</td><td>-1.39</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">3.6</td><td style="background-color: #f7f7f7;">-58</td><td style="background-color: #f7f7f7;">86.25</td><td style="background-color: #f7f7f7;">83.75</td><td style="background-color: #f7f7f7;">-2.50</td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td><td>4.2</td><td>1.6</td><td>-62</td><td>90.36</td><td>89.05</td><td>-1.31</td>
  </tr>
  <tr>
    <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.2</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">90.56</td><td style="background-color: #f7f7f7;">89.29</td><td style="background-color: #f7f7f7;">-1.40</td>
  </tr>
  <tr>
    <td>ViT-Base/16</td><td>17.5</td><td>6.7</td><td>-62</td><td>87.44</td><td>86.14</td><td>-1.45</td>
  </tr>
  <tr style="border-top: 2px solid #333;">
    <td rowspan="10" style="vertical-align: top; font-weight: bold;">Oxford-IIIT Pet</td>
    <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">0.8</td><td style="background-color: #f7f7f7;">-71</td><td style="background-color: #f7f7f7;">85.24</td><td style="background-color: #f7f7f7;">84.20</td><td style="background-color: #f7f7f7;">-1.22</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>6.7</td><td>-66</td><td>85.20</td><td>85.20</td><td>-1.44</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.6</td><td style="background-color: #f7f7f7;">-61</td><td style="background-color: #f7f7f7;">93.11</td><td style="background-color: #f7f7f7;">92.88</td><td style="background-color: #f7f7f7;">-0.26</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>1.9</td><td>-66</td><td>89.34</td><td>88.17</td><td>-1.31</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-45</td><td style="background-color: #f7f7f7;">85.53</td><td style="background-color: #f7f7f7;">84.52</td><td style="background-color: #f7f7f7;">-1.19</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-63</td><td>83.67</td><td>82.43</td><td>-1.49</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">3.1</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">84.08</td><td style="background-color: #f7f7f7;">83.00</td><td style="background-color: #f7f7f7;">-1.29</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.7</td><td>-60</td><td>87.85</td><td>86.98</td><td>-0.99</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.6</td><td style="background-color: #f7f7f7;">-59</td><td style="background-color: #f7f7f7;">88.36</td><td style="background-color: #f7f7f7;">88.19</td><td style="background-color: #f7f7f7;">-0.17</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>5.9</td><td>-66</td><td>89.58</td><td>88.21</td><td>-1.37</td>
    </tr>
  <tr style="border-top: 2px solid #333;">
    <td rowspan="10" style="vertical-align: top; font-weight: bold;">FER2013</td>
    <td style="background-color: #f7f7f7;">DenseNet-121</td><td style="background-color: #f7f7f7;">2.8</td><td style="background-color: #f7f7f7;">0.9</td><td style="background-color: #f7f7f7;">-68</td><td style="background-color: #f7f7f7;">65.13</td><td style="background-color: #f7f7f7;">64.17</td><td style="background-color: #f7f7f7;">-1.47</td>
    </tr>
    <tr>
      <td>VGG19 (BN)</td><td>19.6</td><td>6.8</td><td>-65</td><td>68.34</td><td>67.10</td><td>-1.81</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ResNet-50</td><td style="background-color: #f7f7f7;">4.1</td><td style="background-color: #f7f7f7;">1.4</td><td style="background-color: #f7f7f7;">-66</td><td style="background-color: #f7f7f7;">71.81</td><td style="background-color: #f7f7f7;">70.66</td><td style="background-color: #f7f7f7;">-1.60</td>
    </tr>
    <tr>
      <td>Inception-v3</td><td>5.7</td><td>2.0</td><td>-65</td><td>70.46</td><td>69.11</td><td>-1.91</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">MobileNetV3-L</td><td style="background-color: #f7f7f7;">0.2</td><td style="background-color: #f7f7f7;">0.1</td><td style="background-color: #f7f7f7;">-42</td><td style="background-color: #f7f7f7;">69.88</td><td style="background-color: #f7f7f7;">67.43</td><td style="background-color: #f7f7f7;">-2.45</td>
    </tr>
    <tr>
      <td>ShuffleNetV2 x2.0</td><td>0.5</td><td>0.2</td><td>-58</td><td>67.45</td><td>66.25</td><td>-1.20</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">ConvNeXt-Small</td><td style="background-color: #f7f7f7;">8.6</td><td style="background-color: #f7f7f7;">3.0</td><td style="background-color: #f7f7f7;">-64</td><td style="background-color: #f7f7f7;">76.10</td><td style="background-color: #f7f7f7;">74.84</td><td style="background-color: #f7f7f7;">-1.26</td>
    </tr>
    <tr>
      <td>EfficientNet-B4</td><td>4.2</td><td>1.6</td><td>-62</td><td>74.16</td><td>73.27</td><td>-0.89</td>
    </tr>
    <tr>
      <td style="background-color: #f7f7f7;">EfficientNet V2-S</td><td style="background-color: #f7f7f7;">8.8</td><td style="background-color: #f7f7f7;">3.3</td><td style="background-color: #f7f7f7;">-62</td><td style="background-color: #f7f7f7;">76.89</td><td style="background-color: #f7f7f7;">75.48</td><td style="background-color: #f7f7f7;">-1.83</td>
    </tr>
    <tr>
      <td>ViT-Base/16</td><td>17.5</td><td>6.3</td><td>-64</td><td>70.21</td><td>68.97</td><td>-1.76</td>
    </tr>

<table border="2" cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 100%;">
  <caption style="font-weight: bold; margin-bottom: 10px;">Table III. Our method applied to ResNet-50 on Cifar-10 over 30 pruning iterations</caption>
  <thead style="background-color: #f2f2f2; font-weight: bold;">
    <tr>
      <th style="padding: 8px;">Pruning Step</th>
      <th style="padding: 8px;">Params (M)</th>
      <th style="padding: 8px;">GFlops</th>
      <th style="padding: 8px;">Top-1 Acc. (%)</th>
      <th style="padding: 8px;">Top-5 Acc. (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #f7f7f7;"><td>1</td><td>23.53</td><td>4.09</td><td>98.20</td><td>99.86</td></tr>
    <tr><td>2</td><td>22.27</td><td>3.89</td><td>97.66</td><td>99.85</td></tr>
    <tr style="background-color: #f7f7f7;"><td>3</td><td>21.20</td><td>3.66</td><td>97.23</td><td>99.84</td></tr>
    <tr><td>4</td><td>19.89</td><td>3.46</td><td>96.99</td><td>99.73</td></tr>
    <tr style="background-color: #f7f7f7;"><td>5</td><td>18.78</td><td>3.31</td><td>97.11</td><td>99.89</td></tr>
    <tr><td>6</td><td>17.54</td><td>3.13</td><td>97.74</td><td>99.89</td></tr>
    <tr style="background-color: #f7f7f7;"><td>7</td><td>16.45</td><td>2.90</td><td>97.62</td><td>99.84</td></tr>
    <tr><td>8</td><td>15.50</td><td>2.73</td><td>97.93</td><td>99.87</td></tr>
    <tr style="background-color: #f7f7f7;"><td>9</td><td>14.62</td><td>2.61</td><td>98.09</td><td>99.76</td></tr>
    <tr><td>10</td><td>14.14</td><td>2.52</td><td>98.05</td><td>99.75</td></tr>
    <tr style="background-color: #f7f7f7;"><td>11</td><td>13.50</td><td>2.37</td><td>97.87</td><td>99.77</td></tr>
    <tr><td>12</td><td>12.98</td><td>2.26</td><td>97.85</td><td>99.81</td></tr>
    <tr style="background-color: #f7f7f7;"><td>13</td><td>12.37</td><td>2.15</td><td>97.84</td><td>99.77</td></tr>
    <tr><td>14</td><td>11.82</td><td>2.08</td><td>97.77</td><td>99.79</td></tr>
    <tr style="background-color: #f7f7f7;"><td>15</td><td>11.26</td><td>1.98</td><td>97.70</td><td>99.76</td></tr>
    <tr><td>16</td><td>11.02</td><td>1.94</td><td>97.85</td><td>99.80</td></tr>
    <tr style="background-color: #f7f7f7;"><td>17</td><td>10.77</td><td>1.89</td><td>97.56</td><td>99.81</td></tr>
    <tr><td>18</td><td>10.53</td><td>1.85</td><td>97.50</td><td>99.79</td></tr>
    <tr style="background-color: #f7f7f7;"><td>19</td><td>10.28</td><td>1.81</td><td>97.42</td><td>99.80</td></tr>
    <tr><td>20</td><td>10.04</td><td>1.77</td><td>97.35</td><td>99.78</td></tr>
    <tr style="background-color: #f7f7f7;"><td>21</td><td>9.79</td><td>1.73</td><td>97.28</td><td>99.75</td></tr>
    <tr><td>22</td><td>9.55</td><td>1.68</td><td>97.50</td><td>99.77</td></tr>
    <tr style="background-color: #f7f7f7;"><td>23</td><td>9.30</td><td>1.49</td><td>97.52</td><td>99.78</td></tr>
    <tr><td>24</td><td>9.05</td><td>1.45</td><td>97.08</td><td>99.77</td></tr>
    <tr style="background-color: #f7f7f7;"><td>25</td><td>8.81</td><td>1.40</td><td>97.50</td><td>99.80</td></tr>
    <tr><td>26</td><td>8.56</td><td>1.34</td><td>97.40</td><td>99.81</td></tr>
    <tr style="background-color: #f7f7f7;"><td>27</td><td>8.32</td><td>1.30</td><td>96.91</td><td>99.79</td></tr>
    <tr><td>28</td><td>8.07</td><td>1.26</td><td>97.25</td><td>99.78</td></tr>
    <tr style="background-color: #f7f7f7;"><td>29</td><td>7.83</td><td>1.22</td><td>97.52</td><td>99.80</td></tr>
    <tr><td>30</td><td>7.57</td><td>1.19</td><td>97.63</td><td>99.81</td></tr>
  </tbody>
</table>

## IFAP Pruning Dynamics: ResNet-50 on CIFAR-10

Below are the main plots showing the dynamics of pruning ResNet-50 on CIFAR-10 over 30 iterations:

![Total Parameters](media/Total%20Parameters%20over%2030%20iterations.%20ResNet50%20on%20CIFAR-10.png)

![Training GFlops](media/Training%20GFlops%20over%2030%20iterations.%20ResNet50%20on%20CIFAR-10.png)

![Top-1 Accuracy](media/Top-1%20Accuracy%20over%2030%20iterations.%20ResNet50%20on%20CIFAR-10.png)

![Top-5 Accuracy](media/Top-5%20Accuracy%20over%2030%20iterations.%20ResNet50%20on%20CIFAR-10.png)

Each plot corresponds to the respective metric as a function of the pruning step.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the torchvision team for their open-source model zoo, which was instrumental for our experiments. We also acknowledge the authors of "DepGraph: Towards any structural pruning" for providing their code for structural pruning, which inspired parts of this project.

